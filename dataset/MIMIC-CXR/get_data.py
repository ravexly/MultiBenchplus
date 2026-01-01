import os
import zipfile
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from transformers import BertTokenizer
# Define the 14 labels for the CheXpert competition
CHEXPERT_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
    'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax', 'Support Devices', 'No Finding'
]
def create_collate_fn(tokenizer):
    def collate_fn(batch):
        images, reports, labels = zip(*batch)
        images_tensor = torch.stack(images, dim=0)
        

        encoded_inputs = tokenizer.batch_encode_plus(
            list(reports),         
            padding='longest',     
            truncation=True,      
            max_length=512,         
            return_tensors='pt'     
        )
        
   
        labels_tensor = torch.stack(labels, dim=0)
        

        return [images_tensor, encoded_inputs, labels_tensor]

    return collate_fn

class _MIMICCXRJPGDataset(Dataset):
    """Internal Dataset class for MIMIC-CXR-JPG."""
    def __init__(self, dataframe, image_root, report_dict, transform, uncertainty_strategy):
        self.data = dataframe
        self.image_root = image_root
        self.transform = transform
        self.report_dict = report_dict
        
        # Mapping for one-hot encoding ViewPosition
        self.view_position_map = {'AP': 0, 'PA': 1, 'LATERAL': 2, 'LL': 2, 'RL': 2, 'UNK': 3}
        
        # Process labels based on uncertainty strategy
        # self.labels = self.data[CHEXPERT_LABELS].values.astype(np.float32)
        self.labels = self.data[CHEXPERT_LABELS].fillna(0).values.astype(np.float32)
        self.mask = np.ones_like(self.labels)

        if uncertainty_strategy == 'zero':
            self.labels[self.labels == -1] = 0
        elif uncertainty_strategy == 'one':
            self.labels[self.labels == -1] = 1
        elif uncertainty_strategy == 'ignore':
            self.mask = (self.labels != -1).astype(np.float32)
            self.labels[self.labels == -1] = 0 # Set to 0 but will be masked

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Image
        image_path = os.path.join(self.image_root, row['Path'])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # Text Report
        report = self.report_dict.get(str(row['study_id']), '')
        
        # Metadata (ViewPosition)
        # view_pos = row.get('ViewPosition', 'UNK').upper()
        # view_pos_id = self.view_position_map.get(view_pos, 3) # Default to UNK
        # view_pos_onehot = torch.nn.functional.one_hot(torch.tensor(view_pos_id), num_classes=4).float()
        
        # Labels and Mask
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        mask = torch.tensor(self.mask[idx], dtype=torch.float32)

        return image, report,  label

def _load_mimic_cxr_dataframe(root_dir, split):
    """Helper to load and merge MIMIC-CXR CSVs for a given split."""
    chexpert_csv = os.path.join(root_dir, 'mimic-cxr-2.0.0-chexpert.csv.gz')
    metadata_csv = os.path.join(root_dir, 'mimic-cxr-2.0.0-metadata.csv.gz')
    split_csv = os.path.join(root_dir, 'mimic-cxr-2.0.0-split.csv.gz')
    
    chexpert_df = pd.read_csv(chexpert_csv)
    metadata_df = pd.read_csv(metadata_csv)
    split_df = pd.read_csv(split_csv)

    # Merge dataframes
    merged = metadata_df.merge(chexpert_df, on=['subject_id', 'study_id'])
    merged = merged.merge(split_df[['dicom_id', 'split']], on='dicom_id')
    
    # Filter by split and drop rows where all labels are NaN
    split_df = merged[merged['split'] == split]
    split_df = split_df.dropna(subset=CHEXPERT_LABELS, how='all')

    # Create image path column
    split_df['Path'] = split_df.apply(
        lambda row: f"p{str(row['subject_id'])[:2]}/p{row['subject_id']}/s{row['study_id']}/{row['dicom_id']}.jpg",
        axis=1
    )
    return split_df.reset_index(drop=True)

def _load_report_dict(report_file):
    """Helper to load radiology reports from the zipped archive."""
    report_dict = {}
    with zipfile.ZipFile(report_file, 'r') as archive:
        for name in archive.namelist():
            if name.endswith('.txt'):
                study_id = os.path.splitext(os.path.basename(name))[0].lstrip('s')
                with archive.open(name) as f:
                    report = f.read().decode('utf-8')
                    report_dict[str(int(study_id))] = report
    return report_dict

def get_loader(
    root_dir='../../data/mimic-cxr',
    batch_size=8,
    num_workers=4,
    uncertainty_strategy='ignore',
    image_size=224,
    max_samples={'train': 5000}
):
    """
    Creates and returns data loaders for the MIMIC-CXR-JPG dataset.
    This is a multi-modal, multi-label dataset.

    Args:
        root_dir (str): Path to the root MIMIC-CXR directory.
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of workers for the dataloaders.
        uncertainty_strategy (str): How to handle uncertain labels (-1). 
                                    Options: 'ignore', 'zero', 'one'.
        image_size (int): The size to resize images to.
        max_samples (dict, optional): Max samples to use for each split. 
                                      e.g., {"train": 1000, "val": 200, "test": 200}

    Returns:
        tuple: (train_loader, val_loader, test_loader, n_classes).
               Each loader yields (image, report_text, view_position, labels, mask).
    """
    image_root = os.path.join(root_dir, 'files')
    report_zip = os.path.join(root_dir, 'mimic-cxr-reports.zip')
    
    print("Loading text reports from zip file...")
    report_dict = _load_report_dict(report_zip)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    n_classes = len(CHEXPERT_LABELS)

    # --- Create Datasets ---
    print("Loading dataframes for splits...")
    train_df = _load_mimic_cxr_dataframe(root_dir, split='train')
    val_df = _load_mimic_cxr_dataframe(root_dir, split='validate')
    test_df = _load_mimic_cxr_dataframe(root_dir, split='test')
    
    # Subsample if max_samples is specified
    if max_samples:
        if 'train' in max_samples:
            train_df = train_df.sample(n=max_samples['train'], random_state=42)
        if 'val' in max_samples:
            val_df = val_df.sample(n=max_samples['val'], random_state=42)
        if 'test' in max_samples:
            test_df = test_df.sample(n=max_samples['test'], random_state=42)

    print("Initializing PyTorch Datasets...")
    train_set = _MIMICCXRJPGDataset(train_df, image_root, report_dict, transform, uncertainty_strategy)
    val_set = _MIMICCXRJPGDataset(val_df, image_root, report_dict, transform, uncertainty_strategy)
    test_set = _MIMICCXRJPGDataset(test_df, image_root, report_dict, transform, uncertainty_strategy)
    
    # --- Log Sample Counts ---
    with open("samples.txt", "w") as f:
        f.write(f"Dataset: MIMIC-CXR-JPG\n")
        f.write("="*20 + "\n")
        f.write(f"Train samples: {len(train_set)}\n")
        f.write(f"Val samples: {len(val_set)}\n")
        f.write(f"Test samples: {len(test_set)}\n")
    tokenizer =  BertTokenizer.from_pretrained("../../bert-base-uncased")
    collate_function = create_collate_fn(tokenizer)
    # --- Create DataLoaders ---
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(n_classes)
    return train_loader, val_loader, test_loader, n_classes

