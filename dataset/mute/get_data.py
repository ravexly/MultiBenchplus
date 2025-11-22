
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

class MUTEDataset(Dataset):
    """
    Custom Dataset for the MUTE dataset.
    Handles loading images, text, and labels from an XLSX file.
    """
    def __init__(self, xlsx_file, root_dir, transform=None):
        """
        Args:
            xlsx_file (string): Path to the xlsx file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = pd.read_excel(xlsx_file)
        self.root_dir = root_dir
        self.transform = transform

        self.label_map = {'not-hate': 0, 'hate': 1}
        # The label mapping should use the actual column name from your file, e.g., 'Label'
        # If you are not sure, you can print(self.annotations.columns) here to check.
        label_column_name = self.annotations.columns[2] # Assumes label is in the 3rd column
        self.annotations = self.annotations[self.annotations[label_column_name].isin(self.label_map.keys())].copy()
        self.annotations[label_column_name] = self.annotations[label_column_name].map(self.label_map)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # --- FIX: Reverted to using iloc to access data by column position ---
        # This respects your original structure and fixes the KeyError.
        # It assumes Column 0: image_name, 1: caption, 2: label
        img_name = self.annotations.iloc[idx, 0]
        caption = self.annotations.iloc[idx, 1]
        label = self.annotations.iloc[idx, 2]
        # --- End of Fix ---

        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except (FileNotFoundError, IOError):
            print(f"Warning: Image file not found {img_path}. Using a placeholder.")
            image = torch.zeros((3, 224, 224))

        if not isinstance(caption, str):
            caption = ""

        label = torch.tensor(label, dtype=torch.long)

        return image, caption, label


def get_loader(
    root_dir='../../data/MUTE/Memes/', 
    train_file='../../data/MUTE/train_hate.xlsx', 
    val_file='../../data/MUTE/valid_hate.xlsx', 
    test_file='../../data/MUTE/test_hate.xlsx', 
    bert_model_path="../../bert-base-uncased",
    batch_size=8, 
    num_workers=4
):
    """
    Creates and returns the data loaders for the MUTE dataset.
    This version tokenizes text in batches and returns a list.
    """
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = MUTEDataset(xlsx_file=train_file, root_dir=root_dir, transform=transform)
    val_dataset = MUTEDataset(xlsx_file=val_file, root_dir=root_dir, transform=transform)
    test_dataset = MUTEDataset(xlsx_file=test_file, root_dir=root_dir, transform=transform)

    train_len, val_len, test_len = len(train_dataset), len(val_dataset), len(test_dataset)
    total_len = train_len + val_len + test_len
    
    with open("samples.txt", "w") as f:
        f.write(f"Dataset: MUTE\n")
        f.write("="*20 + "\n")
        f.write(f"Train samples: {train_len}\n")
        f.write(f"Validation samples: {val_len}\n")
        f.write(f"Test samples: {test_len}\n")
        f.write(f"Total samples: {total_len}\n")

    n_classes = len(train_dataset.label_map)

    def custom_collate_fn(batch):
        images, captions, labels = zip(*batch)
        
        images = torch.stack(images, 0)
        labels = torch.stack(labels, 0)

        encoded_text = tokenizer.batch_encode_plus(
            list(captions),
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        return [images, encoded_text, labels]

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=True, collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=True, collate_fn=custom_collate_fn
    )

    return train_loader, val_loader, test_loader, n_classes

# Example Usage
if __name__ == '__main__':
    # Note: For this to run, you need 'openpyxl' and 'transformers': 
    # pip install openpyxl transformers
    
    TRAIN_FILE = '../../data/MUTE/train_hate.xlsx'
    VAL_FILE = '../../data/MUTE/valid_hate.xlsx'
    TEST_FILE = '../../data/MUTE/test_hate.xlsx'
    ROOT_DIR = '../../data/MUTE/Memes/'
    BERT_PATH = "../../bert-base-uncased" 
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    try:
        train_loader, val_loader, test_loader, n_classes = get_loader(
            root_dir=ROOT_DIR,
            train_file=TRAIN_FILE,
            val_file=VAL_FILE,
            test_file=TEST_FILE,
            bert_model_path=BERT_PATH,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS
        )

        print(f"✅ Successfully created data loaders and 'samples.txt'.")
        print(f"Number of classes: {n_classes}")
        print("-" * 30)

        print("Testing the train_loader... 🧪")
        batch_list = next(iter(train_loader))
        images, texts, labels = batch_list
        
        print(f"Image batch shape: {images.shape}")
        print(f"Labels batch shape: {labels.shape}")
        print(f"Text batch is a dictionary with keys: {texts.keys()}")
        print(f"Text input_ids shape: {texts['input_ids'].shape}")
        print(f"Example label: {labels[0].item()}")
        print("-" * 30)
        
        print(f"Train loader batches: {len(train_loader)}")
        print(f"Validation loader batches: {len(val_loader)}")
        print(f"Test loader batches: {len(test_loader)}")

    except FileNotFoundError as e:
        print(f"❌ Error: One of the dataset paths is incorrect: {e}")
    except ImportError as e:
        print(f"❌ Error: A required package is missing. {e}")