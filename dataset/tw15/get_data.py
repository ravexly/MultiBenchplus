
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
from transformers import BertTokenizer

class Twitter2015Dataset(Dataset):
    """
    Custom Dataset for the Twitter2015 multimodal dataset.
    """
    def __init__(self, tsv_file, images_dir, transform=None):
        """
        Args:
            tsv_file (str): Path to the tsv file (train.tsv, dev.tsv, test.tsv).
            images_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(tsv_file, sep='\t', header=0)
        self.data.columns = self.data.columns.str.strip()
        self.images_dir = images_dir
        self.transform = transform
        
        # Get necessary columns using their names for clarity
        self.image_ids = self.data['#2 ImageID'].values
        # Concatenate two text columns, handling potential non-string data
        self.texts = self.data['#3 String'].astype(str) + ' ' + self.data['#3 String.1'].astype(str)
        self.labels = self.data['#1 Label'].values

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.images_dir, image_id)
        
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except (FileNotFoundError, IOError):
            # Return None for missing images, to be filtered by collate_fn
            return None

        text = self.texts[idx]
        label = self.labels[idx]
        
        return image, text, torch.tensor(label, dtype=torch.long)

def get_loader(
    base_dir='../../data/Twitter2015/twitter2015', 
    bert_model_path="../../bert-base-uncased",
    batch_size=8,
    num_workers=4
):
    """
    Creates data loaders for the Twitter2015 dataset.
    This version tokenizes text in batches and returns a list.
    """
    # 1. Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)

    images_dir = os.path.join(base_dir, 'twitter2015_images')
    train_tsv = os.path.join(base_dir, 'twitter2015/train.tsv')
    val_tsv = os.path.join(base_dir, 'twitter2015/dev.tsv')
    test_tsv = os.path.join(base_dir, 'twitter2015/test.tsv')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = Twitter2015Dataset(tsv_file=train_tsv, images_dir=images_dir, transform=transform)
    val_dataset = Twitter2015Dataset(tsv_file=val_tsv, images_dir=images_dir, transform=transform)
    test_dataset = Twitter2015Dataset(tsv_file=test_tsv, images_dir=images_dir, transform=transform)

    if len(train_dataset.labels) > 0:
        n_classes = len(pd.unique(train_dataset.labels))
    else:
        n_classes = 0

    with open("samples.txt", "w") as f:
        f.write(f"Dataset: Twitter2015\n")
        f.write("="*20 + "\n")
        f.write(f"Train samples: {len(train_dataset)}\n")
        f.write(f"Val samples: {len(val_dataset)}\n")
        f.write(f"Test samples: {len(test_dataset)}\n")
        f.write(f"Total samples: {len(train_dataset) + len(val_dataset) + len(test_dataset)}\n")

    # 2. Define the custom collate function for batch processing
    def custom_collate_fn(batch):
        """
        Filters out None items (from missing images) and processes the rest.
        Returns a list: [images, texts, labels].
        """
        # Filter out samples that failed to load (returned as None)
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            return None # Return None if the whole batch was invalid

        images, texts, labels = zip(*batch)
        
        images = torch.stack(images, 0)
        labels = torch.stack(labels, 0)

        encoded_text = tokenizer.batch_encode_plus(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # Return a list as requested
        return [images, encoded_text, labels]

    # 3. Create DataLoaders using the custom collate function
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader, test_loader, n_classes

if __name__ == "__main__":
    BASE_DIR = '../../data/Twitter2015'
    BERT_PATH = "../../bert-base-uncased"
    
    if not os.path.isdir(BASE_DIR):
        print(f"Error: Base directory not found at '{BASE_DIR}'")
    else:
        train_loader, val_loader, test_loader, n_classes = get_loader(
            base_dir=BASE_DIR,
            bert_model_path=BERT_PATH,
            batch_size=4
        )
        
        print(f'Training samples: {len(train_loader.dataset)}')
        print(f'Validation samples: {len(val_loader.dataset)}')
        print(f'Test samples: {len(test_loader.dataset)}')
        print(f'Number of classes: {n_classes}')
        print("-" * 30)

        print("Verifying one batch from train_loader...")
        # Loop to find a valid batch, in case the first one was filtered out
        for batch_list in train_loader:
            if batch_list is not None:
                images, texts, labels = batch_list
                print(f'Images batch shape: {images.shape}')
                print(f"Text batch is a dictionary with keys: {texts.keys()}")
                print(f'Labels batch shape: {labels.shape}')
                break
        else:
             print("Could not retrieve a valid batch from train_loader.")