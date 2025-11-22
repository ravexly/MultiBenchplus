

import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

class Twitter1517Dataset(Dataset):
    """
    Custom Dataset for a combined Twitter dataset (like Twitter1517).
    It handles data loading for image-text pairs.
    """
    def __init__(self, texts, image_ids, labels, images_dir, transform=None):
        """
        Args:
            texts (array): An array of text content.
            image_ids (array): An array of image file IDs.
            labels (array): An array of labels.
            images_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image sample.
        """
        self.texts = texts
        self.image_ids = image_ids
        self.labels = labels
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        text = self.texts[idx]
        if not isinstance(text, str):
            text = "" # Ensure text is always a string

        image_id = self.image_ids[idx]
        img_path = os.path.join(self.images_dir, image_id)
        
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except (FileNotFoundError, IOError):
            # Return None for missing images, to be filtered by collate_fn
            return None

        label = self.labels[idx]
        
        return image, text, torch.tensor(label, dtype=torch.long)

def get_loader(
    base_dir='../../data/Twitter1517', 
    bert_model_path="../../bert-base-uncased",
    batch_size=8,
    num_workers=4
):
    """
    Loads data, splits it, creates DataLoaders, and writes sample counts.
    This version tokenizes text in batches and returns a list.
    """
    # 1. Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)

    images_dir = os.path.join(base_dir, 'image')
    csv_file = os.path.join(base_dir, 'Twitter1517_texts_and_labels.csv')

    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found at {csv_file}")
        return None, None, None, 0

    data = pd.read_csv(csv_file)
    data.columns = data.columns.str.strip()
    
    texts = data['txt'].values
    image_ids = data['new_image_id'].values
    labels = data['cor_label'].values
    n_classes = len(pd.unique(labels))

    train_texts, temp_texts, train_image_ids, temp_image_ids, train_labels, temp_labels = train_test_split(
        texts, image_ids, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_image_ids, test_image_ids, val_labels, test_labels = train_test_split(
        temp_texts, temp_image_ids, temp_labels, test_size=(2/3.), random_state=42, stratify=temp_labels
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = Twitter1517Dataset(train_texts, train_image_ids, train_labels, images_dir, transform)
    val_dataset = Twitter1517Dataset(val_texts, val_image_ids, val_labels, images_dir, transform)
    test_dataset = Twitter1517Dataset(test_texts, test_image_ids, test_labels, images_dir, transform)

    with open("samples.txt", "w") as f:
        f.write(f"Dataset: Twitter1517\n")
        f.write("="*20 + "\n")
        f.write(f"Train samples: {len(train_dataset)}\n")
        f.write(f"Val samples: {len(val_dataset)}\n")
        f.write(f"Test samples: {len(test_dataset)}\n")
        f.write(f"Total samples: {len(data)} (from original file)\n")

    # 2. Define the custom collate function for batch processing
    def custom_collate_fn(batch):
        """
        Filters out None items and processes the rest.
        Returns a list: [images, texts, labels].
        """
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            return None

        images, texts, labels = zip(*batch)
        
        images = torch.stack(images, 0)
        labels = torch.stack(labels, 0)

        encoded_text = tokenizer.batch_encode_plus(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
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
    BASE_DIR = '../../data/Twitter1517'
    BERT_PATH = "../../bert-base-uncased"
    
    if not os.path.isdir(BASE_DIR):
        print(f"Error: Base directory not found at '{BASE_DIR}'")
    else:
        train_loader, val_loader, test_loader, n_classes = get_loader(
            base_dir=BASE_DIR,
            bert_model_path=BERT_PATH,
            batch_size=4
        )
        
        if train_loader:
            print(f'Training samples: {len(train_loader.dataset)}')
            print(f'Validation samples: {len(val_loader.dataset)}')
            print(f'Test samples: {len(test_loader.dataset)}')
            print(f'Number of classes: {n_classes}')
            print("-" * 30)

            print("Verifying one batch from train_loader...")
            for batch_list in train_loader:
                if batch_list is not None:
                    images, texts, labels = batch_list
                    print(f'Images batch shape: {images.shape}')
                    print(f"Text batch is a dictionary with keys: {texts.keys()}")
                    print(f'Labels batch shape: {labels.shape}')
                    break
            else:
                print("Could not retrieve a valid batch from train_loader.")