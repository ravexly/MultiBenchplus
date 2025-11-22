import os
import json
import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from transformers import BertTokenizer

def get_labels_and_frequencies(path):
    """Reads a jsonl file to count label frequencies."""
    data_labels = [json.loads(line)["label"] for line in open(path)]
    label_freqs = Counter(data_labels)
    
    return list(label_freqs.keys()), label_freqs

class MVSADataset(Dataset):
    """
    Custom PyTorch Dataset for the MVSA (Multi-View Sentiment Analysis) dataset.
    Loads image-text pairs.
    """
    def __init__(self, jsonl_file, image_root, labels_list, transform=None):
        """
        Args:
            jsonl_file (Path): Path to the .jsonl file (e.g., train.jsonl).
            image_root (Path): Root directory for the images.
            labels_list (list): A list of unique labels to map labels to indices.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.samples = []
        self.image_root = image_root
        self.transform = transform
        self.labels_list = labels_list

        with open(jsonl_file, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        label = sample["label"]
        img_path = self.image_root / sample["img"]

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except (FileNotFoundError, IOError) as e:
            print(f"Warning: Error loading image {img_path}: {e}. Using a placeholder.")
            image = torch.zeros((3, 224, 224))

        label_index = self.labels_list.index(label)
        
        # The label is already being correctly converted to a tensor.
        return image, text, torch.tensor(label_index, dtype=torch.long)

def get_loader(
    base_dir="/data/xueleyan/MultiBench/data/MVSA_Single", 
    bert_model_path="../../bert-base-uncased",
    batch_size=8, 
    num_workers=4, 
    shuffle_train=True
):
    """
    Creates and returns the data loaders for the MVSA dataset.
    This version tokenizes text in batches and returns a list.
    """
    # 1. Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    
    base_path = Path(base_dir)
    train_path = base_path / "train.jsonl"
    val_path = base_path / "val.jsonl"
    test_path = base_path / "test.jsonl"
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    labels_list, _ = get_labels_and_frequencies(train_path)
    n_classes = len(labels_list)

    train_set = MVSADataset(train_path, base_path, labels_list, transform=transform)
    val_set = MVSADataset(val_path, base_path, labels_list, transform=transform)
    test_set = MVSADataset(test_path, base_path, labels_list, transform=transform)

    with open("samples.txt", "w") as f:
        f.write(f"Dataset: MVSA\n")
        f.write("="*20 + "\n")
        f.write(f"Train samples: {len(train_set)}\n")
        f.write(f"Val samples: {len(val_set)}\n")
        f.write(f"Test samples: {len(test_set)}\n")
        f.write(f"Total samples: {len(train_set) + len(val_set) + len(test_set)}\n")

    # 2. Define the custom collate function for batch processing
    def custom_collate_fn(batch):
        """
        Processes a batch and returns a list: [images, texts, labels].
        The text modality is a dictionary from the tokenizer.
        """
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
        train_set, batch_size=batch_size, shuffle=shuffle_train, 
        num_workers=num_workers, collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, collate_fn=custom_collate_fn
    )

    return train_loader, val_loader, test_loader, n_classes

if __name__ == '__main__':
    # Example usage: Replace with the actual path to your MVSA_Single data
    DATASET_BASE_DIR = "/data/xueleyan/MultiBench/data/MVSA_Single"
    BERT_PATH = "../../bert-base-uncased"
    
    if not Path(DATASET_BASE_DIR).exists():
        print(f"Error: The directory '{DATASET_BASE_DIR}' does not exist.")
        print("Please update the DATASET_BASE_DIR variable in the script.")
    else:
        # Fixed the function name call and added bert_model_path
        train_loader, val_loader, test_loader, n_classes = get_loader(
            base_dir=DATASET_BASE_DIR,
            bert_model_path=BERT_PATH
        )
        
        print(f"Successfully created dataloaders for the MVSA dataset.")
        print(f"Number of classes: {n_classes}")
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
        print(f"Number of test batches: {len(test_loader)}")
        print("Sample counts have been saved to 'samples.txt'.")

        # Inspect a single batch to see the new format
        batch_list = next(iter(train_loader))
        images, texts, labels = batch_list # Unpack the list

        print("\n--- Sample Batch ---")
        print(f"Images batch shape: {images.shape}")
        print(f"Labels batch shape: {labels.shape}")
        print(f"Text batch is a dictionary with keys: {texts.keys()}")
        print(f"Text input_ids shape: {texts['input_ids'].shape}")
        print(f"Example labels in batch: {labels.tolist()}")