

import os
import json
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer

class FoodDataset(Dataset):
    def __init__(self, jsonl_file, image_root, labels_list, transform=None):
        """
        Args:
            jsonl_file (str): Path to the .jsonl file (e.g., train.jsonl).
            image_root (str): Root directory for the images.
            labels_list (list): A list of unique labels for mapping.
            transform (callable, optional): Image preprocessing transform.
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
        img_path = os.path.join(self.image_root, sample["img"])

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except (FileNotFoundError, IOError):
            print(f"Warning: Image file not found {img_path}. Using a placeholder.")
            # Create a placeholder tensor for robustness
            image = torch.zeros((3, 224, 224))
        
        if not isinstance(text, str):
            text = ""

        label_index = self.labels_list.index(label)
        return image, text, torch.tensor(label_index, dtype=torch.long)


def get_loader(
    base_dir="/data/xueleyan/MultiBench/data/food101", 
    bert_model_path="../../bert-base-uncased",
    batch_size=8, 
    num_workers=8, 
    shuffle=True
):
    """
    Creates and returns data loaders for the Food101 dataset.
    This version tokenizes text in batches and returns a list.
    """
    # 1. Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    
    image_root = base_dir

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Make label file path relative to base_dir for better portability
    label_file = Path(base_dir) / "label.txt"
    with open(label_file, 'r') as f:
        labels_list = [s.strip() for s in f.read().strip()[1:-1].replace("'", "").split(", ")]

    train_set = FoodDataset(os.path.join(base_dir, "train.jsonl"), image_root, labels_list, transform=transform)
    val_set   = FoodDataset(os.path.join(base_dir, "dev.jsonl"),   image_root, labels_list, transform=transform)
    test_set  = FoodDataset(os.path.join(base_dir, "test.jsonl"),  image_root, labels_list, transform=transform)

    num_classes = len(labels_list)

    info = (
        f"Dataset: Food101\n"
        f"====================\n"
        f"Train samples: {len(train_set)}\n"
        f"Val samples:   {len(val_set)}\n"
        f"Test samples:  {len(test_set)}\n"
        f"Total samples: {len(train_set) + len(val_set) + len(test_set)}\n"
        f"Num_classes: {num_classes}\n"
    )
    print(info)
    with open("samples.txt", "w") as f:
        f.write(info)

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
        train_set, batch_size=batch_size, shuffle=shuffle, 
        num_workers=num_workers, collate_fn=custom_collate_fn
    )
    val_loader   = DataLoader(
        val_set,   batch_size=batch_size, shuffle=False,   
        num_workers=num_workers, collate_fn=custom_collate_fn
    )
    test_loader  = DataLoader(
        test_set,  batch_size=batch_size, shuffle=False,   
        num_workers=num_workers, collate_fn=custom_collate_fn
    )

    return train_loader, val_loader, test_loader, num_classes

if __name__ == '__main__':
    # Example Usage
    DATA_DIR = "/data/xueleyan/MultiBench/data/food101"
    BERT_PATH = "../../bert-base-uncased"
    
    try:
        train_loader, val_loader, test_loader, n_classes = get_loader(
            base_dir=DATA_DIR,
            bert_model_path=BERT_PATH,
            batch_size=32,
            num_workers=4
        )
        
        print(f"\n✅ Successfully created data loaders.")
        print("-" * 30)

        # Test the train_loader to see the new format
        print("Testing the train_loader... 🧪")
        batch_list = next(iter(train_loader))
        images, texts, labels = batch_list # Unpack the list
        
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
        print(f"❌ Error: A required file or directory was not found: {e}")
    except ImportError:
        print("❌ Error: 'transformers' library not found. Please install it using: pip install transformers")