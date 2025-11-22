import os
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from PIL import ImageFile
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from transformers import BertTokenizer

# Configuration to handle potential image file issues
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

class MemotionDataset(Dataset):
    """
    Custom PyTorch Dataset for the preprocessed unified Memotion dataset.
    This version returns raw text strings for batch processing in collate_fn.
    """
    def __init__(self, tsv_path, image_dir, transform=None):
        """
        Args:
            tsv_path (Path): Path to the unified 'all_data.tsv' file.
            image_dir (Path): Path to the unified 'all_images' directory.
            transform (callable, optional): Transform to be applied to the image.
        """
        self.data_frame = pd.read_csv(tsv_path, sep='\t')
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_frame.iloc[idx]
        image_name = row['image_name']
        text = str(row['text'])
        # Convert label to a tensor for easy stacking in the collate function
        label = torch.tensor(int(row['label']), dtype=torch.long)

        image_path = self.image_dir / image_name
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except (FileNotFoundError, IOError) as e:
            print(f"Warning: Error loading image {image_path}: {e}. Using a black tensor as a placeholder.")
            # Ensure the placeholder has the correct dimensions after transform
            image = torch.zeros((3, 224, 224)) 

        # Return the raw text string; tokenization will happen in the collate_fn
        return image, text, label

def get_loader(
    root_dir= "../../data",
    bert_model_path="../../bert-base-uncased",
    batch_size=8,
    num_workers=4,
    train_shuffle=True,
    val_split=0.1,
    test_split=0.1,
    seed=42
):
    """
    Creates and returns data loaders for the Memotion dataset.
    This version tokenizes text in batches and returns a tuple, where the
    text modality is a dictionary.
    """
    base_path = Path(root_dir) / 'memotion'
    tsv_path = base_path / 'all_data.tsv'
    image_dir = base_path / 'all_images'
    
    if not (tsv_path.exists() and image_dir.exists()):
        raise FileNotFoundError(
            f"Preprocessed data not found. Please run the preprocessing script first. "
            f"Expected to find '{tsv_path}' and '{image_dir}'."
        )

    tokenizer = BertTokenizer.from_pretrained(bert_model_path)

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def custom_collate_fn(batch):
        """
        Processes a batch of data.
        Returns a tuple: (image_tensor, text_dictionary, label_tensor)
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
 
        return [images, encoded_text, labels]

    # --- Create a single, full dataset ---
    full_dataset = MemotionDataset(tsv_path, image_dir, transform=image_transform)
    n_classes = full_dataset.data_frame['label'].nunique()

    # --- Perform a 3-way split (Train, Validation, Test) ---
    total_len = len(full_dataset)
    val_len = int(val_split * total_len)
    test_len = int(test_split * total_len)
    train_len = total_len - val_len - test_len

    if train_len <= 0 or val_len <= 0 or test_len <= 0:
        raise ValueError("Split sizes are invalid. Check val_split and test_split values.")

    train_set, val_set, test_set = random_split(
        full_dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(seed)
    )

    # --- Log Sample Counts ---
    # (This part remains unchanged)
    with open("samples.txt", "w") as f:
        f.write(f"Dataset: Memotion (from single source)\n")
        f.write("="*20 + "\n")
        f.write(f"Train samples: {len(train_set)}\n")
        f.write(f"Val samples: {len(val_set)}\n")
        f.write(f"Test samples: {len(test_set)}\n")
        f.write(f"Total samples: {total_len}\n")

    # --- Create DataLoaders using the custom collate_fn ---
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=train_shuffle,
        num_workers=num_workers, collate_fn=custom_collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=custom_collate_fn, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=custom_collate_fn, drop_last=True
    )

    return train_loader, val_loader, test_loader, n_classes

if __name__ == '__main__':
    PROCESSED_DATA_ROOT = "../../data"
    
    try:
        train_loader, val_loader, test_loader, n_classes = get_loader(
            PROCESSED_DATA_ROOT, val_split=0.1, test_split=0.1
        )
        
        print("Successfully created dataloaders for the Memotion dataset.")
        print(f"Number of classes: {n_classes}")
        print("Sample counts saved to 'samples.txt'.")
        print("-" * 25)
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
        print(f"Number of test batches: {len(test_loader)}")

        # CORRECTED: Unpack the tuple directly
        images, texts, labels = next(iter(train_loader))
        print("\n--- Sample Batch ---")
        print(f"Images batch shape: {images.shape}")
        # 'texts' is now the dictionary from the tokenizer
        print(f"Text batch is a dictionary with keys: {texts.keys()}")
        print(f"Text input_ids shape: {texts['input_ids'].shape}")
        print(f"Labels batch shape: {labels.shape}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ImportError:
        print("Error: 'transformers' library not found. Please install it using: pip install transformers")