import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

class MultiOFF(Dataset):
    """
    Custom Dataset for the MultiOFF dataset.
    Handles loading images, text, and labels from a CSV file.
    """
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Map string labels to numerical values.
        self.label_map = {'Non-offensiv': 0, 'offensive': 1}
        # Drop rows where label is not in the map
        self.annotations = self.annotations[self.annotations['label'].isin(self.label_map.keys())].copy()
        self.annotations['label'] = self.annotations['label'].map(self.label_map)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_name = row['image_name']
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except (FileNotFoundError, IOError):
            print(f"Warning: Image file not found {img_path}. Using a placeholder.")
            # Create a placeholder tensor for robustness
            image = torch.zeros((3, 224, 224))

        sentence = row['sentence']
        if not isinstance(sentence, str):
            sentence = ""
            
        label = row['label']
        # Convert label to a tensor for stacking
        label = torch.tensor(label, dtype=torch.long)

        return image, sentence, label

def get_loader(
    root_dir='../../data/MultiOFF/Labelled Images/', 
    train_csv='../../data/MultiOFF/Split Dataset/Training_meme_dataset.csv', 
    val_csv='../../data/MultiOFF/Split Dataset/Validation_meme_dataset.csv', 
    test_csv='../../data/MultiOFF/Split Dataset/Testing_meme_dataset.csv', 
    bert_model_path="../../bert-base-uncased",
    batch_size=4, 
    num_workers=4
):
    """
    Creates and returns the data loaders for the MultiOFF dataset.
    This version tokenizes text in batches and returns a list.
    """
    # 1. Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = MultiOFF(csv_file=train_csv, root_dir=root_dir, transform=transform)
    val_dataset = MultiOFF(csv_file=val_csv, root_dir=root_dir, transform=transform)
    test_dataset = MultiOFF(csv_file=test_csv, root_dir=root_dir, transform=transform)

    train_len, val_len, test_len = len(train_dataset), len(val_dataset), len(test_dataset)
    total_len = train_len + val_len + test_len
    
    with open("samples.txt", "w") as f:
        f.write(f"Dataset: MultiOFF\n")
        f.write("="*20 + "\n")
        f.write(f"Train samples: {train_len}\n")
        f.write(f"Validation samples: {val_len}\n")
        f.write(f"Test samples: {test_len}\n")
        f.write(f"Total samples: {total_len}\n")

    n_classes = len(train_dataset.label_map)

    # 2. Define the custom collate function for batch processing
    def custom_collate_fn(batch):
        """
        Processes a batch and returns a list: [images, texts, labels].
        The text modality is a dictionary from the tokenizer.
        """
        images, sentences, labels = zip(*batch)
        
        images = torch.stack(images, 0)
        labels = torch.stack(labels, 0)

        encoded_text = tokenizer.batch_encode_plus(
            list(sentences),
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # Return a list as requested
        return [images, encoded_text, labels]

    # 3. Create DataLoaders using the custom collate function
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
    # Define paths and hyperparameters
    TRAIN_CSV = '/data/xueleyan/MultiBench/data/MultiOFF/Split Dataset/Training_meme_dataset.csv'
    VAL_CSV = '/data/xueleyan/MultiBench/data/MultiOFF/Split Dataset/Validation_meme_dataset.csv'
    TEST_CSV = '/data/xueleyan/MultiBench/data/MultiOFF/Split Dataset/Testing_meme_dataset.csv'
    ROOT_DIR = '/data/xueleyan/MultiBench/data/MultiOFF/Labelled Images/'
    BATCH_SIZE = 32
    NUM_WORKERS = 4 
    BERT_PATH = "../../bert-base-uncased" # Define path to your BERT model

    try:
        train_loader, val_loader, test_loader, n_classes = get_loader(
            root_dir=ROOT_DIR,
            train_csv=TRAIN_CSV,
            val_csv=VAL_CSV,
            test_csv=TEST_CSV,
            bert_model_path=BERT_PATH, # Pass the bert path
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS
        )

        print(f"Successfully created data loaders and 'samples.txt'.")
        print(f"Number of classes: {n_classes}")
        print("-" * 30)

        # Test the train_loader to see the new format
        print("Testing the train_loader...")
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

    except FileNotFoundError:
        print("Error: One of the dataset paths is incorrect.")
    except ImportError:
        print("Error: 'transformers' library not found. Please install it using: pip install transformers")