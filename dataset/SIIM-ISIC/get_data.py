import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder


class SIIMISICDataset(Dataset):
    """
    Custom PyTorch Dataset for the SIIM-ISIC Melanoma Classification challenge.
    It loads an image and its corresponding tabular metadata.
    """
    def __init__(self, csv_path, img_dir, transform=None):
        """
        Args:
            csv_path (str): Path to the training CSV file.
            img_dir (str): Directory containing the images.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # --- Preprocessing ---
        # Fill missing values for metadata features
        self.data["sex"].fillna("unknown", inplace=True)
        self.data["age_approx"].fillna(self.data["age_approx"].mean(), inplace=True)
        self.data["anatom_site_general_challenge"].fillna("unknown", inplace=True)

        # Fit label encoders for categorical features
        self.sex_encoder = LabelEncoder().fit(self.data["sex"])
        self.site_encoder = LabelEncoder().fit(self.data["anatom_site_general_challenge"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # --- Image Loading ---
        image_name = row["image_name"]
        img_path = os.path.join(self.img_dir, image_name + ".jpg")
        
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except (FileNotFoundError, IOError) as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder if the image fails to load
            if self.transform:
                # Assuming transforms resize to 224x224
                image = torch.zeros((3, 224, 224))
            else:
                raise e

        # --- Metadata Processing ---
        sex_encoded = self.sex_encoder.transform([row["sex"]])[0]
        site_encoded = self.site_encoder.transform([row["anatom_site_general_challenge"]])[0]
        age = row["age_approx"]
        
        # Combine metadata into a single tensor
        meta = torch.tensor([sex_encoded, age, site_encoded], dtype=torch.float)

        # --- Label ---
        label = int(row["target"])

        return image, meta, label


def get_loader(
    root_path="../../data/SIIM-ISIC",
    batch_size=8,
    num_workers=32,
    train_shuffle=True,
    val_split=0.1,
    test_split=0.1,
    seed=42
):
    """
    Creates and returns the data loaders for the SIIM-ISIC dataset. 
    Also logs sample counts to 'samples.txt'.

    Args:
        root_path (str): The root directory of the dataset.
                         e.g., /public/home/bupt/datas/siim-isic
        batch_size (int): The size of each batch.
        num_workers (int): Number of worker threads for loading data.
        train_shuffle (bool): Whether to shuffle the training data.
        val_split (float): The proportion of the dataset to use for validation.
        test_split (float): The proportion of the dataset to use for testing.
        seed (int): Random seed for reproducible splits.

    Returns:
        tuple: A tuple containing (train_loader, val_loader, test_loader, n_classes).
               Each loader yields (image_tensor, metadata_tensor, label).
    """
    # Define paths
    csv_path = os.path.join(root_path, "train.csv")
    img_dir = os.path.join(root_path, "jpeg/train")

    # Define standard image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Using standard ImageNet normalization values
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create the full dataset
    dataset = SIIMISICDataset(csv_path, img_dir, transform)
    
    # Calculate the number of classes from the dataset's 'target' column
    n_classes = dataset.data['target'].nunique()

    # --- Create Data Splits ---
    total_len = len(dataset)
    val_len = int(val_split * total_len)
    test_len = int(test_split * total_len)
    train_len = total_len - val_len - test_len

    train_set, val_set, test_set = random_split(
        dataset, 
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # --- Log Sample Counts (As Requested) ---
    with open("samples.txt", "w") as f:
        f.write(f"Dataset: SIIM-ISIC\n")
        f.write("="*20 + "\n")
        f.write(f"Train samples: {len(train_set)}\n")
        f.write(f"Val samples: {len(val_set)}\n")
        f.write(f"Test samples: {len(test_set)}\n")
        f.write(f"Total samples: {len(train_set) + len(val_set) + len(test_set)}\n")


    # --- Create DataLoaders ---
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, n_classes

if __name__ == '__main__':
    # Example usage: Replace with the actual path to your SIIM-ISIC data
    DATASET_ROOT = "../../data/SIIM-ISIC" 

    if not os.path.exists(DATASET_ROOT) or "path/to" in DATASET_ROOT:
        print(f"Error: The directory '{DATASET_ROOT}' seems invalid or does not exist.")
        print("Please update the DATASET_ROOT variable in the script.")
    else:
        train_loader, val_loader, test_loader, n_classes = get_loader(DATASET_ROOT)
        
        print(f"Successfully created dataloaders for the SIIM-ISIC dataset.")
        print(f"Number of classes: {n_classes}")
        print("Sample counts have been saved to 'samples.txt'.")
        print("-" * 25)
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
        print(f"Number of test batches: {len(test_loader)}")

        # Inspect a single batch from the training loader
        images, metas, labels = next(iter(train_loader))
        
        print("\n--- Sample Batch ---")
        print(f"Images batch shape: {images.shape}")
        print(f"Metadata batch shape: {metas.shape}")
        print(f"Labels batch shape: {labels.shape}")
        print(f"Sample metadata tensor: {metas[0]}")
        print(f"Sample label: {labels[0]}")