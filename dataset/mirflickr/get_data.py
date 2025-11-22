import os
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Subset
import scipy.io
import numpy as np
import h5py
from sklearn.model_selection import train_test_split

class MIRFlickrDataset(data.Dataset):
    """
    Custom Dataset for the MIRFlickr dataset.
    It expects pre-extracted features for text and images.
    """
    def __init__(self, text_data, image_data, label_data, transform=None):
        """
        Initializes the dataset.
        Args:
            text_data (np.array): Text feature data.
            image_data (np.array): Image feature data.
            label_data (np.array): One-hot encoded label data.
            transform (callable, optional): Optional transform to be applied on a sample.
                                            (Note: Not used for pre-extracted features but kept for consistency).
        """
        self.text_data = text_data
        self.image_data = image_data
        self.label_data = label_data
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.label_data)

    def __getitem__(self, index):
        """
        Generates one sample of data.
        Args:
            index (int): The index of the item.
        Returns:
            tuple: (image, text, label) where label is the class index.
        """
        text = self.text_data[index]
        image = self.image_data[index]
        label = self.label_data[index]


        text_tensor = torch.tensor(text, dtype=torch.float32)
        image_tensor = torch.tensor(image, dtype=torch.float32)
        

        # Convert one-hot encoded label to a class index
        label_index = torch.argmax(torch.tensor(label, dtype=torch.float32))
        label_tensor = torch.tensor(label_index, dtype=torch.long)

        return image_tensor, text_tensor, label_tensor

def get_loader(base_dir= '../../data/MIRFlickr', batch_size=8):
    """
    Loads MIRFlickr data, splits it into training, validation, and test sets (70:10:20),
    creates DataLoaders, and writes sample counts to a file.

    Args:
        base_dir (str): The root directory of the dataset containing the .mat files.
        batch_size (int): How many samples per batch to load.
    
    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training set.
            - val_loader (DataLoader): DataLoader for the validation set.
            - test_loader (DataLoader): DataLoader for the test set.
            - n_classes (int): The number of unique classes in the dataset.
    """

    yall_path = os.path.join(base_dir, 'Cleared-Set/YAll/mirflickr25k-yall.mat')
    lall_path = os.path.join(base_dir, 'Cleared-Set/LAll/mirflickr25k-lall.mat')
    iall_path = os.path.join(base_dir, 'Cleared-Set/IAll/mirflickr25k-iall.mat')

    try:

        text_data = scipy.io.loadmat(yall_path)['YAll']  
        label_data = scipy.io.loadmat(lall_path)['LAll']  
        with h5py.File(iall_path, 'r') as file:
            # h5py loads data as (N, C, H, W), but we need (N, D) for features
            image_data_raw = np.array(file['IAll'])
            # Assuming the features are flattened. If not, you might need to reshape.
            # Example: image_data = image_data_raw.reshape(image_data_raw.shape[0], -1)
            image_data = image_data_raw

    except FileNotFoundError as e:
        print(f"Error: Data file not found. {e}")
        print("Please ensure the MIRFlickr dataset is correctly placed in the base_dir.")
        return None, None, None, 0


    n_classes = label_data.shape[1]


    dataset = MIRFlickrDataset(text_data, image_data, label_data)
    

    np.random.seed(42)
    torch.manual_seed(42)


    dataset_size = len(dataset)
    indices = np.arange(dataset_size)


    # Split data: 70% train, 10% validation, 20% test
    # First split: separate 70% for training
    train_indices, temp_indices, _, temp_labels = train_test_split(
        indices, label_data, test_size=0.3, random_state=42, stratify=label_data.argmax(axis=1)
    )
    # Second split: separate the remaining 30% into 10% validation and 20% test
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=(2/3), random_state=42, stratify=temp_labels.argmax(axis=1)
    )


    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)


    train_len = len(train_dataset)
    val_len = len(val_dataset)
    test_len = len(test_dataset)
    total_len = train_len + val_len + test_len

    try:
        with open("samples.txt", "w") as f:
            f.write(f"Train: {train_len}\n")
            f.write(f"Val: {val_len}\n")
            f.write(f"Test: {test_len}\n")
            f.write(f"Total: {total_len} (from split)\n")
    except IOError as e:
        print(f"Error writing to samples.txt: {e}")


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers = 4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 4)
    
    return train_loader, val_loader, test_loader, n_classes

if __name__ == "__main__":

    base_dir = '../../data/MIRFlickr'
    

    if not os.path.isdir(base_dir):
        print(f"Error: Base directory not found at '{base_dir}'")
        print("Please update the 'base_dir' variable with the correct path to your dataset.")
    else:
  
        train_loader, val_loader, test_loader, n_classes = get_loader(base_dir, batch_size=8)
        
        if train_loader:
      
            print(f'Training samples: {len(train_loader.dataset)}')
            print(f'Validation samples: {len(val_loader.dataset)}')
            print(f'Test samples: {len(test_loader.dataset)}')
            print(f'Number of classes: {n_classes}')
            print("-" * 30)


            if os.path.exists("samples.txt"):
                print("'samples.txt' created successfully. Contents:")
                with open("samples.txt", "r") as f:
                    print(f.read())
            else:
                print("'samples.txt' was not created.")
            
            print("-" * 30)
   
            print("Verifying one batch from train_loader...")
            try:
                images, texts, labels = next(iter(train_loader))
                print(f'Images batch shape: {images.shape}')
                print(f'Texts batch shape: {texts.shape}')
                print(f'Labels batch: {labels}')
                print(f'Label of first sample: {labels[0]}')
            except StopIteration:
                print("Could not retrieve a batch from train_loader. Is the training set empty?")