import sys
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.folder import make_dataset

# Note: The 'find_classes' function is part of torchvision's internal API but included here for completeness.
def find_classes(directory):
    """
    Finds the class folders in a dataset.
    Args:
        directory (string): Root directory path.
    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir).
    """
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

class SUNRGBD:
    """
    SUN RGB-D custom dataset.
    
    Assumes that the input images are concatenations of RGB and Depth images side-by-side.
    It splits them, applies specified transforms, and returns them as a pair.

    NOTE: The provided transform pipelines are applied independently to the RGB and Depth
    images. For spatial transforms like RandomCrop and RandomHorizontalFlip, this can
    cause a mismatch. For a robust implementation, consider using a custom transform
    that applies the same random parameters to both images simultaneously.
    """
    def __init__(self, data_dir=None, rgb_transform=None, depth_transform=None, labeled=True):
        self.data_dir = data_dir
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.labeled = labeled

        self.classes, self.class_to_idx = find_classes(self.data_dir)
        self.int_to_class = {i: c for c, i in self.class_to_idx.items()}
        self.imgs = make_dataset(self.data_dir, self.class_to_idx, 'png')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if self.labeled:
            img_path, label = self.imgs[index]
        else:
            # This branch is not used in the current get_loader but kept for completeness
            img_path = self.imgs[index]
            label = -1 # Default label for unlabeled data

        img_name = os.path.basename(img_path)
        RGB_D = Image.open(img_path)

        # Split RGB and Depth images
        w, h = RGB_D.size
        w2 = int(w / 2)
        
        # Pre-resize images to 256x256 before applying transforms
        if w2 > 256:
            RGB = RGB_D.crop((0, 0, w2, h)).convert('RGB').resize((256, 256), Image.BICUBIC)
            Depth = RGB_D.crop((w2, 0, w, h)).convert('RGB').resize((256, 256), Image.BICUBIC)
        else:
            RGB = RGB_D.crop((0, 0, w2, h)).convert('RGB')
            Depth = RGB_D.crop((w2, 0, w, h)).convert('RGB')

        sample = {'RGB': RGB, 'Depth': Depth, 'label': label}

        if self.rgb_transform:
            sample['RGB'] = self.rgb_transform(sample['RGB'])
            
        if self.depth_transform:
            sample['Depth'] = self.depth_transform(sample['Depth'])

        return sample['RGB'], sample['Depth'], sample['label']

def get_loader(root_dir='../../data/sun_rgbd/', batch_size=8, num_workers=4, pin_memory=True):
    """
    Creates and returns the data loaders for the SUN RGB-D dataset.

    Args:
        root_dir (str): The root directory of the dataset, e.g., '.../sun_rgbd/'.
        batch_size (int): The batch size for the data loaders.
        num_workers (int): The number of subprocesses to use for data loading.
        pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory.

    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes)
    """
    train_dir = os.path.join(root_dir, 'train')
    # Using the 'test' folder for both validation and testing
    test_dir = os.path.join(root_dir, 'test') 

    # --- Define Transforms ---
    # Transforms for training (with data augmentation)
    train_rgb_transforms = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_depth_transforms = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    # Transforms for validation and testing (without data augmentation)
    eval_rgb_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    eval_depth_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    # --- Create Datasets ---
    train_dataset = SUNRGBD(
        data_dir=train_dir,
        rgb_transform=train_rgb_transforms,
        depth_transform=train_depth_transforms
    )
    val_dataset = SUNRGBD(
        data_dir=test_dir,
        rgb_transform=eval_rgb_transforms,
        depth_transform=eval_depth_transforms
    )
    # The test set is the same as the validation set
    test_dataset = val_dataset

    num_classes = len(train_dataset.classes)

    # --- Create DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle validation data
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle test data
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # --- Log Sample Counts ---
    print("Logging sample counts to samples.txt...")
    with open("samples.txt", "w") as f:
        f.write(f"Dataset: SUN RGB-D\n")
        f.write("="*20 + "\n")
        f.write(f"Train samples: {len(train_dataset)}\n")
        f.write(f"Val samples: {len(val_dataset)}\n")
        f.write(f"Test samples: {len(test_dataset)} (uses same data as validation set)\n")
        total_unique_len = len(train_dataset) + len(test_dataset)
        f.write(f"Total unique samples: {total_unique_len}\n")
    print("Done.")
    
    return train_loader, val_loader, test_loader, num_classes