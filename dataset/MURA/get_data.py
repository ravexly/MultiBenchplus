import os
import sys
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

class MURADataset(Dataset):
    def __init__(self, data, transform, root_dir):
        self.data = data  # [(study_path, label)]
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        study_path, label = self.data[idx]
        study_dir = os.path.join(self.root_dir, study_path)

        try:
            image_files = sorted([
                f for f in os.listdir(study_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
        except Exception as e:
            print(f"[ERROR] Cannot access directory: {study_dir} | Error: {e}")
            raise

        if len(image_files) == 0:
            print(f"[WARNING] Empty study: {study_dir}")

        images = []
        for img_name in image_files:
            img_path = os.path.join(study_dir, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.transform(img)
                images.append(img_tensor)
            except Exception as e:
                print(f"[ERROR] Failed to load {img_path} | Error: {e}")
                raise

        return images, label

def mura_collate_fn(batch):
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels

def get_loader(batch_size=8):
    root_dir = "../../data/MURA/"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_lbl = pd.read_csv(os.path.join(root_dir, "MURA-v1.1/train_labeled_studies.csv"), header=None)
    valid_lbl = pd.read_csv(os.path.join(root_dir, "MURA-v1.1/valid_labeled_studies.csv"), header=None)
    train_lbl.columns = ['study_path', 'label']
    valid_lbl.columns = ['study_path', 'label']

    all_lbl = pd.concat([train_lbl, valid_lbl], ignore_index=True)
    all_lbl['study_path'] = all_lbl['study_path'].apply(lambda x: x if x.endswith('/') else x + '/')

    studies = all_lbl.drop_duplicates(subset='study_path').reset_index(drop=True).iloc[:100]
    shuffled = studies.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(shuffled)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_data = list(shuffled.iloc[:n_train].itertuples(index=False, name=None))
    val_data = list(shuffled.iloc[n_train:n_train + n_val].itertuples(index=False, name=None))
    test_data = list(shuffled.iloc[n_train + n_val:].itertuples(index=False, name=None))

    train_set = MURADataset(train_data, transform, root_dir)
    val_set = MURADataset(val_data, transform, root_dir)
    test_set = MURADataset(test_data, transform, root_dir)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=mura_collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=mura_collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=mura_collate_fn)

    total = len(train_set) + len(val_set) + len(test_set)
    with open("samples.txt", "w") as f:
        f.write(f"Total samples: {total}\n")
        f.write(f"Train samples: {len(train_set)}\n")
        f.write(f"Val samples: {len(val_set)}\n")
        f.write(f"Test samples: {len(test_set)}\n")

    n_classes = len(all_lbl['label'].unique())
    for ep in train_loader:
        print(ep[0].shape)
        break
    return train_loader, val_loader, test_loader, n_classes
get_loader()
