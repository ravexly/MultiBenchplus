import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

class Derm7PtDataset(Dataset):
    """ PyTorch Dataset for Derm7Pt """

    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        clinic_img = Image.open(row["clinic_path"]).convert("RGB")
        derm_img = Image.open(row["derm_path"]).convert("RGB")

        if self.transform:
            clinic_img = self.transform(clinic_img)
            derm_img = self.transform(derm_img)

        metadata = torch.tensor(row["metadata"], dtype=torch.float32)
        label = torch.tensor(row["diagnosis_encoded"], dtype=torch.long)

        return clinic_img, derm_img, metadata, label

def get_loader(data_dir="../../data/Derm7pt/", batch_size=32):
    meta_path = os.path.join(data_dir, "meta/meta.csv")
    df = pd.read_csv(meta_path)

    df["clinic_path"] = df["clinic"].apply(lambda x: os.path.join(data_dir, "images", x))
    df["derm_path"] = df["derm"].apply(lambda x: os.path.join(data_dir, "images", x))

    label_encoder = LabelEncoder()
    df["diagnosis_encoded"] = label_encoder.fit_transform(df["diagnosis"])

    seven_point_cols = [
        "pigment_network", "streaks", "pigmentation", "regression_structures",
        "dots_and_globules", "blue_whitish_veil", "vascular_structures"
    ]
    for col in seven_point_cols:
        df[col] = df[col].map({"absent": 0, "present": 1}).fillna(0)

    categorical_cols = ["sex", "location", "elevation", "management"]
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_metadata = onehot_encoder.fit_transform(df[categorical_cols])

    df["metadata"] = list(encoded_metadata)

    train_idx = pd.read_csv(os.path.join(data_dir, "meta/train_indexes.csv"))["indexes"].tolist()
    valid_idx = pd.read_csv(os.path.join(data_dir, "meta/valid_indexes.csv"))["indexes"].tolist()
    test_idx = pd.read_csv(os.path.join(data_dir, "meta/test_indexes.csv"))["indexes"].tolist()

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = Derm7PtDataset(df_train, transform=image_transform)
    valid_dataset = Derm7PtDataset(df_valid, transform=image_transform)
    test_dataset = Derm7PtDataset(df_test, transform=image_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

    n_classes = len(label_encoder.classes_)

    print(f"[INFO] Sample counts:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Valid: {len(valid_dataset)}")
    print(f"  Test : {len(test_dataset)}")
    print(f"  Number of classes: {n_classes}")

    with open("samples.txt", "w") as f:
        f.write(f"Train: {len(train_dataset)}\n")
        f.write(f"Valid: {len(valid_dataset)}\n")
        f.write(f"Test: {len(test_dataset)}\n")
        f.write(f"Classes: {n_classes}\n")

    return train_loader, valid_loader, test_loader, n_classes

if __name__ == "__main__":
    train_loader, valid_loader, test_loader, n_classes = get_loader()
    print("Train batch example shapes:")
    for batch in train_loader:
        print(batch[0].shape, batch[1].shape, batch[2].shape,batch[3].shape)
        break
