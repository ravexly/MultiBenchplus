import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

class MyDataset(Dataset):
    def __init__(self, sample_ids, labels_df, data_dir, transform_img, transform_oct):
        self.sample_ids = sample_ids
        self.labels_df = labels_df
        self.data_dir = data_dir
        self.transform_img = transform_img
        self.transform_oct = transform_oct

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        sample_path = os.path.join(self.data_dir, "multi-modality_images", sample_id)

        # Fundus image
        fundus_img_path = os.path.join(sample_path, f"{sample_id}.jpg")
        fundus_img = cv2.imread(fundus_img_path)[:, :, ::-1]
        fundus_img = Image.fromarray(fundus_img)
        fundus_img = self.transform_img(fundus_img)

        # OCT image
        oct_path = os.path.join(sample_path, sample_id)
        oct_slices = sorted(os.listdir(oct_path), key=lambda s: int(s.split("_")[0]))
        oct_img = np.stack([cv2.imread(os.path.join(oct_path, p), cv2.IMREAD_GRAYSCALE) for p in oct_slices], axis=0)
        oct_img = torch.tensor(oct_img.copy(), dtype=torch.float32)
        oct_img = torch.stack([self.transform_oct(Image.fromarray(oct_img[i].numpy()))[0] for i in range(oct_img.shape[0])])

        # Label
        label_row = self.labels_df[self.labels_df["data"] == sample_id]
        label = label_row.iloc[:, 1:].values.astype(np.float32)
        label = torch.tensor(label).argmax()

        return fundus_img, oct_img, label

def get_loader(batch_size=8, num_workers=4):
    data_dir = "../../data/GAMMA/training/"
    labels_file = os.path.join(data_dir, "glaucoma_grading_training_GT.xlsx")
    labels_df = pd.read_excel(labels_file, dtype=str)

    all_samples = labels_df["data"].tolist()


    test_samples = all_samples[0:70]
    val_samples = all_samples[70:80]
    train_samples = all_samples[80:]

    transform_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    transform_oct = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    train_set = MyDataset(train_samples, labels_df, data_dir, transform_img, transform_oct)
    val_set = MyDataset(val_samples, labels_df, data_dir, transform_img, transform_oct)
    test_set = MyDataset(test_samples, labels_df, data_dir, transform_img, transform_oct)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    with open("samples.txt", "w") as f:
        f.write(f"Total: {len(all_samples)}\n")
        f.write(f"Train: {len(train_set)}\n")
        f.write(f"Val: {len(val_set)}\n")
        f.write(f"Test: {len(test_set)}\n")

    return train_loader,test_loader, test_loader, 3  


if __name__ == '__main__':
    train_loader, test_loader, n_classes = get_loader(batch_size=4)
    print("训练集 DataLoader:")
    for batch in train_loader:
        print(batch[0].shape, batch[1].shape, batch[2].shape)
        break
