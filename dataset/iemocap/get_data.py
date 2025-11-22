import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import os


class IEMOCAPDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.split = split
        self.data_path = data_path
        (self.videoIDs, self.videoSpeakers, self.videoLabels,
         self.videoText, self.videoAudio, self.videoVisual,
         self.videoSentence, self.trainVid, self.testVid) = self.load_data()

        if self.split == 'train':
            self.indices = [(vid, j)
                            for vid in self.videoIDs if vid in self.trainVid
                            for j in range(len(self.videoText[vid]))]
        elif self.split == 'test':
            self.indices = [(vid, j)
                            for vid in self.videoIDs if vid in self.testVid
                            for j in range(len(self.videoText[vid]))]
        elif self.split == 'valid':
            # 如果没有现成验证集，可以简单地把训练集再划分一部分出来
            full_train = [(vid, j)
                          for vid in self.videoIDs if vid in self.trainVid
                          for j in range(len(self.videoText[vid]))]
            train_len = int(0.9 * len(full_train))
            valid_len = len(full_train) - train_len
            if self.split == 'train':
                self.indices = full_train[:train_len]
            elif self.split == 'valid':
                self.indices = full_train[train_len:]
        else:
            raise ValueError("split must be one of ['train', 'valid', 'test']")

    def load_data(self):
        with open(self.data_path, "rb") as f:
            return pickle.load(f, encoding='latin1')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        vid, j = self.indices[idx]
        speaker = self.videoSpeakers[vid][j]
        label = self.videoLabels[vid][j]
        text = torch.tensor(self.videoText[vid][j], dtype=torch.float32)
        audio = torch.tensor(self.videoAudio[vid][j], dtype=torch.float32)
        visual = torch.tensor(self.videoVisual[vid][j], dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return text, audio, visual, label


def get_loader(data_path='../../data/IEMOCAP/IEMOCAP_features_raw.pkl',
               batch_size=8,
               shuffle=True,
               num_workers=4):
    """
    统一返回 train / valid / test DataLoader
    同时返回类别数 n_classes
    """
    # 1. 根据 split 构建三个 Dataset
    train_dataset = IEMOCAPDataset(data_path, split='train')
    test_dataset  = IEMOCAPDataset(data_path, split='test')

    # 如果希望从训练集再划分出验证集，可以用下面这段
    full_train_len = len(train_dataset)
 
    train_len = full_train_len


    # 2. 构建 DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )

    # 3. 计算类别数（假设 label 从 0 开始连续编号）
    all_labels = []
    for _, _, _, label in train_dataset:
        all_labels.append(label.item())
    n_classes = len(set(all_labels))

    # 4. 写入 samples.txt
    with open("samples.txt", "w") as f:
        f.write(f"Train: {len(train_dataset)}\n")
        f.write(f"Test: {len(test_dataset)}\n")
        f.write(f"Total: {len(train_dataset)+len(test_dataset)}\n")

    return train_loader,test_loader, test_loader, n_classes


# 使用方式
if __name__ == "__main__":
    train_loader, valid_loader, test_loader, n_classes = get_loader(batch_size=8)
    print("n_classes =", n_classes)