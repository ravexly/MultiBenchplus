import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


class ROSMAPDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list  # list of [view1, view2, view3, label]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        view1 = torch.tensor(self.data[idx][0], dtype=torch.float32)
        view2 = torch.tensor(self.data[idx][1], dtype=torch.float32)
        view3 = torch.tensor(self.data[idx][2], dtype=torch.float32)
        label = torch.tensor(self.data[idx][3], dtype=torch.long)
        return view1, view2, view3, label


def get_loader(
    data_folder='../../data/ROSMAP/',
    view_list=[1, 2, 3],
    num_class=2,
    batch_size=32,
    num_workers=1,
    train_shuffle=True
):
    num_view = len(view_list)

    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',').astype(int)
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',').astype(int)

    data_tr_list = [
        np.loadtxt(os.path.join(data_folder, f"{i}_tr.csv"), delimiter=',') for i in view_list
    ]
    data_te_list = [
        np.loadtxt(os.path.join(data_folder, f"{i}_te.csv"), delimiter=',') for i in view_list
    ]

    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    le = num_tr + num_te

    data_all = []
    for i in range(num_view):
        data_all.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    labels = np.concatenate((labels_tr, labels_te))

    dataset = [
        [data_all[0][i], data_all[1][i], data_all[2][i], labels[i]]
        for i in range(le)
    ]
    random.seed(10)
    random.shuffle(dataset)

    test_len = le // 5
    val_len = le // 4
    train_len = le - test_len - val_len

    train_data = dataset[:train_len]
    val_data = dataset[train_len:train_len + val_len]
    test_data = dataset[train_len + val_len:]

    train_set = ROSMAPDataset(train_data)
    val_set = ROSMAPDataset(val_data)
    test_set = ROSMAPDataset(test_data)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    with open("samples.txt", "w") as f:
        f.write(f"Train samples: {len(train_set)}\n")
        f.write(f"Val samples: {len(val_set)}\n")
        f.write(f"Test samples: {len(test_set)}\n")
        f.write(f"Total samples: {len(dataset)}\n")

    return train_loader, val_loader, test_loader, num_class
