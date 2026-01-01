import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

def get_loader(data_folder="../../data/BRCA/", view_list=[1,2,3], batch_size=128, num_workers=1, train_shuffle=True):
    num_view = len(view_list)
    
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    
    data_tr_list = []
    data_te_list = []
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, f"{i}_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, f"{i}_te.csv"), delimiter=','))
    
    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    
    labels = np.concatenate((labels_tr, labels_te))
    
    total_len = num_tr + num_te
    datasets = []
    for i in range(total_len):
        datasets.append([data_mat_list[0][i], data_mat_list[1][i], data_mat_list[2][i], labels[i]])
    
    random.seed(10)
    random.shuffle(datasets)
    
    test_split = total_len // 5      # 20%
    val_split = total_len // 4       # 25%
    
    test_data = datasets[:test_split]
    val_data = datasets[test_split:val_split]
    train_data = datasets[val_split:]
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers,drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers,drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers,drop_last=True)
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    print(f"[INFO] Samples count:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val  : {len(val_data)}")
    print(f"  Test : {len(test_data)}")
    print(f"  Total: {total_len}")
    print(f"  Number of classes: {n_classes}")
    
    with open("samples.txt", "w") as f:
        f.write(f"Train: {len(train_data)}\n")
        f.write(f"Val: {len(val_data)}\n")
        f.write(f"Test: {len(test_data)}\n")
        f.write(f"Total: {total_len}\n")
        f.write(f"Classes: {n_classes}\n")
    
    return train_loader, val_loader, test_loader, n_classes

if __name__ == "__main__":
    train_loader, val_loader, test_loader, n_classes = get_loader()
    print("Train batch example shapes:")
    for batch in train_loader:
        print(batch[0].shape, batch[1].shape, batch[2].shape)
        break
