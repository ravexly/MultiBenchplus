import os
import sys
import numpy as np
import torch
import random
from torch.utils.data import DataLoader

# 添加路径设置
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

def get_loader(task=7, batch_size=40, num_workers=4, train_shuffle=True, imputed_path='../../data/MIMIC_III/im.pk', flatten_time_series=False):
   
    import pickle
    with open(imputed_path, 'rb') as f:
        datafile = pickle.load(f)

    X_t = datafile['ep_tdata']         # (num_samples, time_steps, features)
    X_s = datafile['adm_features_all'] # (num_samples, static_features)

    X_t[np.isinf(X_t)] = 0
    X_t[np.isnan(X_t)] = 0
    X_s[np.isinf(X_s)] = 0
    X_s[np.isnan(X_s)] = 0


    X_s_avg = np.average(X_s, axis=0)
    X_s_std = np.std(X_s, axis=0)
    X_t_avg = np.average(X_t, axis=(0, 1))
    X_t_std = np.std(X_t, axis=(0, 1))

    for i in range(len(X_s)):
        X_s[i] = (X_s[i] - X_s_avg) / X_s_std
        for j in range(len(X_t[0])):
            X_t[i][j] = (X_t[i][j] - X_t_avg) / X_t_std

    if flatten_time_series:
        X_t = X_t.reshape(len(X_t), -1)

    if task < 0:

        y = datafile['adm_labels_all'][:, 1].copy()
        admlbl = datafile['adm_labels_all']
        for i in range(len(y)):
            if admlbl[i][1] > 0:
                y[i] = 1
            elif admlbl[i][2] > 0:
                y[i] = 2
            elif admlbl[i][3] > 0:
                y[i] = 3
            elif admlbl[i][4] > 0:
                y[i] = 4
            elif admlbl[i][5] > 0:
                y[i] = 5
            else:
                y[i] = 0
    else:

        y = datafile['y_icd9'][:, task]

    X_s = torch.tensor(X_s, dtype=torch.float32)
    X_t = torch.tensor(X_t, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    dataset = list(zip(X_s, X_t, y))
    random.seed(10)
    random.shuffle(dataset)

    le = len(dataset)
    test_data = dataset[:le // 5]
    val_data = dataset[le // 5: le // 4]
    train_data = dataset[le // 4:]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    num_classes = len(torch.unique(y))

    with open("samples.txt", "w") as f:
        f.write(f"Total samples: {le}\n")
        f.write(f"Train: {len(train_data)}\n")
        f.write(f"Valid: {len(val_data)}\n")
        f.write(f"Test:  {len(test_data)}\n")
        f.write(f"Num classes: {num_classes}\n")

    return train_loader, val_loader, test_loader, num_classes
