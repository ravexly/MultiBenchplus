import os
import sys
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

class TCGADataset(Dataset):
    def __init__(self, X1, X2, X3, X4, X5, X6, labels):
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.X4 = X4
        self.X5 = X5
        self.X6 = X6
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.X1[idx],
            self.X2[idx],
            self.X3[idx],
            self.X4[idx],
            self.X5[idx],
            self.X6[idx],
            self.labels[idx]
        )

def get_loader(batch_size=40, data_dir='../../data/TCGA/', cancer_type='LUAD'):
    cancer_data_dir = os.path.join(data_dir, cancer_type)

    cnv_data = pd.read_csv(os.path.join(cancer_data_dir, 'cnv.csv'), index_col=0)
    meth_data = pd.read_csv(os.path.join(cancer_data_dir, 'meth.csv'), index_col=0)
    mirna_data = pd.read_csv(os.path.join(cancer_data_dir, 'mirna.csv'), index_col=0)
    rnaseq_data = pd.read_csv(os.path.join(cancer_data_dir, 'rnaseq.csv'), index_col=0)
    rppa_data = pd.read_csv(os.path.join(cancer_data_dir, 'rppa.csv'), index_col=0)
    cell_data = pd.read_csv(os.path.join(cancer_data_dir, f'{cancer_type}_cell_composition.csv'), index_col=0)
    survival_data = pd.read_csv(os.path.join(cancer_data_dir, 'survival_clinical.csv'))

    survival_data['Death'] = survival_data['Death'].apply(lambda x: 1 if x > 0 else 0)
    survival_data = survival_data.set_index('Patient Identifier')

    common_ids = cnv_data.index \
        .intersection(meth_data.index) \
        .intersection(mirna_data.index) \
        .intersection(rnaseq_data.index) \
        .intersection(rppa_data.index) \
        .intersection(cell_data.index) \
        .intersection(survival_data.index)

    cnv = torch.tensor(cnv_data.loc[common_ids].values, dtype=torch.float32)
    meth = torch.tensor(meth_data.loc[common_ids].values, dtype=torch.float32)
    mirna = torch.tensor(mirna_data.loc[common_ids].values, dtype=torch.float32)
    rnaseq = torch.tensor(rnaseq_data.loc[common_ids].values, dtype=torch.float32)
    rppa = torch.tensor(rppa_data.loc[common_ids].values, dtype=torch.float32)
    cell = torch.tensor(cell_data.loc[common_ids].values, dtype=torch.float32)
    labels = torch.tensor(survival_data.loc[common_ids]['Death'].values, dtype=torch.long)

    dataset = TCGADataset(cnv, meth, mirna, rnaseq, rppa, cell, labels)

    total_len = len(dataset)
    indices = list(range(total_len))
    random.seed(42)
    random.shuffle(indices)

    test_len = total_len // 5
    val_len = total_len // 4
    train_len = total_len - test_len - val_len

    train_indices = indices[:train_len]
    val_indices = indices[train_len:train_len + val_len]
    test_indices = indices[train_len + val_len:]

    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)


    with open("samples.txt", "w") as f:
        f.write(f"Train samples: {len(train_set)}\n")
        f.write(f"Val samples: {len(val_set)}\n")
        f.write(f"Test samples: {len(test_set)}\n")
        f.write(f"Total samples: {total_len}\n")

    return train_loader, val_loader,  test_loader, 2  # 二分类任务（n_classes=2）
