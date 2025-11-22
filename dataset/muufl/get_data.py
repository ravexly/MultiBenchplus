import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2.functional import crop
from rs_fusion_datasets import Muufl


class MyMuufl(Muufl):
    def __getitem__(self, index):
        w = self.patch_size
        i = self.lbl.row[index]
        j = self.lbl.col[index]
        x_hsi = crop(self.hsi, i, j, w, w)
        x_dsm = crop(self.dsm, i, j, w, w)
        label = self.truth.data[index].item() - 1
        return x_hsi, x_dsm, label


def get_loader(batch_size=128):
    trainset = MyMuufl('train', patch_size=5)
    testset = MyMuufl('test', patch_size=5)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=True)

    train_len = len(trainset)
    test_len = len(testset)
    total_len = train_len + test_len

    print(f"[INFO] Sample counts:")
    print(f"  Train: {train_len}")
    print(f"  Test : {test_len}")
    print(f"  Total: {total_len}")

    with open("samples.txt", "w") as f:
        f.write(f"Train: {train_len}\n")
        f.write(f"Test : {test_len}\n")
        f.write(f"Total: {total_len}\n")

    return train_loader, test_loader, trainset.INFO['n_class']
