
import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2.functional import crop

try:
    from rs_fusion_datasets import AugsburgOuc
except ImportError:
    print("[Warning] 'rs_fusion_datasets' not found. Using a mock class for demonstration.")
    # --- Mock Class for Demonstration ---
    class AugsburgOuc:
        def __init__(self, split='train', patch_size=5):
            self.patch_size = patch_size
            self.split = split
            # Based on your data: HSI: 180 channels, DSM: 4 channels
            self.hsi = torch.randn(180, 100, 100) 
            self.dsm = torch.randn(4, 100, 100)
            num_samples = 761 if split == 'train' else 77533
            self.truth = torch.randint(1, 8, (num_samples,)) # 7 classes, labels 1-7
            self.lbl = type('obj', (object,), {
                'row': torch.randint(0, 100 - patch_size, (num_samples,)),
                'col': torch.randint(0, 100 - patch_size, (num_samples,))
            })()
            self.INFO = {'n_class': 7}
        
        def __len__(self):
            return len(self.truth)
    # --- End of Mock Class ---

class MyAugsburgOuc(AugsburgOuc):
    def __getitem__(self, index):
        w = self.patch_size
        i = self.lbl.row[index]
        j = self.lbl.col[index]
        x_hsi = crop(self.hsi, i, j, w, w)
        x_dsm = crop(self.dsm, i, j, w, w)
        label = self.truth.data[index].item() - 1


        return x_hsi, x_dsm, label


def get_loader(batch_size=128):
    trainset = MyAugsburgOuc('train', patch_size=5)
    testset = MyAugsburgOuc('test', patch_size=5)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=True)

    train_len = len(trainset)
    test_len = len(testset)
    total_len = train_len + test_len

    print(f"[INFO] Sample counts:")
    print(f"   Train: {train_len}")
    print(f"   Test : {test_len}")
    print(f"   Total: {total_len}")

    return train_loader, test_loader, trainset.INFO['n_class']