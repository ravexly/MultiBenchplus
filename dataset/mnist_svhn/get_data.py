import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms

# 忽略下载日志
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ------------- MNISTSVHN 数据集类 -------------
class MNISTSVHN(Dataset):


    def __init__(
        self,
        root: str,
        max_d: int = 10000,
        dm: int = 10,
        train: bool = True,
        download: bool = True,
        flatten: bool = False,
    ):
        super().__init__()
        self.flatten = flatten
        self.mnist = self.load_mnist(train, root, download)
        self.svhn = self.load_svhn(train, root, download)
        self.mnist_index, self.svhn_index = self.rand_match_on_idx(max_d=max_d, dm=dm)

    def __len__(self):
        return len(self.mnist_index)

    def __getitem__(self, idx):
        x, label = self.mnist[self.mnist_index[idx]]
        y, label_ = self.svhn[self.svhn_index[idx]]
        assert label == label_, "Label mismatch!"
        if self.flatten:
            x = torch.flatten(x)
            y = torch.flatten(y)
        return torch.flatten(x), y, label  # MNIST_flattened, SVHN_image, label

    @staticmethod
    def load_mnist(train, root, download):
        return datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.Compose([transforms.ToTensor()])
        )

    @staticmethod
    def load_svhn(train, root, download):
        split = "train" if train else "test"
        svhn = datasets.SVHN(
            root=root,
            split=split,
            download=download,
            transform=transforms.Compose([transforms.ToTensor()])
        )
        svhn.labels = torch.LongTensor(svhn.labels.squeeze().astype(int)) % 10
        return svhn

    def rand_match_on_idx(self, max_d=10000, dm=10):
        mnist_l, mnist_li = self.mnist.targets.sort()
        svhn_l, svhn_li = self.svhn.labels.sort()
        _idx1, _idx2 = [], []
        for l in mnist_l.unique():  # 0~9
            l_idx1 = mnist_li[mnist_l == l]
            l_idx2 = svhn_li[svhn_l == l]
            n = min(l_idx1.size(0), l_idx2.size(0), max_d)
            l_idx1, l_idx2 = l_idx1[:n], l_idx2[:n]
            for _ in range(dm):
                _idx1.append(l_idx1[torch.randperm(n)])
                _idx2.append(l_idx2[torch.randperm(n)])
        return torch.cat(_idx1), torch.cat(_idx2)


# ------------- 新增 get_loader -------------
def get_loader(
    root= "/data/xueleyan/MultiBench/data/MNISTSVHN/",
    batch_size: int = 64,
    max_d: int = 10000,
    dm: int = 10,
    num_workers: int = 4,
    download: bool =True,
):

    train_dataset = MNISTSVHN(
        root=root,
        max_d=max_d,
        dm=dm,
        train=True,
        download=download,
        flatten=False,
    )
    test_dataset = MNISTSVHN(
        root=root,
        max_d=max_d,
        dm=dm,
        train=False,
        download=download,
        flatten=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    n_classes = 10 


    with open("samples.txt", "w") as f:
        f.write(f"Train: {len(train_dataset)}\n")
        f.write(f"Test: {len(test_dataset)}\n")
        f.write(f"Total: {len(train_dataset) + len(test_dataset)}\n")

    return train_loader, test_loader,test_loader, n_classes



if __name__ == "__main__":
    tr_loader, te_loader, cls = get_loader(batch_size=256)
    print(f"num_classes = {cls}")
    for mnist_flat, svhn_img, label in tr_loader:
        print("MNIST shape:", mnist_flat.shape)   # [B, 784]
        print("SVHN  shape:", svhn_img.shape)     # [B, 3, 32, 32]
        print("Label shape:", label.shape)        # [B]
        break