import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings
import pandas as pd
from skimage import io, transform
from rasterio.features import rasterize
from shapely.ops import cascaded_union, unary_union
from shapely.geometry import Polygon
from shapely.errors import ShapelyDeprecationWarning
from PIL import Image
import numpy as np
import torch.nn as nn
import pickle
warnings.filterwarnings("ignore", category=FutureWarning)  # 忽略 PIL 的警告


class ForestDataset(Dataset):
    """Forest dataset."""

    def __init__(self, mode, root_dir='../../data/ForestNetDataset', transform=None, types="classifier"):
        if mode=='train':
            self.csv = pd.read_csv('../../data/ForestNetDataset/train.csv')
        elif mode== 'val':
            self.csv = pd.read_csv('../../data/ForestNetDataset/val.csv')
        elif mode=='test':
            self.csv = pd.read_csv('../../data/ForestNetDataset/test.csv')

        self.root_dir = root_dir
        self.transform = transform
        self.label_to_int = {'Grassland shrubland':0, 'Other':1, 'Plantation':2, 'Smallholder agriculture':3}
        self.types = types

    def __len__(self):
        return len(self.csv)
    
    def poly_from_utm(self, polygon):
        poly_pts = []

        poly = unary_union(polygon)
        for i in np.array(poly.exterior.coords):

            poly_pts.append(tuple(i))

        new_poly = Polygon(poly_pts)
        return new_poly

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.csv.iloc[idx, 0]
        merged_label = self.csv.iloc[idx, 1]
        lat = self.csv.iloc[idx, 2]
        long = self.csv.iloc[idx, 3]
        year = self.csv.iloc[idx, 4]
        folder = self.csv.iloc[idx, 5]
        
        ## Load the image and auxiliary
        image_path = f'{self.root_dir}/{folder}/images/visible/composite.png'
        image = Image.open(image_path)
        slope = np.load(f'{self.root_dir}/{folder}/auxiliary/srtm.npy')
        
        ## Get the segmentation map
        with open(f'{self.root_dir}/{folder}/forest_loss_region.pkl', 'rb') as f:
            data = pickle.load(f)
    
        nx, ny = 332, 332
        xy_array = np.empty((0,2))
        # if data.geom_type == 'Polygon':
        #     data = [data]
        # elif data.geom_type == 'MultiPolygon':
        #     data = list(data)
                
        # poly_shp = []
        # for poly_verts in data:
        #     poly_shp.append(self.poly_from_utm(poly_verts))

        # mask = rasterize(shapes=poly_shp, out_shape=(332,332))
        # seg = np.array(mask)
        
        if self.transform:
            image = self.transform(image)
            
        
        # image = image.permute(2, 0, 1)
        # seg = torch.from_numpy(seg).type(torch.uint8)
        slope = torch.from_numpy(slope).type(torch.float)

        merged_label = self.label_to_int[merged_label]

        # image = image[:, 86:246, 86:246]
        # seg = seg[86:246, 86:246]
        # slope = slope[86:246, 86:246]
        if self.types == "classifier":
            return image, slope, merged_label
        else:
            return image, merged_label
def get_loader(root_dir='../../data/ForestNetDataset',
               batch_size=8,
               num_workers=4,
               image_size=224,
               types='classifier'):
    """
    返回 train_loader, val_loader, test_loader, n_classes
    并把样本数量写入 samples.txt
    """
    # 1. 定义统一的 transform
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 2. 构建数据集
    train_dataset = ForestDataset(mode='train', root_dir=root_dir,
                                  transform=transform, types=types)
    val_dataset   = ForestDataset(mode='val',   root_dir=root_dir,
                                  transform=transform, types=types)
    test_dataset  = ForestDataset(mode='test',  root_dir=root_dir,
                                  transform=transform, types=types)

    # 3. 构建 DataLoader
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              drop_last=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            drop_last=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             drop_last=True)

    # 4. 类别数
    n_classes = len(train_dataset.label_to_int)

    # 5. 写出样本数量
    with open("samples.txt", "w") as f:
        f.write(f"Train: {len(train_dataset)}\n")
        f.write(f"Validation: {len(val_dataset)}\n")
        f.write(f"Test: {len(test_dataset)}\n")
        f.write(f"Total: {len(train_dataset)+len(val_dataset)+len(test_dataset)}\n")

    return train_loader, val_loader, test_loader, n_classes


# ===== 使用示例 =====
if __name__ == '__main__':
    tr_loader, va_loader, te_loader, cls = get_loader(batch_size=8)
    print(f"num_classes = {cls}")