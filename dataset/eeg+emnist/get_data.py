import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
import random
import time
from datetime import datetime
import os
import argparse


from utils import CustomTensorDataset_clip, SpikingEMNIST, filter_first_n_letters



def setup_seed(seed):
 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class DatasetFromTensors(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

def parse_args():

    parser = argparse.ArgumentParser(description="EEG-EMNIST Multimodal Data Loading")
    
    # --- 通用参数 ---
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--batch', type=int, default=24, help='批处理大小')
    parser.add_argument('--seed', type=int, default=123, help='随机种子')
    parser.add_argument('--cuda', type=str, default='cuda:0', help='使用的CUDA设备')
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU ID')
    parser.add_argument('--data_num', type=str, default="data_26", choices=["data_10", "data_26"], help='使用10个还是26个字母类别')

    # --- 数据处理参数 ---
    parser.add_argument('--enlarge_eeg_train', type=int, default=1, help='EEG训练集数据增强因子 (已弃用，为兼容性保留)')
    parser.add_argument('--enlarge_eeg_test', type=int, default=1, help='EEG测试集数据增强因子 (已弃用，为兼容性保留)')
    parser.add_argument('--eeg_std', type=float, default=0.0, help='EEG数据增强中的噪声标准差 (已弃用，为兼容性保留)')
    parser.add_argument('--center_crop', type=int, default=28, help='E-MNIST图像中心裁剪尺寸')
    parser.add_argument('--max_spikes', type=int, default=20, help='脉冲化转换时允许的最大脉冲数')
    parser.add_argument('--emnist_window', type=int, default=50, help='E-MNIST脉冲化转换的时间窗口')

    # --- (已弃用) 为兼容性保留的参数 ---
    parser.add_argument('--in_class', type=str, default='', help='已弃用')
    parser.add_argument('--out_class', type=str, default='', help='已弃用')
    
    args = parser.parse_args()
    return args



def data_generate(options):

    NUM_CLASSES = 26 if options.data_num == "data_26" else 10
    
    # --- Step 1: 加载并统一所有数据到一个数据池 ---
    print("--- Step 1: Loading and unifying all data ---")
    
    # EEG 数据
    eeg_mat_data = scipy.io.loadmat('../../datasets/eeg/t5.2019.05.08/singleLetters.mat')
    letters = [chr(ord('a') + i) for i in range(NUM_CLASSES)]
    data_list = [eeg_mat_data[f'neuralActivityCube_{letter}'] for letter in letters]
    labels_list = [[i] * d.shape[0] for i, d in enumerate(data_list)]
    eeg_data_all = torch.FloatTensor(np.concatenate(data_list, axis=0))
    eeg_labels_all = torch.LongTensor(np.concatenate(labels_list, axis=0))
    
    # E-MNIST 数据
    transform = transforms.Compose([
        transforms.Resize(options.center_crop), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])
    emn_train_raw = datasets.EMNIST(root='../../datasets/emnist/data', train=True, split='letters', download=True, transform=transform)
    emn_test_raw = datasets.EMNIST(root='../../datasets/emnist/data', train=False, split='letters', download=True, transform=transform)
    emn_data_all_raw = ConcatDataset([emn_train_raw, emn_test_raw])

    if options.data_num == "data_10":
        emn_data_all_raw = filter_first_n_letters(emn_data_all_raw, NUM_CLASSES)
        
    emn_data_all = SpikingEMNIST(emn_data_all_raw, options.emnist_window, options.max_spikes)

    # --- Step 2: 按类别整理数据 ---
    print("\n--- Step 2: Organizing data by class ---")
    eeg_by_class = [[] for _ in range(NUM_CLASSES)]
    for i in range(len(eeg_data_all)):
        label = eeg_labels_all[i].item()
        processed_data = torch.clamp(eeg_data_all[i], max=1)
        eeg_by_class[label].append(processed_data)
        
    emn_by_class = [[] for _ in range(NUM_CLASSES)]
    for data, label in emn_data_all:
        label = label - 1 # EMNIST 标签是 1-26, 转换为 0-25
        if 0 <= label < NUM_CLASSES:
             emn_by_class[label].append(data)

    # --- Step 3: 数据配对与分层划分 (70% Train, 15% Val, 15% Test) ---
    print("\n--- Step 3: Pairing data and performing stratified split ---")
    train_samples, val_samples, test_samples = [], [], []

    for c in range(NUM_CLASSES):
        eeg_list, emn_list = eeg_by_class[c], emn_by_class[c]
        num_to_pair = min(len(eeg_list), len(emn_list))
        if num_to_pair == 0: 
            
            continue
            
        paired_data = list(zip(eeg_list[:num_to_pair], emn_list[:num_to_pair]))
        random.shuffle(paired_data)
        
        train_end = int(0.70 * num_to_pair)
        val_end = train_end + int(0.15 * num_to_pair)
        
        train_samples.extend([(eeg, emn, c) for eeg, emn in paired_data[:train_end]])
        val_samples.extend([(eeg, emn, c) for eeg, emn in paired_data[train_end:val_end]])
        test_samples.extend([(eeg, emn, c) for eeg, emn in paired_data[val_end:]])

    # --- Step 4: 创建最终的 Dataset 和 DataLoader ---
    print("\n--- Step 4: Creating final datasets and dataloaders ---")
    
    def build_dataloader(samples, batch_size, shuffle):
        if not samples: return None
        random.shuffle(samples)
        eegs, emns, labels = zip(*samples)
        
        dataset = CustomTensorDataset_clip(
            tensors=(torch.stack(eegs, 0), torch.stack(emns, 0), 
                     torch.LongTensor(labels), torch.LongTensor(labels)),
            transform=None
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = build_dataloader(train_samples, options.batch, shuffle=True)
    val_loader = build_dataloader(val_samples, options.batch, shuffle=False)
    test_loader = build_dataloader(test_samples, options.batch, shuffle=False)
    
    print("\nData loading and splitting complete.")
    return train_loader, val_loader, test_loader

# ===================================================================
# 3. 主加载器函数
# ===================================================================

def get_loader():
    """
     orchestrates the entire data loading process.
    """
    options = parse_args()
    print("Parsed Arguments:")
    print(options)

    setup_seed(options.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(options.gpu_id)
    

    train_loader, val_loader, test_loader = data_generate(options)


    train_len = len(train_loader.dataset) if train_loader else 0
    val_len = len(val_loader.dataset) if val_loader else 0
    test_len = len(test_loader.dataset) if test_loader else 0
    total_len = train_len + val_len + test_len

    print("\n" + "="*20 + " Final Sample Counts " + "="*20)
    print(f"Training samples: {train_len}")
    print(f"Validation samples: {val_len}")
    print(f"Testing samples: {test_len}")
    print(f"Total paired samples: {total_len}")


    try:
        with open("samples.txt", "w") as f:
            f.write(f"Training samples: {train_len}\n")
            f.write(f"Validation samples: {val_len}\n")
            f.write(f"Testing samples: {test_len}\n")
            f.write(f"Total paired samples: {total_len}\n")
        print("\nSample counts have been successfully written to samples.txt")
    except IOError as e:
        print(f"\nError writing to file: {e}")

    num_classes = 26 if options.data_num == "data_26" else 10
    return train_loader, val_loader, test_loader, num_classes



if __name__ == '__main__':
    print("--- Starting Data Generation Process ---")
    
    train_loader, val_loader, test_loader, num_classes = get_loader()
    
    print(f"\nSuccessfully created {num_classes}-class dataloaders.")
    if train_loader:
        print("\nVerifying one batch from train_loader...")
        try:
            eeg_batch, emnist_batch, eeg_labels, emnist_labels = next(iter(train_loader))
            print(f"  EEG batch shape: {eeg_batch.shape}")
            print(f"  E-MNIST batch shape: {emnist_batch.shape}")
            print(f"  Labels batch shape: {eeg_labels.shape}")
            assert torch.equal(eeg_labels, emnist_labels)
            print("  Labels for EEG and E-MNIST are correctly paired in the batch.")
        except StopIteration:
            print("  Train loader is empty.")
    
    print("\n--- Data Generation Process Finished ---")