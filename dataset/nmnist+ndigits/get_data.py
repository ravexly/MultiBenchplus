import torch
import time
from datetime import datetime
import os
import argparse
from utils import MyCenterCropDataset, AvgMeter, CustomTensorDataset
from torch.autograd import Variable
from ebdataset.audio import NTidigits
import random
import numpy as np
from quantities import ms, second
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset


def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=500, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--pd', type=int, default=256, help='projection dimension')
    parser.add_argument('--tp', type=float, default=1.0, help='temperature factor for cosine similarities')
    parser.add_argument('--ts', type=int, default=129, help='time_step')
    parser.add_argument('--Dt', type=int, default=1, help='Integration window')
    parser.add_argument('--img_const', type=float, default=0.5, help='image encoder weight constant')
    parser.add_argument('--aud_const', type=float, default=0.5, help='audio encoder weight constant')
    parser.add_argument('--img_decay', type=float, default=0.9, help='image encoder decay')
    parser.add_argument('--aud_decay', type=float, default=0.97, help='audio encoder decay')
    parser.add_argument('--img_vth', type=float, default=5.0, help='image vth')
    parser.add_argument('--aud_vth', type=float, default=5.3, help='audio vth')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    parser.add_argument('--cuda', type=str, default='cuda')
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU ID')

    parser.add_argument('--dm', type=int, default=10, help='每个样本的配对次数 (Duplication/Matching factor)')
    args = parser.parse_args()
    return args

def create_stratified_split_datasets_emulated(img_dataset, aud_dataset, options, split_ratios=(0.7, 0.15, 0.15)):
    """
    <<< 仿照 MNISTSVHN 的 rand_match_on_idx 逻辑实现 >>>
    """
    NUM_CLASSES = 10 
    

    print("Step 1: Organizing all data by class...")
    img_label_extractor = lambda l: torch.argmax(l).item()
    aud_label_extractor = lambda l: int(l) if l.isdigit() else -1

    img_data_by_class = [[] for _ in range(NUM_CLASSES)]
    for i in range(len(img_dataset)):
        data, raw_label = img_dataset[i]
        label = img_label_extractor(raw_label)
        if 0 <= label < NUM_CLASSES:
            img_data_by_class[label].append(data)

    aud_data_by_class = [[] for _ in range(NUM_CLASSES)]
    for i in range(len(aud_dataset)):
        data, raw_label = aud_dataset[i]
        label = aud_label_extractor(raw_label)
        if 0 <= label < NUM_CLASSES:
            padding = torch.zeros(64, options.ts)
            end_row = min(data.shape[0], 64)
            end_col = min(data.shape[1], options.ts)
            padding[:end_row, :end_col] = data[:end_row, :end_col]
            aud_data_by_class[label].append(padding)


    print(f"Step 2: Pairing data (dm={options.dm}) and performing stratified split...")
    train_imgs, val_imgs, test_imgs = [], [], []
    train_auds, val_auds, test_auds = [], [], []
    train_labels, val_labels, test_labels = [], [], []

    classes_to_process = range(1, NUM_CLASSES) # 我们只关心类别 1-9

    for c in classes_to_process:
        img_list = img_data_by_class[c]
        aud_list = aud_data_by_class[c]
        
     
        n = min(len(img_list), len(aud_list))
        if n == 0:
            print(f"  - Warning: No samples to pair for class {c}.")
            continue
        

        img_pool = img_list[:n]
        aud_pool = aud_list[:n]
        
        class_pairs = []
    
        for _ in range(options.dm):

            random.shuffle(img_pool)
            random.shuffle(aud_pool)

            class_pairs.extend(zip(img_pool, aud_pool))
        # <<< MODIFIED LOGIC END >>>


        random.shuffle(class_pairs)
        
        num_pairs = len(class_pairs)
        train_end = int(split_ratios[0] * num_pairs)
        val_end = train_end + int(split_ratios[1] * num_pairs)


        def distribute_pairs(target_pairs, img_list, aud_list, label_list):
            if not target_pairs: return
            imgs, auds = zip(*target_pairs)
            img_list.extend(imgs)
            aud_list.extend(auds)
            label_tensor = torch.tensor(float(c-1))
            label_list.extend([label_tensor] * len(target_pairs))

        distribute_pairs(class_pairs[:train_end], train_imgs, train_auds, train_labels)
        distribute_pairs(class_pairs[train_end:val_end], val_imgs, val_auds, val_labels)
        distribute_pairs(class_pairs[val_end:], test_imgs, test_auds, test_labels)
    

    print("Step 3: Creating final TensorDatasets...")
    def build_dataset(images, audios, labels):
        if not images: return None
        temp = list(zip(images, audios, labels))
        random.shuffle(temp)
        images, audios, labels = zip(*temp)
        
        img_tensor = torch.stack(images, 0)
        aud_tensor = torch.stack(audios, 0)
        label_tensor = torch.stack(labels, 0)
        return CustomTensorDataset(tensors=(img_tensor, aud_tensor, label_tensor, label_tensor), transform=None)

    train_dataset = build_dataset(train_imgs, train_auds, train_labels)
    val_dataset = build_dataset(val_imgs, val_auds, val_labels)
    test_dataset = build_dataset(test_imgs, test_auds, test_labels)

    return train_dataset, val_dataset, test_dataset


def get_loader():
    options = parse_args()
    print(options)
    setup_seed(options.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(options.gpu_id)
    

    print("="*20 + " Loading Raw Datasets " + "="*20)
    train_path = '../../data/NMNIST/NMNIST_train_data.mat'
    test_path = '../../data/NMNIST/NMNIST_test_data.mat'
    nmnist_train_raw = MyCenterCropDataset(train_path, 'nmnist_h', new_width=16, new_height=16)
    nmnist_test_raw = MyCenterCropDataset(test_path, 'nmnist_r', new_width=16, new_height=16)
    
    DATASET_PATH = '../../data/NTIDIGITS/n-tidigits.hdf5'
    dt = int(options.Dt) * ms
    def rec_array_to_spike_train(sparse_spike_train):
        ts = ((sparse_spike_train.ts * second).rescale(dt.units) / dt).magnitude
        duration = np.ceil(np.max(ts)) + 1
        spike_train = torch.zeros((64, int(duration)))
        spike_train[sparse_spike_train.addr, ts.astype(int)] = 1
        spike_train = spike_train.unsqueeze(1).chunk(options.ts, dim=2)
        spike_train_fixed = torch.cat([s.sum(2) for s in spike_train], dim=1)
        spike_train_fixed[spike_train_fixed > 0] = 1
        return spike_train_fixed
    ntidigits_train_raw = NTidigits(DATASET_PATH, is_train=True, transforms=rec_array_to_spike_train, only_single_digits=True)
    ntidigits_test_raw = NTidigits(DATASET_PATH, is_train=False, transforms=rec_array_to_spike_train, only_single_digits=True)
    

    print("\n" + "="*20 + " Combining Datasets " + "="*20)
    all_nmnist_data = ConcatDataset([nmnist_train_raw, nmnist_test_raw])
    all_ntidigits_data = ConcatDataset([ntidigits_train_raw, ntidigits_test_raw])
    print(f"Total N-MNIST samples: {len(all_nmnist_data)}")
    print(f"Total N-TIDIGITS samples: {len(all_ntidigits_data)}\n")


    print("="*20 + " Creating Stratified Splits " + "="*20)
    train_dataset, val_dataset, test_dataset = create_stratified_split_datasets_emulated(
        all_nmnist_data, all_ntidigits_data, options, split_ratios=(0.7, 0.15, 0.15)
    )


    train_loader = DataLoader(train_dataset, batch_size=options.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=options.batch, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=options.batch, shuffle=False)
    

    print("\n" + "="*20 + " Final Sample Counts " + "="*20)
    train_len = len(train_loader.dataset) if train_loader.dataset else 0
    val_len = len(val_loader.dataset) if val_loader.dataset else 0
    test_len = len(test_loader.dataset) if test_loader.dataset else 0
    with open("samples.txt", "w") as f:
        f.write(f"Training samples: {train_len}\n")
        f.write(f"Validation samples: {val_len}\n")
        f.write(f"Testing samples: {test_len}\n")
        f.write(f"Total paired samples: {train_len+val_len+test_len}\n")

    return train_loader, val_loader, test_loader, 9

if __name__ == '__main__':
    train_loader, val_loader, test_loader, num_classes = get_loader()
    print(f"\n{num_classes}")
    
    if train_loader and len(train_loader.dataset) > 0:
        print("batch:")
        for img, aud, label, _ in train_loader:
            print(f" {img.shape}")
            print(f" {aud.shape}")
            print(f" {label.shape}")
            break