import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger('MMSA')

class MMDataset(Dataset):
    def __init__(self,
                 dataset_name: str,
                 mode: str,
                 feature_path: str,
                 use_bert: bool = True,
                 need_data_aligned: bool = True,
                 need_normalized: bool = False):
        self.dataset_name = dataset_name
        self.mode = mode
        self.use_bert = use_bert
        self.need_data_aligned = need_data_aligned
        self.need_normalized = need_normalized

        with open(feature_path, 'rb') as f:
            data = pickle.load(f)

        self.raw_text = data[mode]['raw_text']
        self.ids = data[mode]['id']
        self.text = data[mode]['text_bert' if use_bert else 'text'].astype(np.float32)
        self.audio = data[mode]['audio'].astype(np.float32)
        self.vision = data[mode]['vision'].astype(np.float32)

        self.labels = {
            'M': np.array(data[mode]['regression_labels']).astype(np.float32)
        }
        if 'sims' in dataset_name.lower():
            for m in 'TAV':
                self.labels[m] = np.array(data[mode][f'regression_labels_{m}']).astype(np.float32)

        logger.info(f"{mode} samples: {self.labels['M'].shape}")

        if not need_data_aligned:
            self.audio_lengths = data[mode]['audio_lengths']
            self.vision_lengths = data[mode]['vision_lengths']

        self.audio[self.audio == -np.inf] = 0

        if need_normalized:
            self._normalize()

    def _normalize(self):
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

    def __len__(self):
        return len(self.labels['M'])

    def __getitem__(self, index):
        sample = {
            'raw_text': self.raw_text[index],
            'id': self.ids[index],
            'index': index,
            'text': torch.tensor(self.text[index]),
            'audio': torch.tensor(self.audio[index]),
            'vision': torch.tensor(self.vision[index]),
            'labels': {k: torch.tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        }
        if not self.need_data_aligned:
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['vision_lengths'] = self.vision_lengths[index]
        return sample['text'], sample['audio'], sample['vision'], sample['labels']['M']

    def get_seq_len(self):
        return (
            self.text.shape[1],
            self.audio.shape[1],
            self.vision.shape[1]
        )

    def get_feature_dim(self):
        return (
            self.text.shape[2],
            self.audio.shape[2],
            self.vision.shape[2]
        )

def get_loader(
    dataset_name='SIMS',
    feature_path='../../data/SIMSv2/unaligned.pkl',
    batch_size=8,
    num_workers=4,
    use_bert=True,
    need_data_aligned=True,
    need_normalized=False
):
    datasets = {
        split: MMDataset(
            dataset_name=dataset_name,
            mode=split,
            feature_path=feature_path,
            use_bert=use_bert,
            need_data_aligned=need_data_aligned,
            need_normalized=need_normalized
        )
        for split in ['train', 'valid', 'test']
    }

    dataloaders = {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=(split == 'train'),
            drop_last=True
        )
        for split in ['train', 'valid', 'test']
    }

    train_len = len(datasets['train'])
    val_len = len(datasets['valid'])
    test_len = len(datasets['test'])
    total_len = train_len + val_len + test_len

    info = (
        f"[INFO] Sample counts:\n"
        f"  Train: {train_len}\n"
        f"  Valid: {val_len}\n"
        f"  Test : {test_len}\n"
        f"  Total: {total_len}\n"
    )
    print(info)
    with open("samples.txt", "w") as f:
        f.write(info)

    return dataloaders['train'], dataloaders['valid'], dataloaders['test']
