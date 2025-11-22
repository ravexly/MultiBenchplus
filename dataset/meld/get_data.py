import os
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np


class BimodalDataset(Dataset):
    def __init__(self, mode='train', task='sentiment'):
        super().__init__()
        assert mode in ['train', 'val', 'test'], "mode must be 'train', 'val', or 'test'"

        text_path = f'../../data/MELD/features/text_glove_average_{task}.pkl'
        audio_path = f'../../data/MELD/features/audio_embeddings_feature_selection_{task}.pkl'
        self.text_data = pickle.load(open(text_path, 'rb'), encoding='latin1')[{'train': 0, 'val': 1, 'test': 2}[mode]]
        self.audio_data = pickle.load(open(audio_path, 'rb'), encoding='latin1')[{'train': 0, 'val': 1, 'test': 2}[mode]]

        data_p_path = f'../../data/MELD/features/data_{task}.p'
        revs, _, _, _, _, label_index = pickle.load(open(data_p_path, 'rb'))
        self.num_classes = len(label_index)

        self.label_dict = {}
        for entry in revs:
            if entry['split'] != mode:
                continue
            uid = f"{entry['dialog']}_{entry['utterance']}"
            self.label_dict[uid] = label_index[entry['y']]


        self.samples = [uid for uid in self.text_data
                        if uid in self.label_dict and uid in self.audio_data]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        uid = self.samples[idx]
        text_feat = torch.tensor(self.text_data[uid], dtype=torch.float32)
        audio_feat = torch.tensor(self.audio_data[uid], dtype=torch.float32)
        label = torch.tensor(self.label_dict[uid], dtype=torch.long)  # 类别索引
        return audio_feat, text_feat, label


def get_loader(task='sentiment',
               batch_size=32,
               shuffle=True,
               num_workers=0):

    train_dataset = BimodalDataset(mode='train', task=task)
    val_dataset   = BimodalDataset(mode='val',   task=task)
    test_dataset  = BimodalDataset(mode='test',  task=task)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=shuffle, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    n_classes = train_dataset.num_classes


    with open("samples.txt", "w") as f:
        f.write(f"Train: {len(train_dataset)}\n")
        f.write(f"Validation: {len(val_dataset)}\n")
        f.write(f"Test: {len(test_dataset)}\n")
        f.write(f"Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)}\n")

    return train_loader, val_loader, test_loader, n_classes


