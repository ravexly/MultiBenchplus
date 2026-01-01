import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

class MAMIDataset(Dataset):
   
    def __init__(self, annotations_file, img_dir, transform=None):

        self.data_frame = pd.read_csv(annotations_file, sep='\t')
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()


        img_name = self.data_frame.loc[idx, 'file_name']
        img_path = os.path.join(self.img_dir, img_name)


        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"警告: 图像文件未找到 {img_path}")
            image = Image.new('RGB', (224, 224), (0, 0, 0)) 

        text = self.data_frame.loc[idx, 'text']

        if not isinstance(text, str):
            text = "" 
            
        label = torch.tensor(self.data_frame.loc[idx, 'label'], dtype=torch.float32)


        if self.transform:
            image = self.transform(image)


        return image, text, label


def get_loader(root_dir="/data/xueleyan/MultiBench/data/MAMI/", batch_size=8, num_workers=4, bert_model_path="../../bert-base-uncased"):
 
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def custom_collate_fn(batch):
        images, texts, labels = zip(*batch)
        
 
        images = torch.stack(images, 0)
        labels = torch.stack(labels, 0)
        

        encoded_text = tokenizer.batch_encode_plus(
            texts,
            padding=True,       
            truncation=True,    
            return_tensors='pt' 
        )
        
        return images, encoded_text, labels

    dataloaders = {}
    splits = ['train', 'validation', 'test']

    print("[INFO] Loading datasets...")
    for split in splits:
        annotations_path = os.path.join(root_dir, f"{split}.tsv")
        
        if not os.path.exists(annotations_path):
            print(f"[WARNING] Annotation file not found for split '{split}' at: {annotations_path}")
            dataloaders[split] = None
            continue

        img_dir_name = 'test_images' if split == 'test' else 'training_images'
        img_dir = os.path.join(root_dir, img_dir_name)
            
        dataset = MAMIDataset(
            annotations_file=annotations_path,
            img_dir=img_dir,
            transform=image_transforms
        )
        
        shuffle = (split == 'train')

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn 
        )
        dataloaders[split] = dataloader

    train_len = len(dataloaders['train'].dataset) if dataloaders['train'] else 0
    val_len = len(dataloaders['validation'].dataset) if dataloaders['validation'] else 0
    test_len = len(dataloaders['test'].dataset) if dataloaders['test'] else 0
    total_len = train_len + val_len + test_len
    with open("samples.txt", "w") as f:
        f.write(f"Dataset: MAMI\n")
        f.write("="*20 + "\n")
        f.write(f"Train samples: {train_len}\n")
        f.write(f"Val samples: {val_len}\n")
        f.write(f"Test samples: {test_len}\n")
        f.write(f"Total samples: {total_len}\n")
    
    return dataloaders['train'], dataloaders['validation'], dataloaders['test'], 2