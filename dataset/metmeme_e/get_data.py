import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import torch
from transformers import BertTokenizer

class METMemeDataset(Dataset):
    
    def __init__(self, csv_file_text, csv_file_label, root_dir, transform=None):
       
        self.text_annotations_df = pd.read_csv(csv_file_text, encoding='latin1')
        self.text_annotations_df.dropna(axis=1, how='all', inplace=True)

        try:
            self.label_annotations_df = pd.read_csv(csv_file_label, encoding='utf-8')
        except UnicodeDecodeError:
            self.label_annotations_df = pd.read_csv(csv_file_label, encoding='ISO-8859-1')
        self.label_annotations_df.dropna(axis=1, how='all', inplace=True)

        # 基于 'file_name' 合并两个 DataFrame，确保数据对齐
        merged_data = pd.merge(self.text_annotations_df, self.label_annotations_df, on='file_name', how='inner')
        
        merged_data.dropna(inplace=True)
        
        # 仅保留需要的列以提高效率
        self.annotations = merged_data[['file_name', 'text', 'sentiment category']]

        self.root_dir = root_dir
        self.transform = transform
        
        # 动态计算类别数量
        sentiment_labels = self.annotations['sentiment category'].apply(lambda x: int(str(x).split('(')[0]))
        self.num_classes = len(sentiment_labels.unique())

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_name = row['file_name']
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except FileNotFoundError:
            print(f"warning:{img_path}")
            # 创建一个占位符张量，使其更健壮
            image = torch.zeros((3, 224, 224))

        text = row['text']
        # 确保文本是字符串类型
        if not isinstance(text, str):
            text = ""

        # 提取 'sentiment category' 标签并将其转换为从 0 开始的索引
        msentiment_label = row['sentiment category']
        msentiment = int(msentiment_label.split('(')[0]) - 1  # 转换为 0-based

        # 将标签转换为张量，以便在collate_fn中进行堆叠
        label = torch.tensor(msentiment, dtype=torch.long)
            
        return image, text, label

def get_loader(
    csv_file_text='../../data/metmeme/E_text.csv', 
    csv_file_label='../../data/metmeme/label_E.csv', 
    root_dir='../../data/metmeme/Eimages/Eimages/Eimages/', 
    bert_model_path="../../bert-base-uncased",
    batch_size=8, 
    num_workers=4, 
    train_split=0.7, 
    val_split=0.1
):
   
    # 1. 初始化BERT分词器
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = METMemeDataset(
        csv_file_text=csv_file_text,
        csv_file_label=csv_file_label,
        root_dir=root_dir,
        transform=transform
    )
    
    num_classes = dataset.num_classes

    total_len = len(dataset)
    train_len = int(train_split * total_len)
    val_len = int(val_split * total_len)
    test_len = total_len - train_len - val_len

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_len, val_len, test_len], generator=generator
    )

    try:
        with open("samples.txt", "w") as f:
            f.write(f"Dataset: MET-Meme (E)\n")
            f.write("="*20 + "\n")
            f.write(f"Train samples: {len(train_dataset)}\n")
            f.write(f"Validation samples: {len(val_dataset)}\n")
            f.write(f"Test samples: {len(test_dataset)}\n")
            f.write(f"Total samples: {total_len}\n")
            f.write(f"Number of Classes: {num_classes}\n")
    except IOError as e:
        print(f" 'samples.txt': {e}")

    # 2. 定义新的collate_fn来处理文本分词和数据打包
    def custom_collate_fn(batch):

        images, texts, labels = zip(*batch)
        
        images = torch.stack(images, 0)
        labels = torch.stack(labels, 0)
        
        encoded_text = tokenizer.batch_encode_plus(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # 按要求返回一个列表
        return [images, encoded_text, labels]

    # 3. 创建使用新collate_fn的DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, drop_last=True, collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, drop_last=True, collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, drop_last=True, collate_fn=custom_collate_fn
    )

    return train_loader, val_loader, test_loader, num_classes