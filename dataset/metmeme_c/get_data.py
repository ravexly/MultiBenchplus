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

        merged_data = pd.merge(self.text_annotations_df, self.label_annotations_df, on='images_name', how='inner')
        merged_data.dropna(inplace=True)
        
        self.text_annotations = merged_data[['images_name', 'text']]
        self.label_annotations = merged_data[['images_name', 'sentiment category']] # Simplified for clarity

        self.root_dir = root_dir
        self.transform = transform
        
        sentiment_labels = self.label_annotations['sentiment category'].apply(lambda x: int(str(x).split('(')[0]))
        self.num_classes = len(sentiment_labels.unique())

    def __len__(self):
        return len(self.text_annotations)

    def __getitem__(self, idx):
        img_name = self.text_annotations.iloc[idx, 0]
        # Dataset-specific name transformation
        img_name_path = img_name.replace('_', '- ') + ".jpg" # Assuming .jpg, adjust if needed
        img_path = os.path.join(self.root_dir, img_name_path)
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except FileNotFoundError:
            print(f"warning:{img_path}")
            # Create a placeholder tensor directly, which is more robust
            image = torch.zeros((3, 224, 224))

        text = self.text_annotations.iloc[idx, 1]
        
        # Ensure text is a string
        if not isinstance(text, str):
            text = ""

        msentiment_label = self.label_annotations.iloc[idx, 1]
        msentiment = int(msentiment_label.split('(')[0]) - 1

        # Convert label to a tensor for stacking in collate_fn
        label = torch.tensor(msentiment, dtype=torch.long)
        
        return image, text, label

def get_loader(
    csv_file_text='../../data/metmeme/C_text.csv', 
    csv_file_label='../../data/metmeme/label_C.csv', 
    root_dir='../../data/metmeme/Cimages/Cimages/Cimages/', 
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
            f.write(f"Dataset: MET-Meme\n")
            f.write("="*20 + "\n")
            f.write(f"Train samples: {len(train_dataset)}\n")
            f.write(f"Validation samples: {len(val_dataset)}\n")
            f.write(f"Test samples: {len(test_dataset)}\n")
            f.write(f"Total samples: {total_len}\n")
            f.write(f"Number of Classes: {num_classes}\n")
    except IOError as e:
        print(f"'samples.txt': {e}")

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