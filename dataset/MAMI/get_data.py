import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

class MAMIDataset(Dataset):
    """
    用于MAMI多模态数据集的自定义PyTorch Dataset。
    此版本返回原始文本字符串，分词操作将在collate_fn中批量处理。
    
    Args:
        annotations_file (str): 包含标注的CSV文件路径。
        img_dir (str): 包含所有图像的目录路径。
        transform (callable, optional): 应用于图像样本的可选转换。
    """
    def __init__(self, annotations_file, img_dir, transform=None):
        # 使用'\t'作为分隔符读取CSV文件
        self.data_frame = pd.read_csv(annotations_file, sep='\t')
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 如果idx是tensor，则转换为int
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 获取图像路径并构建完整路径
        img_name = self.data_frame.loc[idx, 'file_name']
        img_path = os.path.join(self.img_dir, img_name)

        # 加载图像并转换为RGB
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"警告: 图像文件未找到 {img_path}")
            image = Image.new('RGB', (224, 224), (0, 0, 0)) # 加载黑色图像作为替代

        # 获取原始文本和标签
        text = self.data_frame.loc[idx, 'text']
        # 确保文本是字符串类型
        if not isinstance(text, str):
            text = "" # 如果文本是缺失值（如NaN），则替换为空字符串
            
        label = torch.tensor(self.data_frame.loc[idx, 'label'], dtype=torch.float32)

        # 应用图像转换
        if self.transform:
            image = self.transform(image)

        # 依旧返回 图像Tensor, 原始文本字符串, 标签Tensor
        return image, text, label


def get_loader(root_dir="/data/xueleyan/MultiBench/data/MAMI/", batch_size=8, num_workers=4, bert_model_path="../../bert-base-uncased"):
    """
    创建并返回用于训练、验证和测试的DataLoader。
    此版本使用BERT对文本进行分词，并返回一个包含所有模态的字典。

    Args:
        root_dir (str): 存放 train.csv, valid.csv, test.csv 的目录。
        batch_size (int): 每个批次的样本数。
        num_workers (int): 用于数据加载的子进程数。
        bert_model_path (str): 预训练BERT模型和分词器的路径。

    Returns:
        tuple: 包含训练、验证和测试的DataLoader (train_loader, valid_loader, test_loader)。
    """
    # 1. 初始化BERT分词器
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    
    # 定义图像的预处理流程
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. 定义新的collate_fn来处理文本分词和数据打包
    def custom_collate_fn(batch):
        """
        处理一个批次的数据。
        - 将图像和标签堆叠成批次张量。
        - 使用分词器处理文本列表，生成文本字典。
        - 返回一个元组 (images, encoded_text, labels)。
        """
        images, texts, labels = zip(*batch)
        
        # 堆叠图像和标签
        images = torch.stack(images, 0)
        labels = torch.stack(labels, 0)
        
        # 使用分词器批量处理文本
        encoded_text = tokenizer.batch_encode_plus(
            texts,
            padding=True,       # 填充到批次中的最大长度
            truncation=True,    # 截断超过最大长度的文本
            return_tensors='pt' # 返回PyTorch张量
        )
        
        # 3. 将所有数据打包成一个元组返回
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
            collate_fn=custom_collate_fn  # 使用新的collate函数
        )
        dataloaders[split] = dataloader

    # 打印样本数量统计信息 (这部分保持不变)
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