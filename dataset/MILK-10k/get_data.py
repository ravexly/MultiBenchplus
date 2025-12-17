import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

class MILK10kDataset(Dataset):
    """ PyTorch Dataset for MILK10k """
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        clinic_img = Image.open(row["clinic_path"]).convert("RGB")
        derm_img = Image.open(row["derm_path"]).convert("RGB")

        if self.transform:
            clinic_img = self.transform(clinic_img)
            derm_img = self.transform(derm_img)

        metadata = torch.tensor(row["metadata"], dtype=torch.float32)
        label = torch.tensor(row["diagnosis_encoded"], dtype=torch.long)

        return clinic_img, derm_img, metadata, label

def get_loader(data_dir="../../data/MILK10k", batch_size=32):
    """
    加载MILK10k数据集（包含诊断标签）
    
    文件夹结构:
    data_dir/
    ├── MILK10k_Training_Input/     # 包含所有图像文件夹
    ├── MILK10k_Training_Metadata.csv  # 患者元数据
    └── MILK10k_Training_Supplement.csv # 诊断标签
    """
    
    # 确定文件路径
    metadata_path = os.path.join(data_dir, "MILK10k_Training_Metadata.csv")
    supplement_path = os.path.join(data_dir, "MILK10k_Training_Supplement.csv")
    
    # 检查文件是否存在
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(os.path.dirname(data_dir), "MILK10k_Training_Metadata.csv")
        supplement_path = os.path.join(os.path.dirname(data_dir), "MILK10k_Training_Supplement.csv")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not os.path.exists(supplement_path):
        raise FileNotFoundError(f"Supplement file not found: {supplement_path}")
    
    print(f"[INFO] Loading metadata from: {metadata_path}")
    print(f"[INFO] Loading diagnosis from: {supplement_path}")
    
    # 加载CSV数据
    df_meta = pd.read_csv(metadata_path)
    df_diag = pd.read_csv(supplement_path)
    
    # 图像文件夹路径
    image_root = os.path.join(data_dir, "MILK10k_Training_Input")
    if not os.path.exists(image_root):
        image_root = os.path.join(data_dir, "MILK10k_Training_Input")
        if not os.path.exists(image_root):
            raise FileNotFoundError(f"Image folder not found: {image_root}")
    
    print(f"[INFO] Loading images from: {image_root}")
    
    # 构建样本列表
    samples = []
    lesion_folders = [d for d in os.listdir(image_root) 
                     if os.path.isdir(os.path.join(image_root, d)) and d.startswith("IL_")]
    
    print(f"[INFO] Found {len(lesion_folders)} lesion folders")
    
    found_count = 0
    missing_diag_count = 0
    missing_pair_count = 0
    
    for lesion_folder in lesion_folders:
        lesion_path = os.path.join(image_root, lesion_folder)
        
        # 获取文件夹中的所有jpg文件
        jpg_files = [f for f in os.listdir(lesion_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(jpg_files) < 2:
            missing_pair_count += 1
            continue
        
        # 从元数据中获取这个lesion的信息
        lesion_meta = df_meta[df_meta["lesion_id"] == lesion_folder]
        
        if len(lesion_meta) == 0:
            missing_pair_count += 1
            continue
        
        # 分离临床和皮肤镜图像
        clinical_isic = None
        derm_isic = None
        
        for _, row in lesion_meta.iterrows():
            isic_id = row["isic_id"]
            image_type = row["image_type"]
            
            if "clinical" in image_type:
                clinical_isic = isic_id
            elif "dermoscopic" in image_type:
                derm_isic = isic_id
        
        if not clinical_isic or not derm_isic:
            missing_pair_count += 1
            continue
        
        # 检查诊断信息（至少一个图像有诊断）
        clinical_diag = df_diag[df_diag["isic_id"] == clinical_isic]
        derm_diag = df_diag[df_diag["isic_id"] == derm_isic]
        
        # 确定使用哪个诊断（优先使用临床图像的诊断）
        if len(clinical_diag) > 0:
            diagnosis_row = clinical_diag.iloc[0]
        elif len(derm_diag) > 0:
            diagnosis_row = derm_diag.iloc[0]
        else:
            missing_diag_count += 1
            continue  # 两个图像都没有诊断，跳过
        
        # 构建完整路径
        clinical_path = os.path.join(lesion_path, f"{clinical_isic}.jpg")
        derm_path = os.path.join(lesion_path, f"{derm_isic}.jpg")
        
        # 检查文件是否存在，如果.jpg不存在，尝试其他扩展名
        if not os.path.exists(clinical_path):
            # 查找实际的文件名（可能不是.jpg）
            clinical_files = [f for f in jpg_files if clinical_isic in f]
            if clinical_files:
                clinical_path = os.path.join(lesion_path, clinical_files[0])
            else:
                missing_pair_count += 1
                continue
        
        if not os.path.exists(derm_path):
            derm_files = [f for f in jpg_files if derm_isic in f]
            if derm_files:
                derm_path = os.path.join(lesion_path, derm_files[0])
            else:
                missing_pair_count += 1
                continue
        
        # 获取第一条元数据记录
        meta_row = lesion_meta.iloc[0]
        
        # 提取诊断信息，处理可能的缺失值
        diagnosis_full = str(diagnosis_row.get("diagnosis_full", "unknown")).strip()
        diagnosis_confirm = str(diagnosis_row.get("diagnosis_confirm_type", "unknown")).strip()
        
        if diagnosis_full.lower() == "nan" or diagnosis_full == "":
            diagnosis_full = "unknown"
        
        samples.append({
            "lesion_id": lesion_folder,
            "clinical_isic": clinical_isic,
            "derm_isic": derm_isic,
            "clinic_path": clinical_path,
            "derm_path": derm_path,
            "age_approx": meta_row.get("age_approx", 50),
            "sex": meta_row.get("sex", "unknown"),
            "site": meta_row.get("site", ""),
            "skin_tone_class": meta_row.get("skin_tone_class", 3),
            "diagnosis_full": diagnosis_full,
            "diagnosis_confirm_type": diagnosis_confirm
        })
        found_count += 1
    
    if len(samples) == 0:
        raise ValueError(f"No valid samples with diagnosis found")
    
    # 创建DataFrame
    df = pd.DataFrame(samples)
    
    # 处理缺失值
    # 年龄：用中位数填充
    if df['age_approx'].isna().any():
        age_median = df['age_approx'].median()
        df['age_approx'] = df['age_approx'].fillna(age_median)
    
    # 肤色：用中位数填充
    if df['skin_tone_class'].isna().any():
        skin_median = df['skin_tone_class'].median()
        df['skin_tone_class'] = df['skin_tone_class'].fillna(skin_median)
    
    
    # 使用完整的诊断名称作为标签（不进行简化）
    label_encoder = LabelEncoder()
    df["diagnosis"] = df["diagnosis_full"]
    df["diagnosis_encoded"] = label_encoder.fit_transform(df["diagnosis"])
    
    # 编码元数据
    # 编码性别（处理可能的缺失值）
    df["sex"] = df["sex"].str.lower().fillna("unknown")
    sex_mapping = {"male": 0, "female": 1, "unknown": 2}
    df["sex_encoded"] = df["sex"].map(lambda x: sex_mapping.get(x, 2))
    
    # 编码部位（从CSV中读取所有可能的部位）
    site_mapping = {}
    unique_sites = df["site"].dropna().unique()
    for i, site in enumerate(sorted(unique_sites)):
        site_mapping[site] = i
    
    # 添加空白部位
    site_mapping[""] = len(site_mapping)
    df["site_encoded"] = df["site"].map(lambda x: site_mapping.get(x, site_mapping[""]))
    
    # 归一化年龄和肤色
    age_mean = df["age_approx"].mean()
    age_std = df["age_approx"].std()
    df["age_norm"] = (df["age_approx"] - age_mean) / age_std if age_std > 0 else 0
    
    skin_mean = df["skin_tone_class"].mean()
    skin_std = df["skin_tone_class"].std()
    df["skin_tone_norm"] = (df["skin_tone_class"] - skin_mean) / skin_std if skin_std > 0 else 0
    
    # 创建元数据向量 [sex_encoded, site_encoded, age_norm, skin_tone_norm]
    df["metadata"] = df.apply(
        lambda row: [row["sex_encoded"], row["site_encoded"], 
                    row["age_norm"], row["skin_tone_norm"]], 
        axis=1
    )
    
    # 按诊断类别分层划分数据集
    print(f"[INFO] Stratified splitting by diagnosis...")
    
    # 获取所有诊断类别
    unique_diagnoses = df["diagnosis"].unique()
    print(f"[INFO] Found {len(unique_diagnoses)} unique diagnosis categories")
    
    # 分层抽样
    train_indices = []
    valid_indices = []
    test_indices = []
    
    for diagnosis in unique_diagnoses:
        diagnosis_df = df[df["diagnosis"] == diagnosis]
        indices = diagnosis_df.index.tolist()
        n_samples = len(indices)
        
        if n_samples < 3:
            # 如果样本太少，全部放入训练集
            train_indices.extend(indices)
            continue
        
        # 划分：70%训练，15%验证，15%测试
        try:
            train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=[diagnosis]*n_samples)
            valid_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=[diagnosis]*len(temp_idx))
            
            train_indices.extend(train_idx)
            valid_indices.extend(valid_idx)
            test_indices.extend(test_idx)
        except:
            # 如果分层失败，使用随机划分
            np.random.seed(42)
            np.random.shuffle(indices)
            train_size = int(0.7 * n_samples)
            valid_size = int(0.15 * n_samples)
            
            train_indices.extend(indices[:train_size])
            valid_indices.extend(indices[train_size:train_size + valid_size])
            test_indices.extend(indices[train_size + valid_size:])
    
    df_train = df.loc[train_indices].reset_index(drop=True)
    df_valid = df.loc[valid_indices].reset_index(drop=True)
    df_test = df.loc[test_indices].reset_index(drop=True)
    
    # 10. 图像变换
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 11. 创建数据集
    train_dataset = MILK10kDataset(df_train, transform=image_transform)
    valid_dataset = MILK10kDataset(df_valid, transform=image_transform)
    test_dataset = MILK10kDataset(df_test, transform=image_transform)
    
    # 12. 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    
    n_classes = len(label_encoder.classes_)
    
    print(f"[INFO] Sample counts:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Valid: {len(valid_dataset)}")
    print(f"  Test : {len(test_dataset)}")
    print(f"  Number of classes: {n_classes}")

    with open("samples.txt", "w") as f:
        f.write(f"Train: {len(train_dataset)}\n")
        f.write(f"Valid: {len(valid_dataset)}\n")
        f.write(f"Test: {len(test_dataset)}\n")
        f.write(f"Classes: {n_classes}\n")
    
    return train_loader, valid_loader, test_loader, n_classes

if __name__ == "__main__":
    # 使用示例
    train_loader, valid_loader, test_loader, n_classes = get_loader()
    print("Train batch example shapes:")
    for batch in train_loader:
        print(batch[0].shape, batch[1].shape, batch[2].shape,batch[3].shape)
        break