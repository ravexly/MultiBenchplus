import os
import json
import pickle
from collections import defaultdict
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm
import ssl
from transformers import BertTokenizer 

ssl._create_default_https_context = ssl._create_unverified_context  


class Cub(Dataset):
    resources = ["http://www.robots.ox.ac.uk/~yshi/mmdgm/datasets/cub.zip"]

    def __init__(self, root, download=False, split="all", transform=None):
        """
        split: "train" | "val" | "test" | "all"
        """
        self.root = root
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found. Use download=True to download it.")

        # 默认图像变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform 

        train_dir = os.path.join(self.raw_folder, "cub/train")
        test_dir = os.path.join(self.raw_folder, "cub/test")
        self.images = []
        self.labels = []
        self.img_paths = []
        

        for split_dir in [train_dir, test_dir]:
            if not os.path.isdir(split_dir):
                continue
            # Note: We don't apply transform here, ImageFolder is just for file discovery
            temp_ds = datasets.ImageFolder(split_dir)
            self.images.extend(temp_ds.imgs)  # (path, class_index)
            self.img_paths.extend([p for p, _ in temp_ds.imgs])
            self.labels.extend([l for _, l in temp_ds.imgs])


        self.sentences = CUBSentences(self.raw_folder)


        self.split = split
        self.indices = self._make_split_indices() 
        self.n_classes = len(set(self.labels))

    # ----------------------------------------------------------
    def _make_split_indices(self):

        n_imgs = len(self.images)
        img_indices = np.arange(n_imgs)
        np.random.seed(2025)
        np.random.shuffle(img_indices)

        train_end = int(0.70 * n_imgs)
        val_end = int(0.85 * n_imgs)

        if self.split == "train":
            chosen_img_idx = img_indices[:train_end]
        elif self.split == "val":
            chosen_img_idx = img_indices[train_end:val_end]
        elif self.split == "test":
            chosen_img_idx = img_indices[val_end:]
        else: 
            chosen_img_idx = img_indices

 
        caption_indices = []
        for idx in chosen_img_idx:
            base = idx * 10
            caption_indices.extend(range(base, base + 10))
        return caption_indices

    # ----------------------------------------------------------
    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def _check_raw_exists(self):
        return os.path.exists(os.path.join(self.raw_folder, "cub.zip"))

    def _check_exists(self):
        return os.path.exists(os.path.join(self.raw_folder, "cub"))

    def download(self):
        if self._check_exists():
            print("Files already downloaded and verified.")
            return
        os.makedirs(self.raw_folder, exist_ok=True)
        for url in self.resources:
            filename = url.rpartition("/")[2]
            print(f"Downloading {filename}...")
            download_and_extract_archive(url, download_root=self.raw_folder,
                                         filename=filename)
        print("Extraction complete.")

    # ----------------------------------------------------------
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        caption_idx = self.indices[idx]
        img_idx = caption_idx // 10 

        img_path, label = self.images[img_idx]
        
        try:
            img = torchvision.datasets.folder.default_loader(img_path)
            if self.transform:
                img = self.transform(img)
        except (FileNotFoundError, IOError):
            print(f"Warning: Image file not found {img_path}. Using a placeholder.")
            img = torch.zeros((3, 256, 256)) # Placeholder

        caption = self.sentences[caption_idx] 
        
        return img, caption, label



class CUBSentences:
    def __init__(self, root_data_dir):
        self.data_dir = os.path.join(root_data_dir, "cub")

        files = ["text_trainvalclasses.txt", "text_testclasses.txt"]
        self.data = []
        for fname in files:
            path = os.path.join(self.data_dir, fname)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc=f"Loading {fname}"):
                    self.data.append(line.strip())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



def get_loader(
    root="../../data/Cub", 
    batch_size=16, 
    download=True,
    bert_model_path="../../bert-base-uncased"  
):
    print("Loading CUB Dataset (7:1.5:1.5 split)...")


    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = Cub(root=root, download=download, split="train", transform=transform)
    val_ds = Cub(root=root, download=download, split="val", transform=transform)
    test_ds = Cub(root=root, download=download, split="test", transform=transform)


    try:
        tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    except Exception as e:
        print(f"Error loading tokenizer from {bert_model_path}: {e}")
        print("Please ensure 'transformers' is installed and the path is correct.")
        return None, None, None, 0


    def custom_collate_fn(batch):
        images, captions, labels = zip(*batch)

  
        images = torch.stack(images, 0)

        labels = torch.tensor(labels, dtype=torch.long)


        encoded_text = tokenizer.batch_encode_plus(
            list(captions),
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        

        return [images, encoded_text, labels]


    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=custom_collate_fn
    )

    n_classes = train_ds.n_classes

    try:
        with open("samples.txt", "w") as f:
            f.write("Dataset: CUB (Caltech-UCSD Birds-200-2011) - New 70/15/15 Split\n")
            f.write("=" * 55 + "\n")
            f.write(f"Number of classes: {n_classes}\n")
            f.write(f"Train samples (captions): {len(train_ds)}\n")
            f.write(f"Val   samples (captions): {len(val_ds)}\n")
            f.write(f"Test  samples (captions): {len(test_ds)}\n")
            f.write(f"Total samples (captions): {len(train_ds) + len(val_ds) + len(test_ds)}\n")
    except IOError as e:
        print(f"Error writing to samples.txt: {e}")

    return train_loader, val_loader, test_loader, n_classes



if __name__ == "__main__":
    BERT_PATH = "../../bert-base-uncased" 
    
    try:
        tr, va, te, nc = get_loader(
            batch_size=4, 
            bert_model_path=BERT_PATH,
            download=True 
        )
        
        if tr:
            print(f"\n✅ Successfully created data loaders and 'samples.txt'.")
            print(f"Number of classes: {nc}")
            print("-" * 30)

            print("Testing the val_loader... 🧪")
            batch_list = next(iter(va))

            images, texts, labels = batch_list
            
            print(f"Batch format is a list of length: {len(batch_list)}")
            print(f"Image batch shape: {images.shape}")
            print(f"Labels batch shape: {labels.shape}")
            print(f"Text batch is a dictionary with keys: {texts.keys()}")
            print(f"Text input_ids shape: {texts['input_ids'].shape}")
            print(f"Example label: {labels[0].item()}")
            print("-" * 30)
            
            print(f"Train loader batches: {len(tr)}")
            print(f"Validation loader batches: {len(va)}")
            print(f"Test loader batches: {len(te)}")

    except ImportError as e:
        print(f"❌ Error: A required package is missing. {e}")
    except FileNotFoundError as e:
        print(f"❌ Error: A path is incorrect (e.g., BERT model or data path). {e}")