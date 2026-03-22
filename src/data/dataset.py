import os
import io
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

def load_and_prep_data(train_path, val_path, test_path):
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    def uses_column(df):
        df['image_blob'] = df['decoded_image'].apply(lambda x: x['bytes'])
        df['question'] = df['combined_text'].astype(str)
        df['subject'] = df['subject'].astype(str)
        df['label_idx'] = df['label_idx'].astype(int)
        df['image'] = df['image_blob']
        return df[['question', 'subject', 'label_idx', 'image']]

    return uses_column(train_df), uses_column(val_df), uses_column(test_df)

class MathVisionDataset(Dataset):
    def __init__(self, dataframe, processor, transform=None):
        self.df = dataframe
        self.processor = processor
        self.transform = transform
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(io.BytesIO(row['image'])).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        return inputs['pixel_values'].squeeze(0), torch.tensor(row['label_idx'])

def get_transforms():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_augmented_transforms():
    return T.Compose([
        T.Resize((224, 224)),
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.3),
        T.RandomApply([T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.2),
        T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
