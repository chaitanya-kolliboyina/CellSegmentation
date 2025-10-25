from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from config import DataPaths

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = np.array(Image.open(self.mask_paths[idx]))
        if self.transform:
            transformed = self.transform(img, mask)
            img = transformed[0]
            mask = torch.from_numpy(transformed[1]).long()
        return img, mask
    

def prepare_data(config):
    paths = DataPaths(images_root=os.path.join(config.data_path, "images"), masks_root=os.path.join(config.data_path, "masks"))
    image_files = sorted([f for f in os.listdir(paths.images_root) if f.endswith(('.jpg', '.png'))])
    mask_files = sorted([f for f in os.listdir(paths.masks_root) if f.endswith(('.jpg', '.png'))])
    
    image_paths = [os.path.join(paths.images_root, f) for f in image_files]
    mask_paths = [os.path.join(paths.masks_root, f) for f in mask_files]
    
    train_img, val_img, train_mask, val_mask = train_test_split(image_paths, mask_paths, test_size=0.2)
    
    return train_img, val_img, train_mask, val_mask 