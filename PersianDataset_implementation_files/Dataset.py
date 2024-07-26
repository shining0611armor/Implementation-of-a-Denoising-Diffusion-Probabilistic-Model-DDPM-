# Dataset.py

import random
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Lambda
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Resize, ToTensor, Lambda
from torchvision.datasets import ImageFolder
import pandas as pd
import pandas as pd
import os
import gdown
from zipfile import ZipFile





file_id = '1bK1gYdxK92jXtDAcuUvITx6pa4CnyPAw'
output_path = '/content/dataset.zip'
extracted_path = '/content/dataset/extracted'
gdown.download(f'https://drive.google.com/uc?export=download&id={file_id}', output_path, quiet=False)

with ZipFile(output_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_path)

# we remove the zip file after extraction
os.remove(output_path)



class PersianDigitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".png"):
                    self.image_paths.append(os.path.join(root, file))
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # we convert image to grayscale
        if self.transform:
            image = self.transform(image)
        # we extract label from file path
        label = int(os.path.basename(img_path).split('.')[0])

        return image, label

def get_data_loaders(batch_size):

    transform = Compose([
        Resize((64, 64)),  # we resize the image to 64x64
        ToTensor(),
        Lambda(lambda x: (x - 0.5) * 2)  # Normalize to [-1, 1]
    ])

    dataset = PersianDigitDataset(root_dir='/content/dataset/extracted', transform=transform)
    train_ratio = 0.95
    test_ratio = 0.05
    train_length = int(train_ratio * len(dataset))
    test_length = len(dataset) - train_length

    train_dataset, test_dataset = random_split(dataset, [train_length, test_length])
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")
    for images, labels in test_loader:
        print(images.shape, labels.shape)
        break

    for images, labels in test_loader:
        print(images.shape, labels.shape)
        break

    return train_loader, test_loader




