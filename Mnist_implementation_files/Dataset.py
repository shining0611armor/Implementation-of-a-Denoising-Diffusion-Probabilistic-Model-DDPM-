# Dataset.py

import random
import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader

# Set the seed for reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def get_data_loaders(batch_size):
  
    # Normalizing between [-1, 1]
    transform = Compose([
        ToTensor(),
        Lambda(lambda x: (x - 0.5) * 2)]
    )

    train_dataset = MNIST(root='.', train=True, transform=transform, download=True)
    test_dataset = MNIST(root='.', train=False, transform=transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
