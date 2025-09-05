import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np

def build_mnist(data_dir, batch_size=128, train_split=0.9, subset_ratio=None, test_subset_ratio=None):
    """
    Builds MNIST dataloaders with optional train/test split and subset sampling.
    Args:
        data_dir (str): Directory to download/store MNIST
        batch_size (int): Batch size
        train_split (float): Fraction of train set for training (rest for validation)
        subset_ratio (float): Fraction of train set to use (optional)
        test_subset_ratio (float): Fraction of test set to use (optional)
    Returns:
        train_loader, test_loader
    """
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train_dataset = MNIST(root=data_dir, train=True, download=True, transform=train_transform)
    official_test_dataset = MNIST(root=data_dir, train=False, download=True, transform=test_transform)
    if train_split < 1.0:
        n_train = int(len(full_train_dataset) * train_split)
        n_val = len(full_train_dataset) - n_train
        train_dataset, val_dataset = random_split(full_train_dataset, [n_train, n_val])
        test_dataset = val_dataset
    else:
        train_dataset = full_train_dataset
        test_dataset = official_test_dataset
    if subset_ratio is not None:
        n = int(len(train_dataset) * subset_ratio)
        indices = np.random.choice(len(train_dataset), n, replace=False)
        train_dataset = Subset(train_dataset, indices)
    if test_subset_ratio is not None:
        n = int(len(test_dataset) * test_subset_ratio)
        indices = np.random.choice(len(test_dataset), n, replace=False)
        test_dataset = Subset(test_dataset, indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader
