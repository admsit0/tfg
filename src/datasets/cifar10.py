import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms
import numpy as np

def build_cifar10(data_dir, batch_size=128, num_workers=0, aug=True, subset_ratio=None, test_subset_ratio=None, pin_memory=True):
    """
    Construye los dataloaders de CIFAR-10 de manera segura para Windows y GPU.

    Args:
        data_dir (str): directorio donde se descargará CIFAR-10
        batch_size (int)
        num_workers (int)
        aug (bool): si aplicar aumentos de datos
        subset_ratio (float or None): si no es None, usa solo esa fracción del train set (ej. 0.1 = 10%)
        test_subset_ratio (float or None): si no es None, usa solo esa fracción del test set
        pin_memory (bool): usar pinned memory para transferencias a GPU

    Returns:
        train_loader, test_loader
    """
    # Transformaciones
    if aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.ToTensor()

    test_transform = transforms.ToTensor()

    # Datasets
    train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    # Subset para train_dataset
    if subset_ratio is not None and 0 < subset_ratio < 1.0:
        n = int(len(train_dataset) * subset_ratio)
        indices = np.random.choice(len(train_dataset), n, replace=False)
        train_dataset = Subset(train_dataset, indices)

    # Subset para test_dataset
    if test_subset_ratio is not None and 0 < test_subset_ratio < 1.0:
        n = int(len(test_dataset) * test_subset_ratio)
        indices = np.random.choice(len(test_dataset), n, replace=False)
        test_dataset = Subset(test_dataset, indices)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, test_loader
