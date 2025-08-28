import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms
import numpy as np

def build_cifar10(
    data_dir,
    batch_size=128,
    num_workers=0,
    aug=True,
    subset_ratio=None,
    test_subset_ratio=None,
    pin_memory=True,
    train_split=None,   # NEW: custom train/test split from the original train set
    seed=42             # for reproducibility of splits
):
    """
    Construye los dataloaders de CIFAR-10.

    Args:
        data_dir (str): directorio donde se descargará CIFAR-10
        batch_size (int)
        num_workers (int)
        aug (bool): aplicar aumentos de datos al train set
        subset_ratio (float or None): fracción del train set a usar
        test_subset_ratio (float or None): fracción del test set a usar
        pin_memory (bool): usar pinned memory para transferencias GPU
        train_split (float or None): si no es None, redefine la proporción
                                     de train/val desde el train set oficial
        seed (int): semilla para splits reproducibles

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

    # Dataset oficial
    full_train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    official_test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    # -------- NEW: custom split from train set --------
    if train_split is not None and 0 < train_split < 1.0:
        n_train = int(len(full_train_dataset) * train_split)
        n_val = len(full_train_dataset) - n_train
        torch.manual_seed(seed)  # reproducibility
        train_dataset, val_dataset = random_split(full_train_dataset, [n_train, n_val])
        test_dataset = val_dataset
    else:
        train_dataset = full_train_dataset
        test_dataset = official_test_dataset

    # Subset train_dataset
    if subset_ratio is not None and 0 < subset_ratio < 1.0:
        n = int(len(train_dataset) * subset_ratio)
        indices = np.random.choice(len(train_dataset), n, replace=False)
        train_dataset = Subset(train_dataset, indices)

    # Subset test_dataset
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
