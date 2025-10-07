import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np

def build_fashion_mnist(
    data_dir,
    batch_size=128,
    train_split=0.9,
    subset_ratio=None,
    test_subset_ratio=None,
    seed: int = 42
):
    """
    Builds Fashion-MNIST dataloaders with optional train/test split and subset sampling.
    Returns train_loader, test_loader
    """
    mean, std = (0.2860,), (0.3530,)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    full_train_dataset = FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    official_test_dataset = FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    if train_split < 1.0:
        # use torch.Generator for deterministic random_split
        g = torch.Generator()
        try:
            g.manual_seed(int(seed))
            n_train = int(len(full_train_dataset) * train_split)
            n_val = len(full_train_dataset) - n_train
            train_dataset, val_dataset = random_split(full_train_dataset, [n_train, n_val], generator=g)
            test_dataset = val_dataset
        except Exception:
            # fallback to non-deterministic split if generator unsupported
            n_train = int(len(full_train_dataset) * train_split)
            n_val = len(full_train_dataset) - n_train
            train_dataset, val_dataset = random_split(full_train_dataset, [n_train, n_val])
            test_dataset = val_dataset
    else:
        train_dataset = full_train_dataset
        test_dataset = official_test_dataset

    if subset_ratio is not None:
        n = int(len(train_dataset) * subset_ratio)
        rng = np.random.RandomState(int(seed))
        indices = rng.choice(len(train_dataset), n, replace=False)
        train_dataset = Subset(train_dataset, indices)

    if test_subset_ratio is not None:
        n = int(len(test_dataset) * test_subset_ratio)
        rng = np.random.RandomState(int(seed) + 1)
        indices = rng.choice(len(test_dataset), n, replace=False)
        test_dataset = Subset(test_dataset, indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader
