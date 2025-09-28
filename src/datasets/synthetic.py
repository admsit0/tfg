import torch
from torch.utils.data import DataLoader, TensorDataset

def build_synthetic(data_dir, batch_size=32, train_split=0.9, subset_ratio=None, test_subset_ratio=None):
    # tiny synthetic dataset: 1000 train, 200 test, 28x28 flattened images
    N_train = 200
    N_test = 50
    x_train = torch.randn(N_train, 1, 28, 28)
    y_train = torch.randint(0, 10, (N_train,))
    x_test = torch.randn(N_test, 1, 28, 28)
    y_test = torch.randint(0, 10, (N_test,))
    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
