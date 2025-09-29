import sys
import os
import torch
from torch.utils.data import Dataset

# Make the project root importable so tests can import `src`.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.cross_validation import CrossValidator
from src.utils.seed import set_seed


class DummyDataset(Dataset):
    def __init__(self, n):
        self.data = list(range(n))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def indices_from_folds(folds):
    all_train = set()
    all_val = set()
    for tr, va in folds:
        all_train.update(tr)
        all_val.update(va)
    return all_train, all_val


def test_kfold_covers_all_indices_and_non_overlapping():
    ds = DummyDataset(20)
    set_seed(777)
    cv = CrossValidator(n_splits=4, seed=777, shuffle=True)
    folds = cv.split_dataset(ds)

    # Check number of folds
    assert len(folds) == 4

    all_train, all_val = indices_from_folds(folds)

    # Every index should appear either in some train or val set across folds
    combined = all_train.union(all_val)
    assert combined == set(range(len(ds)))

    # Within a single fold, train and val should be disjoint
    for tr, va in folds:
        assert set(tr).isdisjoint(set(va))


def test_create_fold_loaders_and_determinism():
    ds = DummyDataset(12)
    set_seed(42)
    cv = CrossValidator(n_splits=3, seed=42, shuffle=True)
    folds_a = cv.split_dataset(ds)

    # Recreate with same seed and check folds are identical
    set_seed(42)
    cv2 = CrossValidator(n_splits=3, seed=42, shuffle=True)
    folds_b = cv2.split_dataset(ds)
    assert folds_a == folds_b

    # Build a loader for first fold and check batches
    train_loader, val_loader = cv.create_fold_loaders(ds, folds_a[0], batch_size=4)
    # Collect all items from loaders
    train_items = [x for batch in train_loader for x in batch]
    val_items = [x for batch in val_loader for x in batch]

    # Check no overlap
    assert set(train_items).isdisjoint(set(val_items))
