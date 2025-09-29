import sys
import os
import random
import numpy as np
import torch

# Make the project root importable so tests can import `src`.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.seed import set_seed


def test_seed_reproducibility_across_libraries():
    """Setting the same seed twice must produce identical outputs for
    random, numpy and torch within the same process.
    """
    set_seed(1234)
    r1 = random.random()
    n1 = np.random.rand(3)
    t1 = torch.rand(3)

    # Re-seed and generate again
    set_seed(1234)
    r2 = random.random()
    n2 = np.random.rand(3)
    t2 = torch.rand(3)

    assert r1 == r2
    assert np.allclose(n1, n2)
    assert torch.allclose(t1, t2)


def test_different_seeds_produce_different_values():
    set_seed(111)
    a1 = random.random()
    set_seed(222)
    a2 = random.random()
    assert a1 != a2
