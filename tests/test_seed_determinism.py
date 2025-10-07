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


def test_model_initialization_and_one_step_evolves_identically():
    """Initialize two models with the same seed and verify parameters and one training step are identical."""
    from src.models import build_model
    set_seed(999)
    m1 = build_model('simple_ffw', dataset='fashion_mnist', input_dim=784, hidden_dim=16, num_classes=10)
    set_seed(999)
    m2 = build_model('simple_ffw', dataset='fashion_mnist', input_dim=784, hidden_dim=16, num_classes=10)

    # compare parameters initially
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1.data, p2.data)

    # prepare a tiny batch
    xb = torch.randn(4, 1, 28, 28)
    yb = torch.tensor([0,1,2,3], dtype=torch.long)

    opt1 = torch.optim.SGD(m1.parameters(), lr=0.01)
    opt2 = torch.optim.SGD(m2.parameters(), lr=0.01)

    # forward, backward, step
    lfn = torch.nn.CrossEntropyLoss()

    m1.train(); m2.train()
    o1 = m1(xb)
    if isinstance(o1, tuple): o1 = o1[0]
    loss1 = lfn(o1, yb)
    opt1.zero_grad(); loss1.backward(); opt1.step()

    o2 = m2(xb)
    if isinstance(o2, tuple): o2 = o2[0]
    loss2 = lfn(o2, yb)
    opt2.zero_grad(); loss2.backward(); opt2.step()

    # after one step params should still match
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1.data, p2.data, atol=1e-6)
