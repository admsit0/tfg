import torch, random, numpy as np, os

def set_seed(seed: int = 42):
    """Set seeds for python, numpy and torch to make runs reproducible.

    This also configures CUDA/cuDNN for deterministic behaviour. Note that
    some nondeterminism can still come from DataLoader workers or external libs.
    """
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    # For CUDA (if available) set the seeds and enforce deterministic algorithms
    try:
        torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass

    # Force deterministic CuDNN where possible
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    # Ensure deterministic algorithms at torch level (may raise on some ops)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # older torch versions or some platforms may not support
        pass

    # Configure CUDA workspace for reproducibility of some ops
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def get_seed():
    """Return an integer seed derived from torch initial seed for use in workers.

    This is useful to create deterministic per-worker seeds.
    """
    s = torch.initial_seed()
    # torch.initial_seed() returns a large number; reduce to 32-bit positive
    return int(s % (2 ** 31 - 1))
