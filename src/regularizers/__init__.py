from .base import Regularizer
from .dropout import DropoutRegularizer
from .l1 import L1Regularizer

REGISTRY = {
    'dropout': DropoutRegularizer,
    'l1': L1Regularizer,
}


def build_regularizer(name: str, kwargs: dict):
    if name not in REGISTRY:
        raise ValueError(f'Unknown regularizer: {name}')
    return REGISTRY[name](**(kwargs or {}))


def get_regularizer_grid(name: str, custom_grid=None):
    # expose a simple helper to fetch default grids per-method
    if name == 'dropout':
        return custom_grid if custom_grid is not None else [0.0, 0.2, 0.5]
    if name == 'l1':
        return custom_grid if custom_grid is not None else [0.0, 1e-6, 1e-5, 1e-4]
    if name == 'l2':
        return custom_grid if custom_grid is not None else [0.0, 1e-6, 1e-5, 1e-4]
    return custom_grid or []
