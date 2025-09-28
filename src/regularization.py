"""Regularization helpers and simple grid provider.

This module provides a minimal implementation of get_regularizer_grid so
the cross-validation utilities can expand parameter grids.
"""
from typing import Optional, List


DEFAULT_GRIDS = {
    'dropout': [0.0, 0.2, 0.5],
    'l1': [0.0, 1e-6, 1e-5, 1e-4],
    'l2': [0.0, 1e-6, 1e-5, 1e-4],
}


def get_regularizer_grid(name: str, custom_grid: Optional[List] = None):
    """Return a list of parameter values for the given regularizer.

    If a custom_grid is provided (not None), it will be returned directly.
    """
    if custom_grid is not None:
        return list(custom_grid)
    return DEFAULT_GRIDS.get(name, [])


def describe_regularizer(name: str, kwargs: dict):
    """Return a short textual description of a regularizer instance.

    Not used by the runner but handy for future logging.
    """
    if name == 'dropout':
        return f"dropout(p={kwargs.get('p', 0.0)})"
    if name in ('l1', 'l2'):
        return f"{name}(weight={kwargs.get('weight', 0.0)})"
    return f"{name}({kwargs})"
