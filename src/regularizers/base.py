from typing import Dict
import torch


class Regularizer:
    """Base regularizer interface.

    Subclasses should implement:
      - apply_to_model(model): to modify model (e.g., set dropout p)
      - penalty(model): returns a scalar tensor to add to loss
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs or {}

    def apply_to_model(self, model: torch.nn.Module):
        # default: no-op
        return model

    def penalty(self, model: torch.nn.Module) -> torch.Tensor:
        # default: zero penalty
        return torch.tensor(0.0, dtype=torch.float32)
