import torch
from .base import Regularizer


class L2Regularizer(Regularizer):
    def __init__(self, weight: float = 0.0):
        super().__init__(weight=weight)
        self.weight = float(weight)

    def apply_to_model(self, model: torch.nn.Module):
        return model

    def penalty(self, model: torch.nn.Module):
        if self.weight == 0.0:
            return torch.tensor(0.0, dtype=torch.float32)
        total = torch.tensor(0.0, device=next(model.parameters()).device)
        for p in model.parameters():
            total = total + (p ** 2).sum()
        return self.weight * total
