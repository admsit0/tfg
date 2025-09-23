import torch
from .base import Regularizer

class L2Regularizer(Regularizer):
    def __init__(self, weight: float = 1e-5):
        super().__init__(weight)

    def compute(self, model, batch, outputs, loss):
        reg = torch.tensor(0.0, device=outputs.device)
        for param in model.parameters():
            reg += param.pow(2).sum()
        return self.weight * reg
