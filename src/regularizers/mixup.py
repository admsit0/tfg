import torch
from .base import Regularizer

class MixupRegularizer(Regularizer):
    def __init__(self, alpha: float = 1.0):
        super().__init__(weight=0.0)
        self.alpha = alpha

    def compute(self, model, batch, outputs, loss):
        # Mixup is usually applied in the dataloader, but here we provide a placeholder
        return torch.tensor(0.0, device=outputs.device)
