import torch
from .base import Regularizer

class CutoutRegularizer(Regularizer):
    def __init__(self, n_holes: int = 1, length: int = 16):
        super().__init__(weight=0.0)
        self.n_holes = n_holes
        self.length = length

    def compute(self, model, batch, outputs, loss):
        # Cutout is usually applied in the dataloader, but aquí es placeholder
        return torch.tensor(0.0, device=outputs.device)
