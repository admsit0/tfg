import torch
from .base import Regularizer

class RandomErasingRegularizer(Regularizer):
    def __init__(self, probability: float = 0.5, sl: float = 0.02, sh: float = 0.4, r1: float = 0.3):
        super().__init__(weight=0.0)
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def compute(self, model, batch, outputs, loss):
        # Random Erasing is usually applied in the dataloader, aquí es placeholder
        return torch.tensor(0.0, device=outputs.device)
