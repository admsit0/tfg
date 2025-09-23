import torch
from .base import Regularizer

class EarlyStoppingRegularizer(Regularizer):
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        super().__init__(weight=0.0)
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def compute(self, model, batch, outputs, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return torch.tensor(0.0, device=outputs.device)
