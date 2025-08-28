import torch

class Regularizer:
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    def setup(self, model):
        pass
    def compute(self, model, batch, outputs, loss):
        return torch.tensor(0.0, device=outputs.device if torch.is_tensor(outputs) else 'cpu')
