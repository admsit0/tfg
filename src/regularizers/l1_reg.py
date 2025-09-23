import torch
from .base import Regularizer

class L1Regularizer(Regularizer):
    def __init__(self, weight: float = 1e-5):
        super().__init__(weight)

    """
    Regularizador L1 sobre pesos:
    Penaliza la magnitud absoluta de los parámetros (weights) del modelo.
    Induce sparsity directamente en los pesos de la red, lo que puede llevar a modelos más simples y menos propensos al overfitting.
    Ayuda a reducir el número de parámetros efectivos, favoreciendo la simplicidad estructural del modelo.
    """
    def compute(self, model, batch, outputs, loss):
        reg = torch.tensor(0.0, device=outputs.device)
        for param in model.parameters():
            reg += param.abs().sum()
        return self.weight * reg
