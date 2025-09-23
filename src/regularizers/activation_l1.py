"""
Regularizador L1 sobre activaciones:
Penaliza la magnitud absoluta de las activaciones intermedias de la red (por ejemplo, salidas de capas ocultas).
Induce sparsity en las representaciones internas, lo que ayuda a controlar la complejidad de las representaciones aprendidas.
Es útil para interpretabilidad y robustez, ya que permite analizar cómo y dónde la red activa sus neuronas.
"""
import torch
import torch.nn as nn
from .base import Regularizer

class ActivationL1(Regularizer):
    def __init__(self, weight: float = 1e-4, modules=('features',)):
        super().__init__(weight)
        self.handles = []
        self.activations = []
        self.modules = modules

    def _hook(self, module, inp, out):
        # out: [B, C, H, W] or [B, D]
        self.activations.append(out)

    def setup(self, model):
        # register forward hooks on modules matching names
        for name, m in model.named_modules():
            if any(name.startswith(prefix) for prefix in self.modules):
                self.handles.append(m.register_forward_hook(self._hook))

    def compute(self, model, batch, outputs, loss):
        if len(self.activations) == 0:
            return torch.tensor(0.0, device=outputs.device)
        reg = torch.tensor(0.0, device=outputs.device)
        for a in self.activations:
            reg = reg + a.abs().mean()
        self.activations.clear()
        return self.weight * reg
