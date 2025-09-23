import torch
from .base import Regularizer

class DropoutRegularizer(Regularizer):
    def __init__(self, weight: float = 0.5, modules=('features',)):
        super().__init__(weight)
        self.modules = modules
        self.handles = []

    def setup(self, model):
        for name, m in model.named_modules():
            if any(name.startswith(prefix) for prefix in self.modules):
                self.handles.append(m.register_forward_hook(self._hook))

    def _hook(self, module, inp, out):
        if self.weight > 0:
            out = torch.nn.functional.dropout(out, p=self.weight, training=True)

    def compute(self, model, batch, outputs, loss):
        return torch.tensor(0.0, device=outputs.device)
