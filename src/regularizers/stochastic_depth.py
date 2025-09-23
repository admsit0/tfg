import torch
from .base import Regularizer

class StochasticDepthRegularizer(Regularizer):
    def __init__(self, drop_prob: float = 0.2, modules=('features',)):
        super().__init__(weight=0.0)
        self.drop_prob = drop_prob
        self.modules = modules
        self.handles = []

    def setup(self, model):
        for name, m in model.named_modules():
            if any(name.startswith(prefix) for prefix in self.modules):
                self.handles.append(m.register_forward_hook(self._hook))

    def _hook(self, module, inp, out):
        if torch.rand(1).item() < self.drop_prob:
            return torch.zeros_like(out)
        return out

    def compute(self, model, batch, outputs, loss):
        return torch.tensor(0.0, device=outputs.device)
