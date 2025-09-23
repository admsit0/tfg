"""
Regularizador de proximidad gaussiana:
Penaliza diferencias entre neuronas cercanas en una capa, incentivando que neuronas próximas tengan pesos similares.
La penalización se pondera por una gaussiana de la distancia entre índices de neuronas.
Configurable: capa objetivo, sigma, tipo (pesos o activaciones).
"""
import torch
from .base import Regularizer

class GaussianProximityRegularizer(Regularizer):
    def __init__(self, weight: float = 1e-4, layer_name: str = 'features.0', sigma: float = 1.0, mode: str = 'weights'):
        super().__init__(weight)
        self.layer_name = layer_name
        self.sigma = sigma
        self.mode = mode
        self.target_layer = None
        self.handles = []
        self.activations = []

    def setup(self, model):
        for name, m in model.named_modules():
            if name == self.layer_name:
                self.target_layer = m
                if self.mode == 'activations':
                    self.handles.append(m.register_forward_hook(self._hook))
                break

    def _hook(self, module, inp, out):
        self.activations.append(out.detach())

    def compute(self, model, batch, outputs, loss):
        if self.target_layer is None:
            return torch.tensor(0.0, device=outputs.device)
        if self.mode == 'weights':
            w = self.target_layer.weight if hasattr(self.target_layer, 'weight') else None
            if w is None:
                return torch.tensor(0.0, device=outputs.device)
            vecs = w.view(w.size(0), -1)  # [neurons, ...]
        elif self.mode == 'activations':
            if len(self.activations) == 0:
                return torch.tensor(0.0, device=outputs.device)
            a = self.activations[0]  # [B, neurons, ...]
            vecs = a.mean(dim=0).view(a.size(1), -1)  # mean over batch
            self.activations.clear()
        else:
            return torch.tensor(0.0, device=outputs.device)
        n = vecs.size(0)
        penalty = torch.tensor(0.0, device=vecs.device)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = (i - j) ** 2
                    gauss = torch.exp(-dist / (2 * self.sigma ** 2))
                    penalty += gauss * torch.norm(vecs[i] - vecs[j]) ** 2
        penalty = penalty / (n * (n - 1))  # normalizar
        return self.weight * penalty
