
import torch
import numpy as np
from collections import defaultdict

class ActivationCollector:
    def __init__(self, model, module_prefixes=('features',), quantize=None):
        self.quantize = quantize
        self.storage = defaultdict(list)
        self.handles = []
        for name, m in model.named_modules():
            if any(name.startswith(p) for p in module_prefixes):
                self.handles.append(m.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def fn(module, inp, out):
            # flatten per-sample vector
            B = out.shape[0]
            vec = out.view(B, -1).detach().cpu()
            if self.quantize is not None:
                q = (vec / self.quantize).round() * self.quantize
                vec = q
            self.storage[name].append(vec)
        return fn

    def aggregate(self):
        agg = {}
        for name, chunks in self.storage.items():
            X = torch.cat(chunks, dim=0).numpy()
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            # number of unique vectors (approx if quantized)
            if self.quantize is not None:
                uniq = np.unique(X, axis=0).shape[0]
            else:
                uniq = np.unique(np.round(X, 2), axis=0).shape[0]
            agg[name] = {'mean': mean, 'std': std, 'n_unique': int(uniq), 'n_samples': X.shape[0], 'dim': X.shape[1]}
        return agg

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
