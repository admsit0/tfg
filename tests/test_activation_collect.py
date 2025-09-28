import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.getcwd())
from src.analysis.hidden_state_stats import ActivationCollector

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # named submodule 'my_block'
        self.my_block = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.classifier = nn.Flatten()
    def forward(self, x):
        x = self.my_block(x)
        return self.classifier(x)


def run_test():
    m = TinyNet()
    collector = ActivationCollector(m, layer_names=['my_block'], quantize=None)
    inp = torch.randn(2,3,8,8)
    _ = m(inp)
    acts = collector.get_activations()
    print('Collected layers:', list(acts.keys()))
    assert 'my_block' in acts or any(k.startswith('my_block') for k in acts.keys()), "Exact name 'my_block' not found in collected keys"
    for k,v in acts.items():
        print(k, v.shape, 'dtype=', v.dtype)
        flat = v.reshape(-1)
        # check first up to 10 values are floats
        for val in flat[:min(10, flat.shape[0])]:
            assert isinstance(float(val), float)
    collector.remove_hooks()
    print('OK')

if __name__ == '__main__':
    run_test()
