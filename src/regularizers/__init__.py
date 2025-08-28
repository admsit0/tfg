from .base import Regularizer
from .activation_l1 import ActivationL1
from .label_smoothing import LabelSmoothingCrossEntropy

REGISTRY = {
    'activation_l1': ActivationL1,
}

LOSS_REGISTRY = {
    'cross_entropy': None,  # uses torch.nn.CrossEntropyLoss
    'label_smoothing': LabelSmoothingCrossEntropy,
}

def build_regularizers(reg_cfgs):
    regs = []
    for cfg in reg_cfgs or []:
        name = cfg['name']
        kwargs = cfg.get('kwargs', {})
        if name not in REGISTRY:
            raise ValueError(f'Unknown regularizer: {name}')
        regs.append(REGISTRY[name](**kwargs))
    return regs

def build_loss(loss_name: str, **kwargs):
    if loss_name == 'cross_entropy' or loss_name is None:
        import torch
        return torch.nn.CrossEntropyLoss()
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(f'Unknown loss: {loss_name}')
    return LOSS_REGISTRY[loss_name](**kwargs)
