from .gaussian_proximity import GaussianProximityRegularizer
from .base import Regularizer
from .activation_l1 import ActivationL1
from .label_smoothing import LabelSmoothingCrossEntropy

from .l1_reg import L1Regularizer
from .l2_reg import L2Regularizer
from .dropout import DropoutRegularizer
from .early_stopping import EarlyStoppingRegularizer
from .mixup import MixupRegularizer
from .cutout import CutoutRegularizer
from .stochastic_depth import StochasticDepthRegularizer
from .random_erasing import RandomErasingRegularizer

REGISTRY = {
    'activation_l1': ActivationL1,
    'l1': L1Regularizer,
    'l2': L2Regularizer,
    'dropout': DropoutRegularizer,
    'early_stopping': EarlyStoppingRegularizer,
    'mixup': MixupRegularizer,
    'cutout': CutoutRegularizer,
    'stochastic_depth': StochasticDepthRegularizer,
    'random_erasing': RandomErasingRegularizer,
    'gaussian_proximity': GaussianProximityRegularizer,
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
