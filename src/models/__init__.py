from .simple_cnn import SimpleCIFARNet
from .resnet18 import ResNet18CIFAR

MODEL_REGISTRY = {
    'simple_cnn': SimpleCIFARNet,
    'resnet18': ResNet18CIFAR,
}

def build_model(name: str, dataset='cifar10', **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f'Unknown model: {name}')
    # Set in_channels based on dataset
    if 'in_channels' not in kwargs:
        if dataset == 'mnist':
            kwargs['in_channels'] = 1
        else:
            kwargs['in_channels'] = 3
    return MODEL_REGISTRY[name](**kwargs)
