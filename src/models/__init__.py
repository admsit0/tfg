from .simple_cnn import SimpleCIFARNet
from .resnet18 import ResNet18CIFAR

MODEL_REGISTRY = {
    'simple_cnn': SimpleCIFARNet,
    'resnet18': ResNet18CIFAR,
}

def build_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f'Unknown model: {name}')
    return MODEL_REGISTRY[name](**kwargs)
