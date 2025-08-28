from .simple_cnn import SimpleCIFARNet

MODEL_REGISTRY = {
    'simple_cnn': SimpleCIFARNet,
}

def build_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f'Unknown model: {name}')
    return MODEL_REGISTRY[name](**kwargs)
