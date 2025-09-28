from .simple_cnn import SimpleCIFARNet
from .resnet18 import ResNet18CIFAR
from .simple_ffw import SimpleFFW

MODEL_REGISTRY = {
    'simple_cnn': SimpleCIFARNet,
    'resnet18': ResNet18CIFAR,
    'simple_ffw': SimpleFFW,
}

def build_model(name: str, dataset='cifar10', regularizers=None, **kwargs):
    """Build model with regularization parameters extracted from regularizers config."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f'Unknown model: {name}')
    
    # Set in_channels based on dataset
    if 'in_channels' not in kwargs:
        if dataset == 'mnist':
            kwargs['in_channels'] = 1
        else:
            kwargs['in_channels'] = 3
    
    # Extract regularization parameters from regularizers config
    if regularizers:
        for reg_cfg in regularizers:
            reg_name = reg_cfg['name']
            reg_kwargs = reg_cfg.get('kwargs', {})
            
            if reg_name == 'dropout':
                kwargs['dropout'] = reg_kwargs.get('p', 0.0)
            elif reg_name == 'l1':
                kwargs['l1_reg'] = reg_kwargs.get('weight', 0.0)
            elif reg_name == 'l2':
                kwargs['l2_reg'] = reg_kwargs.get('weight', 0.0)
    
    # Filter kwargs to only those accepted by the model constructor to keep callers flexible
    import inspect
    model_cls = MODEL_REGISTRY[name]
    try:
        sig = inspect.signature(model_cls.__init__)
        valid_params = [p for p in sig.parameters.keys() if p != 'self' and sig.parameters[p].kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)]
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    except Exception:
        # if reflection fails for any reason, fall back to passing all kwargs
        filtered_kwargs = kwargs

    return model_cls(**filtered_kwargs)

