import torch
from .base import Regularizer


class DropoutRegularizer(Regularizer):
    def __init__(self, p: float = 0.0):
        super().__init__(p=p)
        self.p = float(p)

    def apply_to_model(self, model: torch.nn.Module):
        # try common attribute names
        if hasattr(model, 'dropout'):
            try:
                if isinstance(model.dropout, torch.nn.Dropout):
                    model.dropout.p = self.p
            except Exception:
                model.dropout = torch.nn.Dropout(p=self.p)
        else:
            # attempt to set on layers named 'dropout' or 'dropout1'
            for name, mod in model.named_modules():
                if name.startswith('dropout') and isinstance(mod, torch.nn.Dropout):
                    mod.p = self.p
        return model

    def penalty(self, model: torch.nn.Module):
        # dropout has no explicit penalty term
        return torch.tensor(0.0, dtype=torch.float32)
