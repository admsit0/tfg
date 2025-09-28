import torch
import torch.nn as nn

class SimpleFFW(nn.Module):
    """A simple feed-forward network with one hidden layer."""
    def __init__(self, input_dim=28*28, hidden_dim=256, num_classes=10, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, return_features=False, collect_layer=None):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        hidden = self.fc1(x)          # raw hidden pre-activation
        hidden_act = self.relu(hidden)
        hidden_act = self.dropout(hidden_act)

        out = self.fc2(hidden_act)

        # If the caller requested activations for a specific layer, return them
        # in the same format other models use: (logits, {layer_name: tensor})
        if collect_layer:
            # accept 'hidden' or the name 'fc1' as aliases
            aliases = [collect_layer]
            if collect_layer == 'hidden':
                aliases.append('fc1')
            activations = {}
            # provide the hidden activations under the canonical key 'hidden'
            if collect_layer in ('hidden', 'fc1') or any(a in ('hidden','fc1') for a in aliases):
                activations['hidden'] = hidden_act
            return out, activations

        if return_features:
            return out, hidden_act
        return out

    def get_hidden(self, x):
        """Always return the hidden layer activations only."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        hidden = self.fc1(x)
        hidden_act = self.dropout(self.relu(hidden))
        return hidden_act
