import torch
import torch.nn as nn

class SimpleCIFARNet(nn.Module):
    def __init__(self, num_classes=10, dropout=0.0, in_channels=3):
        super().__init__()
        
        # Feature extractor
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(p=dropout)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(p=dropout)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Linear(256, num_classes)

        # Organize layers for iteration + activation collection
        self.layers = [
            self.conv1, self.relu1, self.conv2, self.relu2, self.pool1, self.dropout1,
            self.conv3, self.relu3, self.conv4, self.relu4, self.pool2, self.dropout2,
            self.conv5, self.relu5, self.avgpool
        ]
        self.layer_names = [
            'conv1', 'relu1', 'conv2', 'relu2', 'pool1', 'dropout1',
            'conv3', 'relu3', 'conv4', 'relu4', 'pool2', 'dropout2',
            'conv5', 'relu5', 'avgpool'
        ]

    def forward(self, x, return_features=False, collect_layer=None):
        activations = {}

        for layer, name in zip(self.layers, self.layer_names):
            x = layer(x)

            if collect_layer and name == collect_layer:
                activations[name] = x.clone()
                # Stop early if only that layer is requested
                return None, activations

        # Flatten after avgpool
        feats = x.view(x.size(0), -1)
        logits = self.classifier(feats)

        if return_features:
            return logits, feats
        if collect_layer:
            return logits, {**activations, 'classifier': logits.clone()}
        return logits
