import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCIFARNet(nn.Module):
    def __init__(self, num_classes=10, dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(p=dropout),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(p=dropout),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x, return_features=False):
        feats = self.features(x)
        feats = feats.view(feats.size(0), -1)
        logits = self.classifier(feats)
        if return_features:
            return logits, feats
        return logits
