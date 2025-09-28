import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out, inplace=True)
        return out


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, dropout=0.0):
        super().__init__()
        self.in_planes = 64

        # First conv: configurable for CIFAR-10 or MNIST
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual layers
        self.layer1 = self._make_layer(BasicBlock, 64,  2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
        # Ordered layers list for activation collection
        self.layers = [
            ('conv1', lambda x: F.relu(self.bn1(self.conv1(x)), inplace=True)),
            ('layer1', self.layer1),
            ('layer2', self.layer2),
            ('layer3', self.layer3),
            ('layer4', self.layer4),
            ('avgpool', self.avgpool),
            ('dropout', self.dropout)
        ]

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False, collect_layer=None):
        activations = {}

        for layer_name, layer_fn in self.layers:
            x = layer_fn(x)

            # Collect activations if requested
            if collect_layer and layer_name == collect_layer:
                activations[layer_name] = x.clone()
                return None, activations  # stop early

        # Final classifier
        feats = torch.flatten(x, 1)
        feats = self.dropout(feats)
        logits = self.fc(feats)

        if return_features:
            return logits, feats
        if collect_layer:
            return logits, {**activations, 'fc': logits.clone()}
        return logits
