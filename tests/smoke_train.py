import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.getcwd())
from src.train import train_one_epoch

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3,4,3,padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(4, 10)
    def forward(self,x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# synthetic data
x = torch.randn(16,3,8,8)
# make labels in range [0,9]
y = torch.randint(0,10,(16,))
loader = DataLoader(TensorDataset(x,y), batch_size=4)
model = TinyModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()
regs = []

train_loss, train_acc, t = train_one_epoch(model, loader, optimizer, loss_fn, regs, device=torch.device('cpu'))
print('train_loss', train_loss, 'train_acc', train_acc)
