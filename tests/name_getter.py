import sys, os
sys.path.insert(0, os.getcwd())
from src.models import build_model

m = build_model('simple_cnn', dataset='cifar10', num_classes=10)
names = [n for n,_ in m.named_modules()]
for n in names:
    print(n)

