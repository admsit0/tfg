import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import transforms
import torchvision.datasets as datasets 
from tqdm import tqdm 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# CONFIG
# =========================================================

# SET TO FALSE: BatchNorm se probará solo como método explícito en la rejilla
USE_BATCHNORM = False 

CONFIG = {
    'EPOCHS': 60,
    'BATCH_SIZE': 64,
    'SEEDS': [42],
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'OUTPUT_DIR': os.path.join(SCRIPT_DIR, 'outputs_moreMethods'),
    'DATA_LIMIT': 25000
}

def setup_directories():
    subdirs = ['models', 'data'] 
    for sd in subdirs:
        os.makedirs(os.path.join(CONFIG['OUTPUT_DIR'], sd), exist_ok=True)
    print(f"✅ Directorios creados en {CONFIG['OUTPUT_DIR']}")

# =========================================================
# DATA
# =========================================================

def get_cifar_data():
    print("⬇️ Cargando datos CIFAR-10...")
    train_set = datasets.CIFAR10(root='./data_raw', train=True, download=True)
    test_set = datasets.CIFAR10(root='./data_raw', train=False, download=True)
    
    x_train = train_set.data.astype('float32') / 255.0
    y_train = np.array(train_set.targets)
    x_test = test_set.data.astype('float32') / 255.0
    y_test = np.array(test_set.targets)
    
    if CONFIG['DATA_LIMIT']:
        x_train = x_train[:CONFIG['DATA_LIMIT']]
        y_train = y_train[:CONFIG['DATA_LIMIT']]
    
    tensor_x_train = torch.tensor(x_train).permute(0, 3, 1, 2)
    tensor_y_train = torch.tensor(y_train, dtype=torch.long)
    tensor_x_test = torch.tensor(x_test).permute(0, 3, 1, 2)
    tensor_y_test = torch.tensor(y_test, dtype=torch.long)
    
    return (tensor_x_train, tensor_y_train), (tensor_x_test, tensor_y_test)

class AugmentedDataset(Dataset):
    def __init__(self, x_tensor, y_tensor, transform=None):
        self.x = x_tensor
        self.y = y_tensor
        self.transform = transform
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        img = self.x[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.y[idx]

def get_augmentation_levels():
    t0 = None
    t1 = transforms.RandomHorizontalFlip(p=0.5)
    t2 = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4)
    ])
    t3 = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(15)
    ])
    return {0: t0, 1: t1, 2: t2, 3: t3}

# =========================================================
# MÓDULOS DE REGULARIZACIÓN CUSTOM
# =========================================================

class GaussianNoise(nn.Module):
    """Inyecta ruido gaussiano durante el entrenamiento (Data Augmentation de facto)"""
    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training and self.std > 0:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x

# =========================================================
# MODELO
# =========================================================

class ConvNet(nn.Module):
    def __init__(self, reg_method='', reg_val=0.0):
        super(ConvNet, self).__init__()
        
        dropout_p = reg_val if reg_method == 'Dropout' else 0.0
        noise_std = reg_val if reg_method == 'GaussianNoise' else 0.0
        self.use_bn = (reg_method == 'BatchNorm') or USE_BATCHNORM
        
        # Capa de Ruido Entrada
        self.noise = GaussianNoise(noise_std)
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.drop2d = nn.Dropout2d(p=dropout_p/2 if dropout_p > 0 else 0)
        self.drop1d = nn.Dropout(p=dropout_p)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.noise(x) # Ruido en la entrada
        
        if self.use_bn:
            x = self.pool(self.drop2d(F.relu(self.bn1(self.conv1(x)))))
            x = self.pool(self.drop2d(F.relu(self.bn2(self.conv2(x)))))
            x = self.pool(self.drop2d(F.relu(self.bn3(self.conv3(x)))))
        else:
            x = self.pool(self.drop2d(F.relu(self.conv1(x))))
            x = self.pool(self.drop2d(F.relu(self.conv2(x))))
            x = self.pool(self.drop2d(F.relu(self.conv3(x))))
        
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        if self.drop1d.p > 0: 
            x = self.drop1d(x)
        x = self.fc2(x)
        return x

# =========================================================
# TRAIN
# =========================================================

def evaluate(net, loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(CONFIG['DEVICE']), y.to(CONFIG['DEVICE'])
            out = net(x)
            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return correct / total

def train_experiment(net_name, reg_method, reg_val, train_loader, test_loader):
    
    net = ConvNet(reg_method=reg_method, reg_val=reg_val).to(CONFIG['DEVICE'])
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    
    # Early Stopping Config
    patience = int(reg_val) if reg_method == 'EarlyStopping' else 999
    best_val_acc = -1.0
    epochs_no_improve = 0
    
    print(f"\n🚀 Run: {reg_method} = {reg_val}")

    for epoch in tqdm(range(1, CONFIG['EPOCHS'] + 1), leave=False):
        net.train()
        
        for x, y in train_loader:
            x, y = x.to(CONFIG['DEVICE']), y.to(CONFIG['DEVICE'])
            optimizer.zero_grad()
            out = net(x)
            loss = criterion(out, y)
            
            # --- CORRECCIÓN CRÍTICA L1 / L2 ---
            if reg_method in ['L1', 'L2']:
                reg_loss = 0
                for name, param in net.named_parameters():
                    # Solo penalizar pesos (weight) de convoluciones y densas. Ignorar bias y BatchNorm
                    if 'weight' in name and not 'bn' in name:
                        if reg_method == 'L1':
                            reg_loss += torch.norm(param, 1)
                        else:
                            reg_loss += torch.norm(param, 2)**2 # ELEVADO AL CUADRADO
                loss += reg_val * reg_loss
            # ----------------------------------
            
            loss.backward()
            optimizer.step()
        
        train_acc = evaluate(net, train_loader) 
        val_acc = evaluate(net, test_loader)
        
        history.append({
            'epoch': epoch,
            'reg_method': reg_method,
            'reg_val': reg_val,
            'train_acc': train_acc,
            'val_acc': val_acc
        })

        # -------- GUARDADO DE MODELO (como en scripts anteriores) --------
        model_name = f"{net_name}_{reg_method}_{reg_val:.5f}_ep{epoch}.pth"
        save_path = os.path.join(CONFIG['OUTPUT_DIR'], 'models', model_name)
        torch.save(net.state_dict(), save_path)
# ------------------------------------------------------------------
        
        # Early Stopping Check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"\n🛑 Early Stopping activado en época {epoch} (No mejora en {patience} épocas).")
            break
            
    return pd.DataFrame(history)

def save_csv_append(df, filepath):
    if os.path.exists(filepath):
        df.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df.to_csv(filepath, index=False)

# =========================================================
# MACRO EXPERIMENTO
# =========================================================

def run_macro_experiment():
    setup_directories()
    
    (tx_train, ty_train), (tx_test, ty_test) = get_cifar_data()
    
    NET_TYPES = ['CNN']
    
    # EXPERIMENTS = {
    #     'L1': [0.0, 0.0001, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
    #     'L2': [0.0, 0.0001, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
    #     'Dropout': [0.0, 0.1, 0.15, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6],
    #     'DataAug': [0, 1, 2, 3],
    #     'EarlyStopping': [3, 5, 8, 12, 15],
    #     'GaussianNoise': [0.01, 0.05, 0.1, 0.2],
    #     'BatchNorm': [1]
    # }

    EXPERIMENTS = {
        'L2': [0.0, 0.0001, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
        'Dropout': [0.0, 0.1, 0.15, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6],
        'DataAug': [0, 1, 2, 3],
        'EarlyStopping': [3, 5, 8, 12, 15],
        'GaussianNoise': [0.01, 0.05, 0.1, 0.2],
        'BatchNorm': [1]
    }

    aug_transforms = get_augmentation_levels()
    
    for net_name in NET_TYPES:
        for reg_method, param_values in EXPERIMENTS.items():

            csv_name = f"data_{net_name}_{reg_method}.csv"
            csv_path = os.path.join(CONFIG['OUTPUT_DIR'], 'data', csv_name)
            
            for val in param_values:
                if reg_method == 'DataAug':
                    current_transform = aug_transforms[val]
                    train_ds = AugmentedDataset(tx_train, ty_train, transform=current_transform)
                else:
                    train_ds = AugmentedDataset(tx_train, ty_train, transform=None)
                
                train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
                test_loader = DataLoader(TensorDataset(tx_test, ty_test), batch_size=1000)
                
                run_df = train_experiment(net_name, reg_method, val, train_loader, test_loader)
                save_csv_append(run_df, csv_path)
            

if __name__ == "__main__":
    run_macro_experiment()
