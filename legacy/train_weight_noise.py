import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.datasets as datasets
from tqdm import tqdm
import argparse
import copy

# =========================================================
# CONFIGURACIÓN ESTRICTA DEL EXPERIMENTO
# =========================================================
TARGET_EPOCH = 60
TRAIN_WEIGHT_NOISE_STD = 0.02 # Intensidad del ruido al ENTRENAR
USE_BATCHNORM = True

# Niveles de ruido para la EVALUACIÓN (Flat minima test)
NOISE_LEVELS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15]
N_RUNS = 3

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--"
})

class ConvNet(nn.Module):
    def __init__(self, num_classes=10, use_bn=False):
        super(ConvNet, self).__init__()
        self.use_bn = use_bn
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        if self.use_bn:
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
        else:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_loaders(dataset_name, batch_size=200):
    print(f"Cargando {dataset_name} (Train y Val)...")
    if dataset_name == 'CIFAR10':
        train_set = datasets.CIFAR10(root='./data_raw', train=True, download=True)
        test_set = datasets.CIFAR10(root='./data_raw', train=False, download=True)
        x_train = torch.tensor(train_set.data).permute(0, 3, 1, 2).float() / 255.0
        y_train = torch.tensor(train_set.targets, dtype=torch.long)
        x_test = torch.tensor(test_set.data).permute(0, 3, 1, 2).float() / 255.0
        y_test = torch.tensor(test_set.targets, dtype=torch.long)
        num_classes = 10
    elif dataset_name == 'SVHN':
        train_set = datasets.SVHN(root='./data_raw', split='train', download=True)
        test_set = datasets.SVHN(root='./data_raw', split='test', download=True)
        x_train = torch.tensor(train_set.data).float() / 255.0
        y_train = torch.tensor(train_set.labels, dtype=torch.long)
        x_test = torch.tensor(test_set.data).float() / 255.0
        y_test = torch.tensor(test_set.labels, dtype=torch.long)
        num_classes = 10
    elif dataset_name == 'FashionMNIST':
        train_set = datasets.FashionMNIST(root='./data_raw', train=True, download=True)
        test_set = datasets.FashionMNIST(root='./data_raw', train=False, download=True)
        x_train = F.pad(train_set.data.float() / 255.0, (2, 2, 2, 2)).unsqueeze(1).repeat(1, 3, 1, 1)
        y_train = torch.tensor(train_set.targets, dtype=torch.long)
        x_test = F.pad(test_set.data.float() / 255.0, (2, 2, 2, 2)).unsqueeze(1).repeat(1, 3, 1, 1)
        y_test = torch.tensor(test_set.targets, dtype=torch.long)
        num_classes = 10
    
    x_train, y_train = x_train[:25000], y_train[:25000]
    
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=500, shuffle=False)
    return train_loader, test_loader, num_classes

def train_weight_noise_model(train_loader, val_loader, device, num_classes, save_path):
    print(f"\n⚙️ Entrenando modelo con RUIDO EN LOS PESOS (std={TRAIN_WEIGHT_NOISE_STD}) por {TARGET_EPOCH} épocas...")
    net = ConvNet(num_classes=num_classes, use_bn=USE_BATCHNORM).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(TARGET_EPOCH):
        net.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            clean_state = {n: p.data.clone() for n, p in net.named_parameters() if 'weight' in n and 'bn' not in n}
            
            for n, p in net.named_parameters():
                if 'weight' in n and 'bn' not in n:
                    p.data.add_(torch.randn_like(p) * TRAIN_WEIGHT_NOISE_STD)
                    
            optimizer.zero_grad()
            out = net(x)
            loss = criterion(out, y)
            loss.backward() # Los gradientes se calculan sobre la red ruidosa
            
            for n, p in net.named_parameters():
                if 'weight' in n and 'bn' not in n:
                    p.data.copy_(clean_state[n])
                    
            optimizer.step()
            
        print(f"Época {epoch+1}/{TARGET_EPOCH} completada.")
        
    torch.save(net.state_dict(), save_path)
    print(f"✅ Modelo WeightNoise guardado en {save_path}")
    return net

def get_baseline_info(data_dir):
    for f in os.listdir(data_dir):
        if f.startswith("data_CNN_") and f.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_dir, f))
            if 0.0 in df['reg_val'].values:
                return f.replace("data_CNN_", "").replace(".csv", "")
    return None

def evaluate_with_noise(net, loader, device, original_state_dict, sigma, target_layer):
    net.load_state_dict(original_state_dict)
    if sigma > 0.0:
        with torch.no_grad():
            for name, param in net.named_parameters():
                if target_layer in name and 'weight' in name and 'bn' not in name:
                    noise = torch.randn_like(param) * sigma
                    param.add_(noise)
    
    net.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = net(x)
            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            
    return (correct / total) * 100.0

def plot_flat_minima(pdf, df, dataset_name, layer_name):
    methods = df['method'].unique()
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    fig.suptitle(f"Análisis Flat Minima: Baseline vs WeightNoise\nCapa Evaluada: {layer_name.upper()} | Dataset: {dataset_name}", fontsize=16, fontweight='bold')
    
    for m in methods:
        sub = df[df['method'] == m].sort_values('sigma')
        linewidth = 3.5 if m == 'Baseline' else 2.5
        linestyle = '--' if m == 'Baseline' else '-'
        ax.plot(sub['sigma'], sub['acc_mean'], marker='o', linewidth=linewidth, linestyle=linestyle, label=m)
        ax.fill_between(sub['sigma'], sub['acc_mean'] - sub['acc_std'], sub['acc_mean'] + sub['acc_std'], alpha=0.15)
            
    ax.set_title("Inyección de Ruido en Tiempo de Evaluación", fontsize=14)
    ax.set_xlabel("Magnitud del Ruido en Pesos (Desviación Estándar $\\sigma$)", fontsize=12)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_xticks(NOISE_LEVELS)
    ax.legend(fontsize=11)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

def main(dataset_name):
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    input_base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'analysis', f'outputs_{dataset_name}')
    models_dir = os.path.join(input_base_dir, 'models')
    data_dir = os.path.join(input_base_dir, 'data')
    
    output_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weight_noise_analysis', f'{dataset_name}')
    os.makedirs(output_base_dir, exist_ok=True)
    
    pdf_output_path = os.path.join(output_base_dir, f'Train_WeightNoise_Report_{dataset_name}.pdf')
    wn_model_path = os.path.join(output_base_dir, f'CNN_WeightNoise_{TRAIN_WEIGHT_NOISE_STD:.5f}_ep{TARGET_EPOCH}.pth')
    
    print(f"\nINICIANDO ENTRENAMIENTO Y TEST WEIGHT NOISE (Dataset: {dataset_name})")
    train_loader, val_loader, num_classes = get_loaders(dataset_name)
    
    # 1. ENTRENAR EL MODELO O CARGARLO SI YA EXISTE
    if not os.path.exists(wn_model_path):
        train_weight_noise_model(train_loader, val_loader, device, num_classes, wn_model_path)
    
    # Cargar Baseline
    baseline_src = get_baseline_info(data_dir)
    if not baseline_src:
        print("❌ Falla crítica: No se encontró el Baseline.")
        return
        
    baseline_path = os.path.join(models_dir, f"CNN_{baseline_src}_0.00000_ep{TARGET_EPOCH}.pth")
    
    # Diccionario de modelos a evaluar
    models_to_test = {
        'Baseline': baseline_path,
        f'WeightNoise (std={TRAIN_WEIGHT_NOISE_STD})': wn_model_path
    }
    
    # 2. EVALUACIÓN FLAT MINIMA POR CAPA
    with PdfPages(pdf_output_path) as pdf:
        for layer in ['conv1', 'conv3', 'fc1']:
            print(f"\n▶ EVALUANDO FLAT MINIMA EN CAPA: {layer.upper()}")
            results = []
            
            for m_name, path in models_to_test.items():
                state_dict = torch.load(path, map_location=device)
                net = ConvNet(num_classes=num_classes, use_bn=USE_BATCHNORM).to(device)
                net.load_state_dict(state_dict)
                clean_state_dict = copy.deepcopy(net.state_dict())
                
                for sigma in NOISE_LEVELS:
                    if sigma == 0.0:
                        acc = evaluate_with_noise(net, val_loader, device, clean_state_dict, 0.0, layer)
                        mean_acc, std_acc = acc, 0.0
                    else:
                        runs_acc = []
                        for _ in range(N_RUNS):
                            acc = evaluate_with_noise(net, val_loader, device, clean_state_dict, sigma, layer)
                            runs_acc.append(acc)
                        mean_acc = np.mean(runs_acc)
                        std_acc = np.std(runs_acc)
                    
                    results.append({
                        'method': m_name, 'sigma': sigma, 'acc_mean': mean_acc, 'acc_std': std_acc
                    })
                    
            df_results = pd.DataFrame(results)
            plot_flat_minima(pdf, df_results, dataset_name, layer)

    print(f"\n🚀 ¡PDF de 3 páginas completado y guardado en: {pdf_output_path}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    main(args.dataset)
    