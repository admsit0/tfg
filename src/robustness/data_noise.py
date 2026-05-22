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
from torch.utils.data import TensorDataset, DataLoader
import torchvision.datasets as datasets
from tqdm import tqdm
import argparse

# =========================================================
# CONFIGURACIÓN ESTRICTA DEL EXPERIMENTO
# =========================================================
TARGET_EPOCH = 60
USE_BATCHNORM = True

# Niveles de ruido en la IMAGEN (Desviación estándar)
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
N_RUNS = 3 # Evaluamos 3 veces por cada nivel de ruido estocástico

CUSTOM_TARGETS = {
    'Baseline': 0.0,
    'L1': 0.0001
}


plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--"
})

class GaussianNoise(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training and self.std > 0:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x

class ConvNet(nn.Module):
    def __init__(self, reg_method='', reg_val=0.0, num_classes=10, use_bn=False):
        super(ConvNet, self).__init__()
        dropout_p = reg_val if reg_method == 'Dropout' else 0.0
        noise_std = reg_val if reg_method == 'GaussianNoise' else 0.0
        
        self.use_bn = use_bn or (reg_method == 'BatchNorm')
        
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
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.noise(x)
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

def get_validation_loader(dataset_name, batch_size=500):
    print(f"Cargando Validacion de {dataset_name}...")
    if dataset_name == 'CIFAR10':
        test_set = datasets.CIFAR10(root='./data_raw', train=False, download=True)
        x_test = torch.tensor(test_set.data).permute(0, 3, 1, 2).float() / 255.0
        y_test = torch.tensor(test_set.targets, dtype=torch.long)
        num_classes = 10
    elif dataset_name == 'SVHN':
        test_set = datasets.SVHN(root='./data_raw', split='test', download=True)
        x_test = torch.tensor(test_set.data).float() / 255.0
        y_test = torch.tensor(test_set.labels, dtype=torch.long)
        num_classes = 10
    elif dataset_name == 'FashionMNIST':
        test_set = datasets.FashionMNIST(root='./data_raw', train=False, download=True)
        x_test = F.pad(test_set.data.float() / 255.0, (2, 2, 2, 2)).unsqueeze(1).repeat(1, 3, 1, 1)
        y_test = torch.tensor(test_set.targets, dtype=torch.long)
        num_classes = 10
    elif dataset_name == 'CIFAR100':
        test_set = datasets.CIFAR100(root='./data_raw', train=False, download=True)
        x_test = torch.tensor(test_set.data).permute(0, 3, 1, 2).float() / 255.0
        y_test = torch.tensor(test_set.targets, dtype=torch.long)
        num_classes = 100
        
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    return test_loader, num_classes

def get_global_champions(data_dir):
    champions = {}
    baseline_source_method = None
    for f in os.listdir(data_dir):
        if f.startswith("data_CNN_") and f.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_dir, f))
            method = f.replace("data_CNN_", "").replace(".csv", "")
            if 0.0 in df['reg_val'].values and baseline_source_method is None:
                champions['Baseline'] = {'val': 0.0, 'source_method': method}
                baseline_source_method = method
            df_reg = df[df['reg_val'] > 0.0]
            if not df_reg.empty:
                best_idx = df_reg['val_acc'].idxmax()
                champ_val = df_reg.loc[best_idx, 'reg_val']
                champions[method] = {'val': champ_val, 'source_method': method}
    return champions

def get_custom_champions(data_dir, custom_targets):
    champions = {}
    baseline_method = None
    for f in os.listdir(data_dir):
        if f.startswith("data_CNN_") and f.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_dir, f))
            if 0.0 in df['reg_val'].values:
                baseline_method = f.replace("data_CNN_", "").replace(".csv", "")
                break
    for method, val in custom_targets.items():
        if method == 'Baseline':
            champions['Baseline'] = {'val': 0.0, 'source_method': baseline_method}
        else:
            champions[method] = {'val': val, 'source_method': method}
    return champions

def evaluate_with_data_noise(net, loader, device, sigma):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # INYECCIÓN DE RUIDO EN LOS DATOS
            if sigma > 0.0:
                noise = torch.randn_like(x) * sigma
                x = torch.clamp(x + noise, 0.0, 1.0) # Mantenemos los píxeles en rango válido
            
            out = net(x)
            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            
    return (correct / total) * 100.0

def plot_data_robustness(pdf, df, dataset_name, exp_title):
    methods = df['method'].unique()
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    fig.suptitle(f"{exp_title}\nDataset: {dataset_name} | Epoca: {TARGET_EPOCH}", fontsize=16, fontweight='bold')
    
    for m in methods:
        sub = df[df['method'] == m].sort_values('sigma')
        if not sub.empty:
            linewidth = 3.5 if m == 'Baseline' else 2.5
            linestyle = '--' if m == 'Baseline' else '-'
            reg_val = sub['reg_val'].iloc[0]
            label_name = f"{m}" if m == 'Baseline' else f"{m} (val: {reg_val})"
            
            ax.plot(sub['sigma'], sub['acc_mean'], marker='o', linewidth=linewidth, linestyle=linestyle, label=label_name)
            ax.fill_between(sub['sigma'], sub['acc_mean'] - sub['acc_std'], sub['acc_mean'] + sub['acc_std'], alpha=0.15)
            
    ax.set_title("Robustez a la Perturbacion de Datos (Out-Of-Distribution Test)", fontsize=14)
    ax.set_xlabel("Magnitud del Ruido en Imagenes (Desviacion Estandar $\\sigma$)", fontsize=12)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_xticks(NOISE_LEVELS)
    ax.legend(fontsize=11)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

def run_experiment_section(pdf, champions, val_loader, device, models_dir, dataset_name, exp_title, csv_path, num_classes):
    method_names = list(champions.keys())
    print(f"Modelos seleccionados: {[(m, champions[m]['val']) for m in method_names]}")

    results = []
    
    for method_name, info in tqdm(champions.items(), desc=f"Evaluando Robustez en Datos", colour='green'):
        val = info['val']
        src_method = info['source_method']
        
        filename = f"CNN_{src_method}_{val:.5f}_ep{TARGET_EPOCH}.pth"
        filepath = os.path.join(models_dir, filename)
        
        if not os.path.exists(filepath):
            continue
            
        state_dict = torch.load(filepath, map_location=device)
        has_bn = any("bn1.weight" in k for k in state_dict.keys())
        net = ConvNet(reg_method=src_method, reg_val=val, num_classes=num_classes, use_bn=has_bn).to(device)
        net.load_state_dict(state_dict)
        
        for sigma in NOISE_LEVELS:
            if sigma == 0.0:
                acc = evaluate_with_data_noise(net, val_loader, device, 0.0)
                mean_acc, std_acc = acc, 0.0
            else:
                runs_acc = []
                for _ in range(N_RUNS):
                    acc = evaluate_with_data_noise(net, val_loader, device, sigma)
                    runs_acc.append(acc)
                mean_acc = np.mean(runs_acc)
                std_acc = np.std(runs_acc)
            
            results.append({
                'method': method_name,
                'reg_val': val,
                'sigma': sigma,
                'acc_mean': mean_acc,
                'acc_std': std_acc
            })

    df_results = pd.DataFrame(results)
    df_results.to_csv(csv_path, index=False)
    plot_data_robustness(pdf, df_results, dataset_name, exp_title)

def main(dataset_name):
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    input_base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'analysis', f'outputs_{dataset_name}')
    models_dir = os.path.join(input_base_dir, 'models')
    data_dir = os.path.join(input_base_dir, 'data')
    
    output_base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'robustness_data_noise')
    os.makedirs(output_base_dir, exist_ok=True)
    
    pdf_output_path = os.path.join(output_base_dir, f'Data_Noise_Report_{dataset_name}.pdf')
    
    print(f"\nINICIANDO EXPERIMENTO RUIDO EN DATOS (Dataset: {dataset_name})")
    val_loader, num_classes = get_validation_loader(dataset_name)
    
    with PdfPages(pdf_output_path) as pdf:
        global_champions = get_global_champions(data_dir)
        custom_champions = get_custom_champions(data_dir, CUSTOM_TARGETS)
        
        if 'Baseline' in global_champions:
            csv_optimum_path = os.path.join(output_base_dir, f'Data_Noise_Data_{dataset_name}_Optimum.csv')
            run_experiment_section(pdf, global_champions, val_loader, device, models_dir, dataset_name, "Exp 1: Robustez a Ruido en Datos (Optimos)", csv_optimum_path, num_classes)

        if 'Baseline' in custom_champions:
            csv_custom_path = os.path.join(output_base_dir, f'Data_Noise_Data_{dataset_name}_Custom.csv')
            run_experiment_section(pdf, custom_champions, val_loader, device, models_dir, dataset_name, "Exp 2: Robustez a Ruido en Datos (Custom)", csv_custom_path, num_classes)

    print(f"\nPDF completado y guardado en: {pdf_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    main(args.dataset)
    