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
import copy

TARGET_EPOCH = 60
USE_BATCHNORM = True

NOISE_LEVELS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15]
N_RUNS = 3

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

def evaluate_with_noise(net, loader, device, original_state_dict, sigma, target_layer):
    net.load_state_dict(original_state_dict)
    
    if sigma > 0.0:
        with torch.no_grad():
            for name, param in net.named_parameters():
                if target_layer in name and 'weight' in name and 'bn' not in name:
                    noise = torch.randn_like(param) * sigma
                    param.add_(noise)
    
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = net(x)
            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            
    return (correct / total) * 100.0

def plot_flat_minima(pdf, df, dataset_name, exp_title):
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
            
    ax.set_title("Robustez a la Perturbacion de Pesos (Flat Minima Test)", fontsize=14)
    ax.set_xlabel("Magnitud del Ruido en Pesos (Desviacion Estandar $\\sigma$)", fontsize=12)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_xticks(NOISE_LEVELS)
    
    ax.legend(fontsize=11)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

def run_experiment_section(pdf, champions, val_loader, num_classes, device, models_dir, dataset_name, exp_title, csv_path, target_layer):
    method_names = list(champions.keys())
    print(f"Modelos seleccionados: {[(m, champions[m]['val']) for m in method_names]}")

    results = []
    
    for method_name, info in tqdm(champions.items(), desc=f"Evaluando Flat Minima ({target_layer})", colour='magenta'):
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
        
        clean_state_dict = copy.deepcopy(net.state_dict())
        
        for sigma in NOISE_LEVELS:
            if sigma == 0.0:
                acc = evaluate_with_noise(net, val_loader, device, clean_state_dict, 0.0, target_layer)
                mean_acc, std_acc = acc, 0.0
            else:
                runs_acc = []
                for _ in range(N_RUNS):
                    acc = evaluate_with_noise(net, val_loader, device, clean_state_dict, sigma, target_layer)
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
    
    plot_flat_minima(pdf, df_results, dataset_name, exp_title)

def main(dataset_name):
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    input_base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'analysis', f'outputs_{dataset_name}')
    models_dir = os.path.join(input_base_dir, 'models')
    data_dir = os.path.join(input_base_dir, 'data')
    
    output_base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'robustness_weight_noise')
    os.makedirs(output_base_dir, exist_ok=True)
    
    pdf_output_path = os.path.join(output_base_dir, f'Flat_Minima_Report_{dataset_name}.pdf')
    
    print(f"\nINICIANDO EXPERIMENTO FLAT MINIMA (Dataset: {dataset_name})")
    val_loader, num_classes = get_validation_loader(dataset_name)
    
    with PdfPages(pdf_output_path) as pdf:
        global_champions = get_global_champions(data_dir)
        custom_champions = get_custom_champions(data_dir, CUSTOM_TARGETS)
        
        # for layer in ['conv1', 'conv3', 'fc1']:
        for layer in ['fc1']:
            print("\n" + "="*50)
            print(f"PERTURBANDO CAPA: {layer.upper()}")
            print("="*50)
            
            if 'Baseline' in global_champions:
                csv_optimum_path = os.path.join(output_base_dir, f'Flat_Minima_Data_{dataset_name}_{layer}_Optimum.csv')
                run_experiment_section(pdf, global_champions, val_loader, num_classes, device, models_dir, dataset_name, f"Exp 1: Flat Minima - Capa {layer.upper()} (Optimos)", csv_optimum_path, layer)

            if 'Baseline' in custom_champions:
                csv_custom_path = os.path.join(output_base_dir, f'Flat_Minima_Data_{dataset_name}_{layer}_Custom.csv')
                run_experiment_section(pdf, custom_champions, val_loader, num_classes, device, models_dir, dataset_name, f"Exp 2: Flat Minima - Capa {layer.upper()} (Custom)", csv_custom_path, layer)

    print(f"\nPDF completado y guardado en: {pdf_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    main(args.dataset)

"""
El L2 funciona forzando a que los pesos originales de la red se hagan minúsculos (muy cercanos a 0).
Por lo tanto, si le inyectas un ruido absoluto de $\sigma=0.05$ a un peso de L2 que mide 0.01, acabas de destruir el peso por completo
(un 500% de ruido).
En cambio, el Baseline tiene pesos descontrolados que miden 1.5, por lo que ese mismo ruido apenas le hace cosquillas temporalmente.
"""

