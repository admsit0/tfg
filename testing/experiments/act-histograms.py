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
# CONFIGURACIÓN
# =========================================================
N_BINS = 30  

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--"
})

# =========================================================
# CLASES DE LA RED 
# =========================================================
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

# =========================================================
# CARGA DE DATOS
# =========================================================
def get_dataset_loaders(dataset_name, batch_size=200, limit=25000):
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
        x_test = F.pad(test_set.data.float() / 255.0, (2, 2, 2, 2)).unsqueeze(1).repeat(1, 3, 1, 1)
        y_train = torch.tensor(train_set.targets, dtype=torch.long)
        y_test = torch.tensor(test_set.targets, dtype=torch.long)
        num_classes = 10
    elif dataset_name == 'CIFAR100':
        train_set = datasets.CIFAR100(root='./data_raw', train=True, download=True)
        test_set = datasets.CIFAR100(root='./data_raw', train=False, download=True)
        x_train = torch.tensor(train_set.data).permute(0, 3, 1, 2).float() / 255.0
        y_train = torch.tensor(train_set.targets, dtype=torch.long)
        x_test = torch.tensor(test_set.data).permute(0, 3, 1, 2).float() / 255.0
        y_test = torch.tensor(test_set.targets, dtype=torch.long)
        num_classes = 100
    
    if limit:
        x_train, y_train = x_train[:limit], y_train[:limit]
        
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, num_classes

# =========================================================
# AYUDANTES
# =========================================================
def extract_model_info(filename):
    pattern = r"CNN_(.*?)_([0-9\.]+)_ep(\d+)\.pth"
    match = re.search(pattern, filename)
    if match: return match.group(1), float(match.group(2)), int(match.group(3))
    return None, None, None

def get_best_epochs_and_acc(data_dir):
    best_models = {}
    if not os.path.exists(data_dir): return best_models
    
    for f in os.listdir(data_dir):
        if f.startswith("data_CNN_") and f.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_dir, f))
            method = f.replace("data_CNN_", "").replace(".csv", "")
            
            idx_max = df.groupby('reg_val')['val_acc'].idxmax()
            for _, row in df.loc[idx_max].iterrows():
                val = round(row['reg_val'], 6)
                best_models[(method, val)] = {
                    'epoch': int(row['epoch']),
                    'val_acc': row['val_acc'],
                    'train_acc': row['train_acc']
                }
    return best_models

def setup_log_scale_safe(ax, method):
    if method in ["L1", "L2", "GaussianNoise"]:
        ax.set_xscale("symlog", linthresh=1e-5)

# =========================================================
# LÓGICA PRINCIPAL (HOOKS Y ESTADOS)
# =========================================================
def find_global_limits(device, base_models_dir, train_loader, num_classes, best_models_info):
    print("📏 Calculando límites globales del Baseline (Dropout=0.0 o L1=0.0)...")
    
    baseline_method = None
    best_ep = None
    for (m, v), info in best_models_info.items():
        if v == 0.0:
            baseline_method = m
            best_ep = info['epoch']
            break
            
    if not baseline_method:
        raise ValueError("No se ha encontrado un modelo Baseline (valor 0.0) en los CSVs.")

    baseline_file = None
    model_files = [f for f in os.listdir(base_models_dir) if f.endswith('.pth')]
    for f in model_files:
        m, v, ep = extract_model_info(f)
        if m == baseline_method and v == 0.0 and ep == best_ep:
            baseline_file = f
            break
            
    if not baseline_file:
        raise ValueError(f"No se encontró el .pth del baseline ({baseline_method} 0.0, ep {best_ep})")
        
    filepath = os.path.join(base_models_dir, baseline_file)
    
    state_dict = torch.load(filepath, map_location=device)
    has_bn = any("bn1.weight" in k for k in state_dict.keys())
    net = ConvNet(num_classes=num_classes, use_bn=has_bn).to(device)
    net.load_state_dict(state_dict)
    net.eval()
    
    layers = ['conv1', 'conv3', 'fc1']
    limits = {l: {'min': float('inf'), 'max': float('-inf')} for l in layers}
    
    handles = []
    
    def get_hook(layer_name):
        def hook_fn(m, i, o):
            batch_min = o.detach().min().item()
            batch_max = o.detach().max().item()
            limits[layer_name]['min'] = min(limits[layer_name]['min'], batch_min)
            limits[layer_name]['max'] = max(limits[layer_name]['max'], batch_max)
        return hook_fn
    
    handles.append(net.conv1.register_forward_hook(get_hook('conv1')))
    handles.append(net.conv3.register_forward_hook(get_hook('conv3')))
    handles.append(net.fc1.register_forward_hook(get_hook('fc1')))
    
    print(f"   -> Escaneando dataset entero con el modelo {baseline_file}...")
    with torch.no_grad():
        for x, _ in train_loader:
            net(x.to(device))
            
    for h in handles: h.remove()
    print(f"   -> Límites calibrados: {limits}")
    return limits

def compute_unique_states(net, loader, device, layers, limits):
    activations = {l: None for l in layers}
    handles = []
    
    def get_hook(layer_name):
        def hook_fn(m, i, o): activations[layer_name] = o.detach()
        return hook_fn
    
    handles.append(net.conv1.register_forward_hook(get_hook('conv1')))
    handles.append(net.conv3.register_forward_hook(get_hook('conv3')))
    handles.append(net.fc1.register_forward_hook(get_hook('fc1')))
    
    unique_states = {l: set() for l in layers}
    total_images = 0
    
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            net(x)
            batch_size = x.size(0)
            total_images += batch_size
            
            for l in layers:
                act = activations[l] 
                act = act.view(batch_size, -1) 
                
                act = torch.clamp(act, min=limits[l]['min'], max=limits[l]['max'])
                
                for i in range(batch_size):
                    hist = torch.histc(act[i], bins=N_BINS, min=limits[l]['min'], max=limits[l]['max'])
                    state = tuple(hist.cpu().numpy().astype(int))
                    unique_states[l].add(state)
                    
    for h in handles: h.remove()
    
    results = {}
    for l in layers:
        results[l] = (len(unique_states[l]) / total_images) * 100.0
        
    return results, total_images

# =========================================================
# GENERADOR DEL REPORTE PDF
# =========================================================
def generate_pdf_report(df, output_pdf, dataset_name):
    methods = df['reg_method'].unique()
    layers = ['conv1', 'conv3', 'fc1']
    
    with PdfPages(output_pdf) as pdf:
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(3, 3, wspace=0.3, hspace=0.4)
        fig.suptitle(f"EXECUTIVE SUMMARY: Internal States ({dataset_name})\n(N_bins = {N_BINS}, Validation Set)", fontsize=20, fontweight='bold')
        
        for col_idx, layer in enumerate(layers):
            df_layer = df[(df['layer'] == layer) & (df['split'] == 'val')]
            
            ax0 = fig.add_subplot(gs[0, col_idx])
            for m in methods:
                sub = df_layer[df_layer['reg_method'] == m].sort_values('reg_val')
                ax0.plot(sub['reg_val'], sub['unique_pctg'], marker='o', label=m, alpha=0.8)
                setup_log_scale_safe(ax0, m)
            ax0.set_title(f"{layer} - States (%) vs Reg Val")
            ax0.set_ylabel("Unique States (%)")
            ax0.set_xlabel("Reg Value")
            if col_idx == 0: ax0.legend(fontsize=8)
                
            ax1 = fig.add_subplot(gs[1, col_idx])
            for m in methods:
                sub = df_layer[df_layer['reg_method'] == m].sort_values('reg_val')
                ax1.plot(sub['reg_val'], sub['val_acc'], marker='s', linestyle='--', label=m, alpha=0.8)
                setup_log_scale_safe(ax1, m)
            ax1.set_title(f"Accuracy vs Reg Val (Repeated)")
            ax1.set_ylabel("Val Acc")
            ax1.set_xlabel("Reg Value")
            
            ax2 = fig.add_subplot(gs[2, col_idx])
            for m in methods:
                sub = df_layer[df_layer['reg_method'] == m]
                ax2.scatter(sub['val_acc'], sub['unique_pctg'], label=m, alpha=0.8, s=50)
            ax2.set_title(f"{layer} - States (%) vs Validation Acc")
            ax2.set_xlabel("Val Acc")
            ax2.set_ylabel("Unique States (%)")
            
        pdf.savefig(fig)
        plt.close(fig)
        
        for m in methods:
            fig = plt.figure(figsize=(18, 12))
            gs = gridspec.GridSpec(3, 3, wspace=0.3, hspace=0.4)
            fig.suptitle(f"METHOD ANALYSIS: {m} ({dataset_name})", fontsize=18, fontweight='bold', color='darkblue')
            
            for col_idx, layer in enumerate(layers):
                df_layer_val = df[(df['layer'] == layer) & (df['split'] == 'val') & (df['reg_method'] == m)].sort_values('reg_val')
                df_layer_train = df[(df['layer'] == layer) & (df['split'] == 'train') & (df['reg_method'] == m)].sort_values('reg_val')
                
                ax0 = fig.add_subplot(gs[0, col_idx])
                ax0.plot(df_layer_val['reg_val'], df_layer_val['unique_pctg'], marker='o', label='Val States', color='orange')
                ax0.plot(df_layer_train['reg_val'], df_layer_train['unique_pctg'], marker='x', label='Train States', color='blue', linestyle='--')
                setup_log_scale_safe(ax0, m)
                ax0.set_title(f"{layer} - States (%)")
                ax0.set_ylabel("Unique States (%)")
                ax0.legend()
                
                ax1 = fig.add_subplot(gs[1, col_idx])
                ax1.plot(df_layer_val['reg_val'], df_layer_val['val_acc'], marker='o', label='Val Acc', color='orange')
                ax1.plot(df_layer_train['reg_val'], df_layer_train['train_acc'], marker='x', label='Train Acc', color='blue', linestyle='--')
                setup_log_scale_safe(ax1, m)
                ax1.set_title("Accuracy")
                ax1.set_ylabel("Accuracy")
                ax1.legend()
                
                ax2 = fig.add_subplot(gs[2, col_idx])
                ax2.scatter(df_layer_val['val_acc'], df_layer_val['unique_pctg'], label='Val', color='orange', s=60)
                ax2.scatter(df_layer_train['train_acc'], df_layer_train['unique_pctg'], label='Train', color='blue', marker='x', s=60)
                ax2.set_title(f"{layer} - States vs Acc")
                ax2.set_xlabel("Accuracy")
                ax2.set_ylabel("Unique States (%)")
                ax2.legend()
                
            pdf.savefig(fig)
            plt.close(fig)

# =========================================================
# FUNCIÓN PRINCIPAL
# =========================================================
def main(dataset_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    input_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'outputs_{dataset_name}')
    models_dir = os.path.join(input_base_dir, 'models')
    data_dir = os.path.join(input_base_dir, 'data')
    
    output_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'activation_histograms', f'{dataset_name}')
    os.makedirs(output_base_dir, exist_ok=True)
    
    csv_output_path = os.path.join(output_base_dir, 'states_results.csv')
    pdf_output_path = os.path.join(output_base_dir, f'states_report_{dataset_name}.pdf')
    
    print(f"\n📂 Analizando Dataset: {dataset_name}")
    
    train_loader, test_loader, num_classes = get_dataset_loaders(dataset_name)
    best_models_info = get_best_epochs_and_acc(data_dir)
    
    if not best_models_info:
        print("❌ No se encontraron datos CSV del experimento previo.")
        return

    layers = ['conv1', 'conv3', 'fc1']
    global_limits = find_global_limits(device, models_dir, train_loader, num_classes, best_models_info)
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    pth_map = {}
    for f in model_files:
        m, v, ep = extract_model_info(f)
        if m and (m, v) in best_models_info:
            if best_models_info[(m, v)]['epoch'] == ep:
                pth_map[(m, v)] = f

    results = []
    
    for (method, val), info in tqdm(best_models_info.items(), desc="Procesando modelos", colour='magenta'):
        if (method, val) not in pth_map:
            print(f"\n⚠️  No se encontró el .pth para {method}={val} (Ep: {info['epoch']})")
            continue
            
        filepath = os.path.join(models_dir, pth_map[(method, val)])
        
        state_dict = torch.load(filepath, map_location=device)
        has_bn = any("bn1.weight" in k for k in state_dict.keys())
        
        net = ConvNet(reg_method=method, reg_val=val, num_classes=num_classes, use_bn=has_bn).to(device)
        net.load_state_dict(state_dict)
        net.eval()
        
        train_states_pct, n_train = compute_unique_states(net, train_loader, device, layers, global_limits)
        val_states_pct, n_val = compute_unique_states(net, test_loader, device, layers, global_limits)
        
        for l in layers:
            results.append({
                'reg_method': method, 'reg_val': val, 'best_epoch': info['epoch'],
                'val_acc': info['val_acc'], 'train_acc': info['train_acc'],
                'layer': l, 'split': 'train', 
                'unique_pctg': train_states_pct[l], 'num_images': n_train, 'n_bins': N_BINS
            })
            results.append({
                'reg_method': method, 'reg_val': val, 'best_epoch': info['epoch'],
                'val_acc': info['val_acc'], 'train_acc': info['train_acc'],
                'layer': l, 'split': 'val', 
                'unique_pctg': val_states_pct[l], 'num_images': n_val, 'n_bins': N_BINS
            })

    df_results = pd.DataFrame(results)
    df_results.to_csv(csv_output_path, index=False)
    print(f"✅ Resultados numéricos guardados en {csv_output_path}")
    
    print("📊 Generando gráficos y ensamblando PDF...")
    generate_pdf_report(df_results, pdf_output_path, dataset_name)
    print(f"🚀 ¡Reporte PDF completado: {pdf_output_path}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Nombre del dataset (ej. CIFAR10, SVHN, FashionMNIST)")
    args = parser.parse_args()
    
    main(args.dataset)