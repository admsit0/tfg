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
N_BINS = 30  
USE_BATCHNORM = True 

# --- UMBRALES DE CONSENSO ---
AGREEMENT_THRESHOLD = 0.9 # Similitud Coseno (>= 0.95)
# EMD: Representa cuántas "cajas" (bins) de media se puede desplazar la masa. 
# 1.5 permite flexibilidad para distribuciones similares pero ligeramente desplazadas.
EMD_THRESHOLD = 0.8       

# --- CONFIGURACIÓN PARA EL EXPERIMENTO 2 (CUSTOM) ---
CUSTOM_EPOCH = 60
CUSTOM_TARGETS = {
    'Baseline': 0.0,
    'L2': 0.001,
    'Dropout': 0.3,
    'DataAug': 2,
    'GaussianNoise': 0.1,
    'BatchNorm': 1.0
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.grid": False,
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
# CARGA DE DATOS (DETERMINISTA)
# =========================================================
def get_validation_loader(dataset_name, batch_size=200):
    print(f"⬇️ Cargando Validación de {dataset_name} (Determinista)...")
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

# =========================================================
# OBTENCIÓN DE CAMPEONES (DOS MÉTODOS)
# =========================================================
def get_global_champions(data_dir):
    """EXPERIMENTO 1: Busca la mejor época absoluta por val_acc para cada método"""
    champions = {}
    baseline_method = None
    best_baseline_acc = -1
    best_baseline_ep = -1
    
    for f in os.listdir(data_dir):
        if f.startswith("data_CNN_") and f.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_dir, f))
            method = f.replace("data_CNN_", "").replace(".csv", "")
            
            # Buscar el mejor Baseline global
            df_base = df[df['reg_val'] == 0.0]
            if not df_base.empty:
                idx = df_base['val_acc'].idxmax()
                acc = df_base.loc[idx, 'val_acc']
                if acc > best_baseline_acc:
                    best_baseline_acc = acc
                    best_baseline_ep = int(df_base.loc[idx, 'epoch'])
                    baseline_method = method
            
            # Buscar el mejor parámetro global para el método
            df_reg = df[df['reg_val'] > 0.0]
            if not df_reg.empty:
                idx = df_reg['val_acc'].idxmax()
                champions[method] = {
                    'val': df_reg.loc[idx, 'reg_val'], 
                    'epoch': int(df_reg.loc[idx, 'epoch']),
                    'source_method': method
                }
                
    if baseline_method:
        champions['Baseline'] = {'val': 0.0, 'epoch': best_baseline_ep, 'source_method': baseline_method}
        
    return champions

def get_custom_champions(data_dir, custom_targets, target_epoch):
    """EXPERIMENTO 2: Usa los parámetros objetivo pasados por diccionario y una época fija"""
    champions = {}
    baseline_method = None
    
    # Encontrar qué archivo contiene el baseline
    for f in os.listdir(data_dir):
        if f.startswith("data_CNN_") and f.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_dir, f))
            if 0.0 in df['reg_val'].values:
                baseline_method = f.replace("data_CNN_", "").replace(".csv", "")
                break

    for method, val in custom_targets.items():
        if method == 'Baseline':
            champions['Baseline'] = {'val': 0.0, 'epoch': target_epoch, 'source_method': baseline_method}
        else:
            champions[method] = {'val': val, 'epoch': target_epoch, 'source_method': method}
            
    return champions

# =========================================================
# LÓGICA PRINCIPAL (CALIBRACIÓN Y VECTORES)
# =========================================================
def find_global_limits(device, base_models_dir, loader, num_classes, champions):
    
    base_info = champions.get('Baseline')
    if not base_info:
        raise ValueError("No hay Baseline disponible para calcular los límites.")
        
    baseline_file = f"CNN_{base_info['source_method']}_0.00000_ep{base_info['epoch']}.pth"
    filepath = os.path.join(base_models_dir, baseline_file)
    
    print(f"📏 Calibrando límites con el Baseline (Ep {base_info['epoch']})...")
    
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
    
    with torch.no_grad():
        for x, _ in loader:
            net(x.to(device))
            
    for h in handles: h.remove()
    return limits

def extract_state_vectors(net, loader, device, layers, limits):
    activations = {l: None for l in layers}
    handles = []
    
    def get_hook(layer_name):
        def hook_fn(m, i, o): activations[layer_name] = o.detach()
        return hook_fn
    
    handles.append(net.conv1.register_forward_hook(get_hook('conv1')))
    handles.append(net.conv3.register_forward_hook(get_hook('conv3')))
    handles.append(net.fc1.register_forward_hook(get_hook('fc1')))
    
    vectors = {l: [] for l in layers}
    
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            net(x)
            batch_size = x.size(0)
            
            for l in layers:
                act = activations[l].view(batch_size, -1) 
                act = torch.clamp(act, min=limits[l]['min'], max=limits[l]['max'])
                
                for i in range(batch_size):
                    hist = torch.histc(act[i], bins=N_BINS, min=limits[l]['min'], max=limits[l]['max']).float()
                    
                    # NORMALIZACIÓN CRÍTICA PARA EMD
                    hist_sum = hist.sum()
                    if hist_sum > 0:
                        hist = hist / hist_sum
                        
                    vectors[l].append(hist.cpu())
                    
    for h in handles: h.remove()
    for l in layers:
        vectors[l] = torch.stack(vectors[l])
        
    return vectors

# =========================================================
# GENERADOR VISUAL (COSINE + EMD)
# =========================================================
def plot_similarity_matrices(pdf, sim_matrices, emd_matrices, champions, processed_methods, layers, dataset_name, exp_title):
    
    # Creamos las etiquetas incluyendo la época para que sea súper explicativo
    method_labels = [f"{m}\n(Ep {champions[m]['epoch']})" for m in processed_methods]
    
    # --- PÁGINA 1: COSINE SIMILARITY ---
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, wspace=0.3)
    fig.suptitle(f"{exp_title}: SIMILITUD COSENO (Umbral $\\geq$ {AGREEMENT_THRESHOLD})\nDataset: {dataset_name} | N_Bins: {N_BINS}", fontsize=16, fontweight='bold')
    
    for idx, layer in enumerate(layers):
        ax = fig.add_subplot(gs[0, idx])
        matrix = sim_matrices[layer]
        
        cax = ax.imshow(matrix, cmap='YlGnBu', vmin=0, vmax=100)
        
        ax.set_xticks(np.arange(len(method_labels)))
        ax.set_yticks(np.arange(len(method_labels)))
        ax.set_xticklabels(method_labels, rotation=45, ha="right", fontsize=10, fontweight='bold')
        ax.set_yticklabels(method_labels, fontsize=10, fontweight='bold')
        
        for i in range(len(method_labels)):
            for j in range(len(method_labels)):
                val = matrix[i, j]
                color = "white" if val > 65 else "black"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=10, fontweight='bold')
                
        ax.set_title(f"Capa: {layer.upper()}", fontsize=14, fontweight='bold')
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    pdf.savefig(fig)
    plt.close(fig)

    # --- PÁGINA 2: EARTH MOVER'S DISTANCE (EMD) ---
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, wspace=0.3)
    fig.suptitle(f"{exp_title}: EARTH MOVER'S DISTANCE (Umbral $\\leq$ {EMD_THRESHOLD})\nDataset: {dataset_name} | N_Bins: {N_BINS}", fontsize=16, fontweight='bold')
    
    for idx, layer in enumerate(layers):
        ax = fig.add_subplot(gs[0, idx])
        matrix = emd_matrices[layer]
        
        cax = ax.imshow(matrix, cmap='YlGnBu', vmin=0, vmax=100) 
        
        ax.set_xticks(np.arange(len(method_labels)))
        ax.set_yticks(np.arange(len(method_labels)))
        ax.set_xticklabels(method_labels, rotation=45, ha="right", fontsize=10, fontweight='bold')
        ax.set_yticklabels(method_labels, fontsize=10, fontweight='bold')
        
        for i in range(len(method_labels)):
            for j in range(len(method_labels)):
                val = matrix[i, j]
                color = "white" if val > 65 else "black"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=10, fontweight='bold')
                
        ax.set_title(f"Capa: {layer.upper()}", fontsize=14, fontweight='bold')
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    pdf.savefig(fig)
    plt.close(fig)

# =========================================================
# MOTOR DEL EXPERIMENTO (REUTILIZABLE)
# =========================================================
def run_experiment_section(pdf, champions, val_loader, num_classes, device, models_dir, dataset_name, exp_title):
    
    method_names = list(champions.keys())
    print(f"🏆 Modelos seleccionados: {[(m, champions[m]['epoch']) for m in method_names]}")

    layers = ['conv1', 'conv3', 'fc1']
    global_limits = find_global_limits(device, models_dir, val_loader, num_classes, champions)
    
    all_vectors = {}
    
    for method_name, info in tqdm(champions.items(), desc=f"Vectorizando", colour='cyan'):
        val = info['val']
        src_method = info['source_method']
        ep = info['epoch']
        
        filename = f"CNN_{src_method}_{val:.5f}_ep{ep}.pth"
        filepath = os.path.join(models_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"\n⚠️ Archivo no encontrado: {filepath}. Saltando...")
            continue
            
        state_dict = torch.load(filepath, map_location=device)
        has_bn = any("bn1.weight" in k for k in state_dict.keys())
        net = ConvNet(reg_method=src_method, reg_val=val, num_classes=num_classes, use_bn=has_bn).to(device)
        net.load_state_dict(state_dict)
        net.eval()
        
        all_vectors[method_name] = extract_state_vectors(net, val_loader, device, layers, global_limits)

    processed_methods = list(all_vectors.keys())
    
    sim_matrices = {l: np.zeros((len(processed_methods), len(processed_methods))) for l in layers}
    emd_matrices = {l: np.zeros((len(processed_methods), len(processed_methods))) for l in layers}
    
    for l in layers:
        for i, m1 in enumerate(processed_methods):
            for j, m2 in enumerate(processed_methods):
                v1 = all_vectors[m1][l] 
                v2 = all_vectors[m2][l] 
                
                # --- MÉTRICA 1: SIMILITUD COSENO ---
                sim = F.cosine_similarity(v1, v2 + 1e-8, dim=1)
                consensus_votes_cos = (sim >= AGREEMENT_THRESHOLD).float()
                sim_matrices[l][i, j] = consensus_votes_cos.mean().item() * 100.0
                
                # --- MÉTRICA 2: EMD (Wasserstein-1) ---
                cdf1 = torch.cumsum(v1, dim=1)
                cdf2 = torch.cumsum(v2, dim=1)
                emd = torch.sum(torch.abs(cdf1 - cdf2), dim=1)
                
                consensus_votes_emd = (emd <= EMD_THRESHOLD).float()
                emd_matrices[l][i, j] = consensus_votes_emd.mean().item() * 100.0

    plot_similarity_matrices(pdf, sim_matrices, emd_matrices, champions, processed_methods, layers, dataset_name, exp_title)

# =========================================================
# FUNCIÓN PRINCIPAL
# =========================================================
def main(dataset_name):
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    input_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'outputs_{dataset_name}')
    models_dir = os.path.join(input_base_dir, 'models')
    data_dir = os.path.join(input_base_dir, 'data')
    
    output_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dynamics', 'similarity_analysis', f'{dataset_name}_similarity')
    os.makedirs(output_base_dir, exist_ok=True)
    
    pdf_output_path = os.path.join(output_base_dir, f'Consensus_Analysis_{dataset_name}.pdf')
    
    val_loader, num_classes = get_validation_loader(dataset_name)
    
    print(f"\n🧪 INICIANDO EXPERIMENTO DE SIMILITUD (Dataset: {dataset_name})")
    
    with PdfPages(pdf_output_path) as pdf:
        
        # --------------------------------------------------------------------
        # SECCIÓN 1: MEJOR ÉPOCA GLOBAL POR VAL_ACC
        # --------------------------------------------------------------------
        print("\n" + "="*50)
        print("▶ SECCIÓN 1: MEJORES MODELOS (ÉPOCA GLOBAL ÓPTIMA)")
        print("="*50)
        global_champions = get_global_champions(data_dir)
        if 'Baseline' not in global_champions:
            print("❌ No se encontró Baseline para la Sección 1.")
        else:
            run_experiment_section(pdf, global_champions, val_loader, num_classes, device, models_dir, dataset_name, "Exp 1: Mejor Val_Acc (Épocas Óptimas)")

        # --------------------------------------------------------------------
        # SECCIÓN 2: PARÁMETROS CUSTOMIZADOS
        # --------------------------------------------------------------------
        print("\n" + "="*50)
        print(f"▶ SECCIÓN 2: PARÁMETROS OBJETIVO (ÉPOCA FIJA: {CUSTOM_EPOCH})")
        print("="*50)
        custom_champions = get_custom_champions(data_dir, CUSTOM_TARGETS, CUSTOM_EPOCH)
        if 'Baseline' not in custom_champions:
             print("❌ No se encontró Baseline para la Sección 2.")
        else:
            run_experiment_section(pdf, custom_champions, val_loader, num_classes, device, models_dir, dataset_name, f"Exp 2: Parámetros Custom (Época {CUSTOM_EPOCH})")

    print(f"\n🚀 ¡PDF de 4 páginas completado y guardado en: {pdf_output_path}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Nombre del dataset (ej. CIFAR10, SVHN, FashionMNIST)")
    args = parser.parse_args()
    
    main(args.dataset)