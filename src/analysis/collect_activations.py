import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.datasets as datasets

# =========================================================
# CONFIG
# =========================================================
DATASET_NAME = 'SVHN' # Cambia esto a SVHN, FashionMNIST etc. para evaluar otras carpetas

# =========================================================
# CLASES DE LA RED (Idénticas)
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
    def __init__(self, reg_method='', reg_val=0.0, num_classes=10):
        super(ConvNet, self).__init__()
        dropout_p = reg_val if reg_method == 'Dropout' else 0.0
        noise_std = reg_val if reg_method == 'GaussianNoise' else 0.0
        self.use_bn = (reg_method == 'BatchNorm')
        
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
# FUNCIONES AUXILIARES
# =========================================================
def get_test_loader_and_classes():
    if DATASET_NAME == 'CIFAR10':
        test_set = datasets.CIFAR10(root='./data_raw', train=False, download=True)
        x_test = torch.tensor(test_set.data).permute(0, 3, 1, 2).float() / 255.0
        y_test = torch.tensor(test_set.targets, dtype=torch.long)
        num_classes = 10
    elif DATASET_NAME == 'SVHN':
        test_set = datasets.SVHN(root='./data_raw', split='test', download=True)
        x_test = torch.tensor(test_set.data).float() / 255.0
        y_test = torch.tensor(test_set.labels, dtype=torch.long)
        num_classes = 10
    elif DATASET_NAME == 'FashionMNIST':
        test_set = datasets.FashionMNIST(root='./data_raw', train=False, download=True)
        x_test_raw = test_set.data.float() / 255.0
        x_test = F.pad(x_test_raw, (2, 2, 2, 2)).unsqueeze(1).repeat(1, 3, 1, 1)
        y_test = torch.tensor(test_set.targets, dtype=torch.long)
        num_classes = 10
    elif DATASET_NAME == 'CIFAR100':
        test_set = datasets.CIFAR100(root='./data_raw', train=False, download=True)
        x_test = torch.tensor(test_set.data).permute(0, 3, 1, 2).float() / 255.0
        y_test = torch.tensor(test_set.targets, dtype=torch.long)
        num_classes = 100
        
    return DataLoader(TensorDataset(x_test, y_test), batch_size=1000), num_classes

def extract_model_info(filename):
    pattern = r"CNN_(.*?)_([0-9\.]+)_ep(\d+)\.pth"
    match = re.search(pattern, filename)
    if match: return match.group(1), float(match.group(2)), int(match.group(3))
    return None, None, None

def get_best_epochs_from_csv(csv_path):
    best_epochs = {}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        idx_max = df.groupby('reg_val')['val_acc'].idxmax()
        for _, row in df.loc[idx_max].iterrows():
            best_epochs[round(row['reg_val'], 6)] = int(row['epoch'])
    return best_epochs

# =========================================================
# MAIN
# =========================================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # LA RUTA SE BASA EN EL DATASET_NAME
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'outputs_{DATASET_NAME}')
    models_dir = os.path.join(base_dir, 'models')
    data_dir = os.path.join(base_dir, 'data')
    output_pdf = os.path.join(base_dir, f'activation_histograms_multilayer_{DATASET_NAME}.pdf')
    
    test_loader, num_classes = get_test_loader_and_classes()
    
    print("🔍 Escaneando modelos...")
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    available_models = {}
    for f in model_files:
        method, val, epoch = extract_model_info(f)
        if method:
            val_rounded = round(val, 6)
            if method not in available_models: available_models[method] = {}
            if val_rounded not in available_models[method]: available_models[method][val_rounded] = {}
            available_models[method][val_rounded][epoch] = f

    best_models_to_process = {} 
    for method, vals_dict in available_models.items():
        csv_path = os.path.join(data_dir, f"data_CNN_{method}.csv")
        best_epochs = get_best_epochs_from_csv(csv_path)
        best_models_to_process[method] = []
        for val_rounded, epochs_dict in vals_dict.items():
            target_epoch = best_epochs.get(val_rounded)
            if target_epoch is None or target_epoch not in epochs_dict:
                target_epoch = max(epochs_dict.keys())
            best_models_to_process[method].append((val_rounded, epochs_dict[target_epoch], target_epoch))
        best_models_to_process[method].sort(key=lambda x: x[0])

    print("📊 Extrayendo activaciones Multi-capa y generando PDF...")
    
    # CAPAS A ANALIZAR
    layers_to_hook = ['conv1', 'conv3', 'fc1']

    with PdfPages(output_pdf) as pdf:
        for method, items in best_models_to_process.items():
            print(f"\n--- Procesando Método: {method} ---")
            
            for val, filename, epoch in items:
                # 1 Página = 1 Modelo (Configuración exacta), con 3 filas (Capas) x 2 columnas (Tipos de Gráfico)
                fig, axes = plt.subplots(len(layers_to_hook), 2, figsize=(14, 4 * len(layers_to_hook)))
                fig.suptitle(f'{method} = {val} | Dataset: {DATASET_NAME} | Best Ep: {epoch}', fontsize=16, fontweight='bold')
                
                filepath = os.path.join(models_dir, filename)
                net = ConvNet(reg_method=method, reg_val=val, num_classes=num_classes).to(device)
                net.load_state_dict(torch.load(filepath, map_location=device))
                net.eval()
                
                # Setup Hooks dinámicos
                activations = {layer: [] for layer in layers_to_hook}
                handles = []
                
                def get_hook(layer_name):
                    def hook_fn(m, i, o):
                        activations[layer_name].append(o.detach().cpu().numpy())
                    return hook_fn
                
                handles.append(net.conv1.register_forward_hook(get_hook('conv1')))
                handles.append(net.conv3.register_forward_hook(get_hook('conv3')))
                handles.append(net.fc1.register_forward_hook(get_hook('fc1')))
                
                with torch.no_grad():
                    for x, _ in test_loader:
                        net(x.to(device))
                        
                for h in handles: h.remove()
                
                # Procesar y plotear capa por capa
                for row_idx, layer_name in enumerate(layers_to_hook):
                    # Concatenamos todo el batch. Formato: (N, C, H, W) para Convs, (N, Neuronas) para FC
                    full_acts = np.concatenate(activations[layer_name], axis=0) 
                    sparsity_pct = np.mean(full_acts <= 0) * 100
                    
                    # 1. Gráfico Global (Derecha en paper Srivastava)
                    acts_raw_flat = full_acts.flatten()
                    acts_alive = acts_raw_flat[acts_raw_flat > 0]
                    
                    # 2. Gráfico Media por Neurona (Izquierda en paper Srivastava)
                    # Si es convolucional (N, C, H, W), hacemos media en la dimensión de la imagen (H, W) y N para sacar la media por filtro C, 
                    # o directamente la media por cada "unidad" (NxUnits). El paper lo hace por unidad.
                    mean_acts = full_acts.mean(axis=0).flatten() 
                    mean_acts_alive = mean_acts[mean_acts > 0]
                    
                    # Eje Izquierdo: MEDIA
                    ax_left = axes[row_idx, 0]
                    ax_left.hist(mean_acts_alive, bins=50, color='royalblue', alpha=0.8)
                    ax_left.set_title(f'{layer_name} - Mean Activation/Unit')
                    ax_left.set_xlabel('Mean Activation')
                    ax_left.set_ylabel('Num Units')
                    ax_left.grid(True, alpha=0.3)
                    
                    # Eje Derecho: GLOBAL
                    ax_right = axes[row_idx, 1]
                    ax_right.hist(acts_alive, bins=100, color='indigo', alpha=0.7)
                    ax_right.set_title(f'{layer_name} - Global Freq | Sparsity: {sparsity_pct:.1f}%')
                    ax_right.set_yscale('log')
                    ax_right.set_xlabel('Activation Value')
                    ax_right.set_ylabel('Frequency (Log)')
                    ax_right.grid(True, alpha=0.3)
                    
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                pdf.savefig(fig)
                plt.close()

    print(f"\n✅ Reporte final guardado en: {output_pdf}")

if __name__ == '__main__':
    main()

    