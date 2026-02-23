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
# CLASES DE LA RED (Debe coincidir con tu train)
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
    def __init__(self, reg_method='', reg_val=0.0):
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
        self.fc2 = nn.Linear(128, 10)

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
def get_test_loader():
    test_set = datasets.CIFAR10(root='./data_raw', train=False, download=True)
    x_test = test_set.data.astype('float32') / 255.0
    y_test = np.array(test_set.targets)
    tensor_x_test = torch.tensor(x_test).permute(0, 3, 1, 2)
    tensor_y_test = torch.tensor(y_test, dtype=torch.long)
    return DataLoader(TensorDataset(tensor_x_test, tensor_y_test), batch_size=1000)

def extract_model_info(filename):
    pattern = r"CNN_(.*?)_([0-9\.]+)_ep(\d+)\.pth"
    match = re.search(pattern, filename)
    if match:
        return match.group(1), float(match.group(2)), int(match.group(3))
    return None, None, None

def get_best_epochs_from_csv(csv_path):
    """Lee el CSV y devuelve un diccionario con la MEJOR época para cada reg_val basada en val_acc"""
    best_epochs = {}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Agrupar por reg_val y buscar el índice del máximo val_acc
        idx_max = df.groupby('reg_val')['val_acc'].idxmax()
        best_rows = df.loc[idx_max]
        for _, row in best_rows.iterrows():
            # Redondeamos a 6 decimales para evitar problemas de coma flotante al buscar en diccionarios
            best_epochs[round(row['reg_val'], 6)] = int(row['epoch'])
    return best_epochs

# =========================================================
# MAIN
# =========================================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs_moreMethods')
    models_dir = os.path.join(base_dir, 'models')
    data_dir = os.path.join(base_dir, 'data')
    output_pdf = os.path.join(base_dir, 'activation_histograms_report.pdf')
    
    print("⬇️ Cargando Test Set...")
    test_loader = get_test_loader()
    
    print("🔍 Escaneando modelos y buscando las MEJORES épocas en los CSVs...")
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    # 1. Catalogar todos los modelos disponibles: available[method][round(val)][epoch] = filename
    available_models = {}
    for f in model_files:
        method, val, epoch = extract_model_info(f)
        if method:
            val_rounded = round(val, 6)
            if method not in available_models:
                available_models[method] = {}
            if val_rounded not in available_models[method]:
                available_models[method][val_rounded] = {}
            available_models[method][val_rounded][epoch] = f

    # 2. Seleccionar el mejor archivo apoyándonos en los CSV
    best_models_to_process = {} # method -> list of (val, filename)
    
    for method, vals_dict in available_models.items():
        csv_path = os.path.join(data_dir, f"data_CNN_{method}.csv")
        best_epochs = get_best_epochs_from_csv(csv_path)
        
        best_models_to_process[method] = []
        for val_rounded, epochs_dict in vals_dict.items():
            # Intentar coger la mejor época según el CSV
            target_epoch = best_epochs.get(val_rounded)
            
            # Si por algún motivo no está en el CSV o no se guardó el .pth de esa época, cogemos la última guardada
            if target_epoch is None or target_epoch not in epochs_dict:
                target_epoch = max(epochs_dict.keys())
                print(f"  ⚠️ Para {method}={val_rounded}, mejor época no encontrada. Usando última: {target_epoch}")
            
            best_filename = epochs_dict[target_epoch]
            best_models_to_process[method].append((val_rounded, best_filename, target_epoch))
            
        # Ordenar por valor de regularización para que el PDF quede ordenado
        best_models_to_process[method].sort(key=lambda x: x[0])

    print("📊 Extrayendo activaciones, calculando Sparsity y generando PDF...")
    with PdfPages(output_pdf) as pdf:
        for method, items in best_models_to_process.items():
            print(f"\n--- Procesando Método: {method} ---")
            
            # Crear figura con subplots
            fig, axes = plt.subplots(len(items), 1, figsize=(8, 4 * len(items)))
            if len(items) == 1: axes = [axes]
                
            fig.suptitle(f'fc1 Activations - {method} (Best Epochs)', fontsize=16)
            
            for idx, (val, filename, epoch) in enumerate(items):
                filepath = os.path.join(models_dir, filename)
                
                # Instanciar y cargar pesos
                net = ConvNet(reg_method=method, reg_val=val).to(device)
                net.load_state_dict(torch.load(filepath, map_location=device))
                net.eval() # Fundamental: Apaga el dropout de test para ver el estado real asimilado
                
                activations = []
                # Hook para capturar la salida de fc1
                def hook_fn(m, i, o):
                    activations.append(o.detach().cpu().numpy())
                    
                handle = net.fc1.register_forward_hook(hook_fn)
                
                # Pasar todo el test set
                with torch.no_grad():
                    for x, _ in test_loader:
                        net(x.to(device))
                        
                handle.remove()
                
                # Procesar activaciones
                acts_raw = np.concatenate(activations).flatten()
                
                # CALCULAR SPARSITY (Neuronas muertas == 0.0 por culpa del ReLU)
                sparsity_pct = np.mean(acts_raw <= 0) * 100
                print(f"  {method}={val} (Ep {epoch}) -> Sparsity: {sparsity_pct:.2f}%")
                
                # Filtrar ceros para el histograma (para ver la forma de las que sí están vivas)
                acts_alive = acts_raw[acts_raw > 0] 
                
                # Dibujar
                ax = axes[idx]
                ax.hist(acts_alive, bins=100, color='indigo', alpha=0.7)
                ax.set_title(f'{method} = {val} | Best Ep: {epoch} | Sparsity: {sparsity_pct:.1f}%')
                ax.set_yscale('log') # Escala logarítmica para ver la cola de la distribución
                ax.set_xlabel('Activation Value')
                ax.set_ylabel('Frequency (Log)')
                ax.grid(True, alpha=0.3)
                
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            pdf.savefig(fig)
            plt.close()

    print(f"\n✅ Reporte final guardado en: {output_pdf}")

if __name__ == '__main__':
    main()
    