import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter
import os
from torchvision import transforms

# --- CONFIGURACIÓN ---
SAVE_MODELS = True
SAVE_PLOTS = True
SEEDS = [42] # Seed fija
BINS = 30 
EPOCHS = 60 # Epochs suficientes
BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = "results_cnn_dataAug"

if SAVE_MODELS or SAVE_PLOTS:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- DATOS: CIFAR-10 ---
print(">>> Cargando CIFAR-10...")
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalización [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Conversión a PyTorch (N, 3, 32, 32)
tensor_x_train = torch.tensor(train_images, dtype=torch.float).permute(0, 3, 1, 2)
tensor_y_train = torch.tensor(train_labels, dtype=torch.long).squeeze()

tensor_x_test = torch.tensor(test_images, dtype=torch.float).permute(0, 3, 1, 2)
tensor_y_test = torch.tensor(test_labels, dtype=torch.long).squeeze()

# Dataset de Test estándar (sin aumentación)
test_dataset = TensorDataset(tensor_x_test, tensor_y_test)
test_dataloader = DataLoader(test_dataset, batch_size=1000)

# Constantes CIFAR
INPUT_SHAPE = (3, 32, 32)
FLATTEN_DIM = 3 * 32 * 32

# --- CUSTOM DATASET PARA DATA AUGMENTATION ---
class AugmentedDataset(Dataset):
    def __init__(self, x_tensor, y_tensor, transform=None):
        self.x = x_tensor
        self.y = y_tensor
        self.transform = transform
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        img = self.x[idx]
        label = self.y[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

# --- MODELO CONVNET (Igual que Main4) ---
class ConvNet(nn.Module):
    def __init__(self, dropout_p=0.0, hidden_size=64, **kwargs):
        super(ConvNet, self).__init__()
        
        # Bloque 1: 3 -> 32 filtros
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # Bloque 2: 32 -> 64 filtros
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Bloque 3: 64 -> 128 filtros
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # 32 -> 16 -> 8 -> 4
        # 128 canales * 4 * 4
        self.flatten_size = 128 * 4 * 4
        
        self.fc1 = nn.Linear(self.flatten_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout_p) # Solo se usará si dropout_p > 0, aquí será 0
        self.fc2 = nn.Linear(hidden_size, 10)

    def hidden_layer(self, x):
        x = x.view(-1, 3, 32, 32)
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, self.flatten_size)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- ENTRENAMIENTO (Sin L1/L2 loops, solo básico) ---
def train_network(net, train_loader, nepochs=EPOCHS, lr=0.01):
    # Momentum para ayudar en CIFAR
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    net.to(DEVICE)
    net.train()
    
    for it in range(nepochs):
        for x, t in train_loader:
            x, t = x.to(DEVICE), t.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(outputs, t)
            
            # Sin regularización explícita L1/L2 en este script
            
            loss.backward()
            optimizer.step()
    return net

def evaluate_accuracy(net, dataloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, t in dataloader:
            x, t = x.to(DEVICE), t.to(DEVICE)
            outputs = net(x)
            _, predicted = torch.max(outputs.data, 1)
            total += t.size(0)
            correct += (predicted == t).sum().item()
    return correct / total

# --- ANÁLISIS ---
def compute_simple_bin_edges(activations, num_bins=BINS):
    vals = activations.flatten()
    _, bin_edges = np.histogram(vals, bins=num_bins)
    bin_edges = bin_edges.astype(float)
    bin_edges[-1] = bin_edges[-1] + 0.01 * (bin_edges[-1] - bin_edges[-2])
    print(f"  -> Bin Edges (Simple): Min={bin_edges[0]:.3f}, Max={bin_edges[-1]:.3f}")
    return bin_edges

def get_num_internal_states_new(net, bin_edges, input_data):
    net.eval()
    input_data = input_data.to(DEVICE)
    with torch.no_grad():
        activations = net.hidden_layer(input_data).cpu().numpy()

    b = np.zeros_like(activations)
    for i in range(activations.shape[1]):
        b[:, i] = np.digitize(activations[:, i], bin_edges)

    num_states = np.unique(b, axis=0).shape[0]
    return num_states

# --- PLOTTING (Adaptado para DataAug linear x-axis) ---
def plot_sweep_results(df, title_suffix=''):
    if df.empty: return

    # No hay Baseline separado en el DF como antes, el nivel 0 es el baseline
    # Pero mantenemos la lógica de plotear todos los puntos
    
    fig1, axes1 = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot States
    sns.lineplot(data=df, x='aug_level', y='pct_states_train', label='Train States %', marker='o', color='blue', ax=axes1)
    sns.lineplot(data=df, x='aug_level', y='pct_states_val', label='Val States %', marker='o', color='orange', ax=axes1)
    
    # Configurar eje X con nombres de augmentación
    aug_names = df[['aug_level', 'aug_name']].drop_duplicates().sort_values('aug_level')
    plt.xticks(aug_names['aug_level'], aug_names['aug_name'], rotation=45)
    
    axes1.set_title(f'Internal States Usage (%) vs Data Augmentation')
    axes1.set_ylabel('Unique States / Total Samples (%)')
    axes1.set_ylim(-5, 105)
    axes1.grid(True, alpha=0.3)
    
    if SAVE_PLOTS: fig1.savefig(f'{OUTPUT_DIR}/states_pct_sweep_{title_suffix}.png')
    
    fig2, axes2 = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot Accuracy
    sns.lineplot(data=df, x='aug_level', y='train_acc', label='Train Acc', marker='o', color='blue', ax=axes2)
    sns.lineplot(data=df, x='aug_level', y='val_acc', label='Val Acc', marker='o', color='orange', ax=axes2)
    
    plt.xticks(aug_names['aug_level'], aug_names['aug_name'], rotation=45)
    
    axes2.set_title(f'Accuracy vs Data Augmentation')
    axes2.set_ylabel('Accuracy')
    axes2.grid(True, alpha=0.3)
    
    if SAVE_PLOTS: fig2.savefig(f'{OUTPUT_DIR}/accuracy_sweep_{title_suffix}.png')
    plt.show()

def plot_amplitude_grid_unified(nets_dict, aug_levels, input_tensor, dataset_name, title_suffix='', seed_to_plot=42):
    n_cols = len(aug_levels)
    n_rows = 1 # Solo 1 fila (DataAug)
    
    total_samples = len(input_tensor)
    subset_size = min(200, total_samples)
    subset_idx = np.random.choice(total_samples, subset_size, replace=False)
    x_sample = input_tensor[subset_idx].to(DEVICE)
    
    fig_p, axes_p = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 4), sharex=True, sharey=True)
    fig_m, axes_m = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 4), sharex=True, sharey=True)

    # Si hay una sola fila, axes es array 1D
    if n_cols == 1:
        axes_p = [axes_p]; axes_m = [axes_m]

    for j, (level_idx, level_name) in enumerate(aug_levels):
        ax_p = axes_p[j]
        ax_m = axes_m[j]
        
        if level_idx in nets_dict and seed_to_plot in nets_dict[level_idx]:
            net = nets_dict[level_idx][seed_to_plot]
            net.eval()
            with torch.no_grad():
                H = net.hidden_layer(x_sample).cpu().numpy()
            
            mu = H.mean(axis=0, keepdims=True)
            sd = H.std(axis=0, keepdims=True)
            sd[sd == 0] = 1.0
            H_norm = (H - mu) / sd
            
            median_curve = np.median(H_norm, axis=1)
            p10 = np.percentile(H_norm, 10, axis=1)
            p90 = np.percentile(H_norm, 90, axis=1)
            mean_curve = np.mean(H_norm, axis=1)
            std_curve = np.std(H_norm, axis=1)
            
            amp_perc = np.mean(p90 - p10)
            amp_std = np.mean(2 * std_curve)
            
            window = 15
            if len(median_curve) > window:
                median_sg = savgol_filter(median_curve, window, 2)
                p10_sg = savgol_filter(p10, window, 2)
                p90_sg = savgol_filter(p90, window, 2)
                mean_sg = savgol_filter(mean_curve, window, 2)
                std_sg = savgol_filter(std_curve, window, 2)
            else:
                median_sg, p10_sg, p90_sg = median_curve, p10, p90
                mean_sg, std_sg = mean_curve, std_curve
            
            x_axis = np.arange(len(median_sg))
            
            ax_p.fill_between(x_axis, p10_sg, p90_sg, color='C1', alpha=0.3)
            ax_p.plot(x_axis, median_sg, color='C1', label='Median')
            ax_p.set_title(f'{level_name}\n$\\overline{{P_{{90}} - P_{{10}}}} = {amp_perc:.2f}$')
            
            ax_m.fill_between(x_axis, mean_sg - std_sg, mean_sg + std_sg, color='C2', alpha=0.3)
            ax_m.plot(x_axis, mean_sg, color='C2', label='Mean')
            ax_m.set_title(f'{level_name}\n$2\\overline{{\\sigma}} = {amp_std:.2f}$')
        else:
            ax_p.text(0.5, 0.5, 'N/A', ha='center')
            ax_m.text(0.5, 0.5, 'N/A', ha='center')
        
        if j == 0:
            ax_p.set_ylabel(f'Act (Z)')
            ax_m.set_ylabel(f'Act (Z)')
        
        ax_p.set_xlabel('Sample Idx')
        ax_m.set_xlabel('Sample Idx')

    fig_p.suptitle(f'{dataset_name}: Amplitudes (Percentiles) - {title_suffix}', fontsize=16)
    fig_m.suptitle(f'{dataset_name}: Amplitudes (Mean $\pm$ Std) - {title_suffix}', fontsize=16)
    plt.tight_layout()
    
    if SAVE_PLOTS:
        fig_p.savefig(f'{OUTPUT_DIR}/amp_perc_{dataset_name}_{title_suffix}.png')
        fig_m.savefig(f'{OUTPUT_DIR}/amp_mean_{dataset_name}_{title_suffix}.png')
    plt.show()

# --- EJECUCIÓN PRINCIPAL ---
def run_experiment(train_tensor, train_labels, title_suffix="DataAug_Sweep", hidden_size=128):
    # Definir Niveles de Data Augmentation
    # Lista de tuplas: (Nivel Numérico, Nombre, Transform)
    
    aug_configs = [
        (0, "None", None),
        
        (1, "H-Flip", transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0) # Forzar flip para ver efecto
        ])),
        
        (2, "Standard", transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4)
        ])),
        
        (3, "Strong", transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(15)
        ])),
        
        (4, "Extreme", transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ]))
    ]
    
    results = []
    nets_storage = {} # Key: aug_level
    
    # 1. BASELINE & BINS (Usamos Level 0 'None' para calcular los bins de referencia)
    print(f"Training Baseline (None) & Computing Bins...")
    bin_edges = None
    
    # Preparamos datasets
    # IMPORTANTE: Usamos el subset de tensores pasado como argumento
    n_train_samples = len(train_tensor)
    n_val_samples = len(tensor_x_test)
    
    # Eval loader (sin augmentación, para medir accuracy real en train set limpio)
    train_eval_ds = TensorDataset(train_tensor, train_labels)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=1000, shuffle=False)
    
    # Loop principal sobre configs
    for (aug_level, aug_name, transform) in aug_configs:
        print(f"Training with Data Augmentation: {aug_name} (Level {aug_level})...")
        nets_storage[aug_level] = {}
        
        for seed in SEEDS:
            torch.manual_seed(seed)
            
            # Crear Loader con Augmentación
            # Nota: augmented dataset aplica transform al vuelo
            aug_ds = AugmentedDataset(train_tensor, train_labels, transform=transform)
            train_loader = DataLoader(aug_ds, batch_size=BATCH_SIZE, shuffle=True)
            
            # Crear red limpia
            net = ConvNet(dropout_p=0.0, hidden_size=hidden_size)
            
            # Entrenar
            net = train_network(net, train_loader, nepochs=EPOCHS)
            
            # Calcular Bins (SOLO EN LA PRIMERA ITERACIÓN DEL BASELINE)
            if bin_edges is None:
                net.eval()
                with torch.no_grad():
                    # Calculamos bins sobre todo el set global de entrenamiento sin augmentación
                    base_activations = net.hidden_layer(tensor_x_train.to(DEVICE)).cpu().numpy()
                bin_edges = compute_simple_bin_edges(base_activations, num_bins=BINS)
            
            # Métricas
            acc = evaluate_accuracy(net, test_dataloader)
            train_acc = evaluate_accuracy(net, train_eval_loader) # Acc en train limpio
            
            states_train = get_num_internal_states_new(net, bin_edges, train_tensor)
            states_val   = get_num_internal_states_new(net, bin_edges, tensor_x_test)
            
            pct_train = (states_train / n_train_samples) * 100
            pct_val   = (states_val / n_val_samples) * 100
            
            results.append({
                'reg_type': 'DataAug', 
                'aug_level': aug_level,
                'aug_name': aug_name,
                'seed': seed,
                'val_acc': acc, 
                'train_acc': train_acc, 
                'pct_states_train': pct_train, 
                'pct_states_val': pct_val
            })
            nets_storage[aug_level][seed] = net
            
            if SAVE_MODELS and seed == SEEDS[0]:
                torch.save(net.state_dict(), f'{OUTPUT_DIR}/model_DataAug_{aug_name}.pth')

    # Guardar resultados
    df = pd.DataFrame(results)
    plot_sweep_results(df, title_suffix)
    
    # List of (level, name) for plotting grid
    aug_levels_list = [(x[0], x[1]) for x in aug_configs]
    
    print("Plotting TRAIN Amplitudes...")
    plot_amplitude_grid_unified(nets_storage, aug_levels_list, train_tensor, "TRAIN", title_suffix)
    
    print("Plotting VALIDATION Amplitudes...")
    plot_amplitude_grid_unified(nets_storage, aug_levels_list, tensor_x_test, "VAL", title_suffix)
    
    return df

# --- RUN ---
print(f"\n>>> RUN DATA AUGMENTATION: Reduced Dataset (15k, ConvNet)")
# Reducimos datos
indices = torch.randperm(len(tensor_x_train))[:15000]
x_sub = tensor_x_train[indices]
y_sub = tensor_y_train[indices]

df_aug = run_experiment(x_sub, y_sub, "DataAug_Sweep", hidden_size=128)