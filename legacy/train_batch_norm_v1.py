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
import random

# --- CONFIGURACIÓN ---
SAVE_MODELS = True
SAVE_PLOTS = True
SEEDS = [42] 
BINS = 30 
EPOCHS = 50 # Con Batch Norm converge más rápido, 50 es suficiente
BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = "results_cnn_batchnorm"

if SAVE_MODELS or SAVE_PLOTS:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- DATOS: CIFAR-10 ---
print(">>> Cargando CIFAR-10...")
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

tensor_x_train = torch.tensor(train_images, dtype=torch.float).permute(0, 3, 1, 2)
tensor_y_train = torch.tensor(train_labels, dtype=torch.long).squeeze()

tensor_x_test = torch.tensor(test_images, dtype=torch.float).permute(0, 3, 1, 2)
tensor_y_test = torch.tensor(test_labels, dtype=torch.long).squeeze()

# Dataset de Test estándar
test_dataset = TensorDataset(tensor_x_test, tensor_y_test)
test_dataloader = DataLoader(test_dataset, batch_size=1000)

# --- DATA AUGMENTATION ---
class CustomAugmentedDataset(Dataset):
    def __init__(self, x_tensor, y_tensor, mode='None'):
        self.x = x_tensor
        self.y = y_tensor
        self.mode = mode
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        img = self.x[idx]
        label = self.y[idx]
        
        if self.mode == 'H-Flip':
            if random.random() < 0.5: img = torch.flip(img, [2])
        
        elif self.mode == 'Shift+Flip':
            if random.random() < 0.5: img = torch.flip(img, [2])
            pad = 4
            img = F.pad(img, (pad, pad, pad, pad), mode='reflect')
            h, w = 32, 32
            H_full, W_full = img.shape[1], img.shape[2]
            top = random.randint(0, H_full - h)
            left = random.randint(0, W_full - w)
            img = img[:, top:top+h, left:left+w]

        return img, label

# --- MODELO CONVNET + BATCH NORM (ESTABILIZADO) ---
class ConvNetBN(nn.Module):
    def __init__(self, dropout_p=0.0, hidden_size=64, **kwargs):
        super(ConvNetBN, self).__init__()
        self.p = dropout_p
        
        # Bloque 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # BN añadido
        
        # Bloque 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) # BN añadido
        
        # Bloque 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) # BN añadido
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout2d = nn.Dropout2d(p=dropout_p)
        
        self.flatten_size = 128 * 4 * 4
        
        self.fc1 = nn.Linear(self.flatten_size, hidden_size)
        self.dropout1d = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden_size, 10)

    def hidden_layer(self, x):
        x = x.view(-1, 3, 32, 32)
        
        # Conv -> BN -> ReLU -> Dropout -> Pool
        x = self.pool(self.dropout2d(F.relu(self.bn1(self.conv1(x)))))
        x = self.pool(self.dropout2d(F.relu(self.bn2(self.conv2(x)))))
        x = self.pool(self.dropout2d(F.relu(self.bn3(self.conv3(x)))))
        
        x = x.view(-1, self.flatten_size)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        x = self.hidden_layer(x)
        if self.p > 0: x = self.dropout1d(x)
        x = self.fc2(x)
        return x

# --- ENTRENAMIENTO ---
def train_network(net, reg_type, reg_val, train_loader, nepochs=EPOCHS, lr=0.01):
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

            if reg_type in ['L1', 'L2']:
                reg_loss = 0
                for module in net.modules():
                    # Aplicar L1/L2 a Conv y Linear (Pesos)
                    if isinstance(module, (nn.Linear, nn.Conv2d)):
                        if reg_type == 'L1':
                            reg_loss += torch.norm(module.weight, p=1)
                        elif reg_type == 'L2':
                            reg_loss += torch.norm(module.weight, p=2)
                loss += reg_val * reg_loss
            
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

# --- PLOTTING ---
def plot_sweep_results(df, title_suffix=''):
    if df.empty: return

    baseline_df = df[df['reg_type'] == 'Baseline']
    base_states_train = baseline_df['pct_states_train'].mean() if not baseline_df.empty else 0
    base_states_val   = baseline_df['pct_states_val'].mean() if not baseline_df.empty else 0
    base_train_acc    = baseline_df['train_acc'].mean() if not baseline_df.empty else 0
    base_val_acc      = baseline_df['val_acc'].mean() if not baseline_df.empty else 0
    
    methods = [m for m in df['reg_type'].unique() if m != 'Baseline']
    
    # FIG 1: Internal States %
    fig1, axes1 = plt.subplots(1, len(methods), figsize=(6*len(methods), 5))
    if len(methods) == 1: axes1 = [axes1]
    
    for i, method in enumerate(methods):
        sub_df = df[df['reg_type'] == method]
        if sub_df.empty: continue
        ax = axes1[i]
        
        x_col = 'aug_name' if method == 'DataAug' else 'reg_val'
        
        sns.lineplot(data=sub_df, x=x_col, y='pct_states_train', label='Train States %', marker='o', color='blue', ax=ax)
        sns.lineplot(data=sub_df, x=x_col, y='pct_states_val', label='Val States %', marker='x', color='orange', linestyle='--', ax=ax)

        ax.axhline(y=base_states_train, color='blue', linestyle=':', alpha=0.5, label='Base Train')
        ax.axhline(y=base_states_val, color='orange', linestyle=':', alpha=0.5, label='Base Val')
        
        ax.set_title(f'States (%) vs {method}')
        if method in ['L1', 'L2']: ax.set_xscale('log')
        ax.set_ylim(-5, 105)
        ax.legend()
        if method == 'DataAug': ax.tick_params(axis='x', rotation=45)

    if SAVE_PLOTS: fig1.savefig(f'{OUTPUT_DIR}/states_pct_{title_suffix}.png')
    
    # FIG 2: Accuracy
    fig2, axes2 = plt.subplots(1, len(methods), figsize=(6*len(methods), 5), sharey=True)
    if len(methods) == 1: axes2 = [axes2]
    
    for i, method in enumerate(methods):
        sub_df = df[df['reg_type'] == method]
        if sub_df.empty: continue
        ax = axes2[i]
        
        x_col = 'aug_name' if method == 'DataAug' else 'reg_val'

        sns.lineplot(data=sub_df, x=x_col, y='train_acc', label='Train Acc', marker='o', color='blue', ax=ax)
        sns.lineplot(data=sub_df, x=x_col, y='val_acc', label='Val Acc', marker='o', color='orange', ax=ax)
        
        ax.axhline(y=base_train_acc, color='blue', linestyle='--', alpha=0.5, label='Base Train')
        ax.axhline(y=base_val_acc, color='orange', linestyle='--', alpha=0.5, label='Base Val')
        
        ax.set_title(f'Accuracy vs {method}')
        if method in ['L1', 'L2']: ax.set_xscale('log')
        ax.legend()
        if method == 'DataAug': ax.tick_params(axis='x', rotation=45)

    if SAVE_PLOTS: fig2.savefig(f'{OUTPUT_DIR}/accuracy_{title_suffix}.png')
    plt.show()

def plot_amplitude_grid_unified(nets_dict, param_lists, input_tensor, dataset_name, title_suffix='', seed_to_plot=42):
    all_methods = list(param_lists.keys())
    n_rows = len(all_methods)
    n_cols = max(len(v) for v in param_lists.values())
    
    total_samples = len(input_tensor)
    subset_size = min(200, total_samples)
    subset_idx = np.random.choice(total_samples, subset_size, replace=False)
    x_sample = input_tensor[subset_idx].to(DEVICE)
    
    fig_p, axes_p = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 3.5*n_rows), sharex=True, sharey=True)
    fig_m, axes_m = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 3.5*n_rows), sharex=True, sharey=True)

    if n_rows == 1: axes_p = [axes_p]; axes_m = [axes_m]
    if n_cols == 1: axes_p = [[ax] for ax in axes_p]; axes_m = [[ax] for ax in axes_m]

    for i, method in enumerate(all_methods):
        params = param_lists[method]
        for j in range(n_cols):
            ax_p = axes_p[i][j]
            ax_m = axes_m[i][j]
            
            if j >= len(params):
                ax_p.set_axis_off(); ax_m.set_axis_off()
                continue
                
            val_info = params[j]
            val = val_info[0] if isinstance(val_info, tuple) else val_info
            label = val_info[1] if isinstance(val_info, tuple) else f"{val:.2g}"

            if val in nets_dict[method] and seed_to_plot in nets_dict[method][val]:
                net = nets_dict[method][val][seed_to_plot]
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
                
                x_axis = np.arange(len(median_curve))
                
                ax_p.fill_between(x_axis, p10, p90, color='C1', alpha=0.3)
                ax_p.plot(x_axis, median_curve, color='C1', label='Median')
                ax_p.set_title(f'{method} {label}\n$\\overline{{P_{{90}} - P_{{10}}}} = {amp_perc:.2f}$')
                
                ax_m.fill_between(x_axis, mean_curve - std_curve, mean_curve + std_curve, color='C2', alpha=0.3)
                ax_m.plot(x_axis, mean_curve, color='C2', label='Mean')
                ax_m.set_title(f'{method} {label}\n$2\\overline{{\\sigma}} = {amp_std:.2f}$')
            else:
                ax_p.text(0.5, 0.5, 'N/A', ha='center')
                ax_m.text(0.5, 0.5, 'N/A', ha='center')

            if j == 0:
                ax_p.set_ylabel(f'{method}\nAct (Z)')
                ax_m.set_ylabel(f'{method}\nAct (Z)')

    fig_p.suptitle(f'{dataset_name}: Amplitudes (Percentiles) - {title_suffix}', fontsize=16)
    fig_m.suptitle(f'{dataset_name}: Amplitudes (Mean $\pm$ Std) - {title_suffix}', fontsize=16)
    plt.tight_layout()
    
    if SAVE_PLOTS:
        fig_p.savefig(f'{OUTPUT_DIR}/amp_perc_{dataset_name}_{title_suffix}.png')
        fig_m.savefig(f'{OUTPUT_DIR}/amp_mean_{dataset_name}_{title_suffix}.png')
    plt.show()

# --- EJECUCIÓN ---
def run_experiment(train_tensor, train_labels, title_suffix="BatchNorm_Sweep"):
    reg_vals = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    drop_vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    aug_levels = [(0, 'None'), (1, 'H-Flip'), (2, 'Shift+Flip')]
    
    param_lists = {'L1': reg_vals, 'L2': reg_vals, 'Dropout': drop_vals, 'DataAug': aug_levels}
    
    results = []
    nets_storage = {m: {} for m in param_lists}
    for m in param_lists:
        for v in param_lists[m]:
            key = v[0] if isinstance(v, tuple) else v
            nets_storage[m][key] = {}
    nets_storage['Baseline'] = {0.0: {}}
    
    print(f"Training Baseline (ConvNetBN) & Computing Bins...")
    bin_edges = None
    n_train = len(train_tensor)
    n_val = len(tensor_x_test)
    
    train_eval_ds = TensorDataset(train_tensor, train_labels)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=1000, shuffle=False)

    # 1. BASELINE
    for seed in SEEDS:
        torch.manual_seed(seed)
        train_loader = DataLoader(train_eval_ds, batch_size=BATCH_SIZE, shuffle=True)
        net = ConvNetBN(dropout_p=0.0) # Usamos ConvNetBN
        net = train_network(net, 'Baseline', 0.0, train_loader)
        
        if bin_edges is None:
            net.eval()
            with torch.no_grad():
                base_acts = net.hidden_layer(tensor_x_train.to(DEVICE)).cpu().numpy()
            bin_edges = compute_simple_bin_edges(base_acts, num_bins=BINS)
            
        acc = evaluate_accuracy(net, test_dataloader)
        train_acc = evaluate_accuracy(net, train_eval_loader)
        st_t = get_num_internal_states_new(net, bin_edges, train_tensor)
        st_v = get_num_internal_states_new(net, bin_edges, tensor_x_test)
        
        results.append({'reg_type': 'Baseline', 'reg_val': 0.0, 'aug_name': 'None', 'seed': seed,
                        'val_acc': acc, 'train_acc': train_acc, 
                        'pct_states_train': (st_t/n_train)*100, 'pct_states_val': (st_v/n_val)*100})
        nets_storage['Baseline'][0.0][seed] = net

    # 2. BARRIDOS REGULARES
    for method in ['L1', 'L2', 'Dropout']:
        vals = param_lists[method]
        print(f"Training {method}...")
        for v in vals:
            for seed in SEEDS:
                torch.manual_seed(seed)
                train_loader = DataLoader(train_eval_ds, batch_size=BATCH_SIZE, shuffle=True)
                
                if method == 'Dropout':
                    net = ConvNetBN(dropout_p=v) # ConvNetBN
                    r_type, r_val = 'Dropout', 0.0
                else:
                    net = ConvNetBN(dropout_p=0.0)
                    r_type, r_val = method, v
                
                net = train_network(net, r_type, r_val, train_loader)
                acc = evaluate_accuracy(net, test_dataloader)
                train_acc = evaluate_accuracy(net, train_eval_loader)
                st_t = get_num_internal_states_new(net, bin_edges, train_tensor)
                st_v = get_num_internal_states_new(net, bin_edges, tensor_x_test)
                
                results.append({'reg_type': method, 'reg_val': v, 'aug_name': 'None', 'seed': seed,
                                'val_acc': acc, 'train_acc': train_acc,
                                'pct_states_train': (st_t/n_train)*100, 'pct_states_val': (st_v/n_val)*100})
                nets_storage[method][v][seed] = net

    # 3. DATA AUG
    method = 'DataAug'
    vals = param_lists[method]
    print(f"Training {method}...")
    for (aug_level, aug_name) in vals:
        if aug_level == 0: continue
        for seed in SEEDS:
            torch.manual_seed(seed)
            aug_ds = CustomAugmentedDataset(train_tensor, train_labels, mode=aug_name)
            train_loader = DataLoader(aug_ds, batch_size=BATCH_SIZE, shuffle=True)
            
            net = ConvNetBN(dropout_p=0.0)
            net = train_network(net, 'DataAug', 0.0, train_loader)
            
            acc = evaluate_accuracy(net, test_dataloader)
            train_acc = evaluate_accuracy(net, train_eval_loader)
            st_t = get_num_internal_states_new(net, bin_edges, train_tensor)
            st_v = get_num_internal_states_new(net, bin_edges, tensor_x_test)
            
            results.append({'reg_type': method, 'reg_val': aug_level, 'aug_name': aug_name, 'seed': seed,
                            'val_acc': acc, 'train_acc': train_acc,
                            'pct_states_train': (st_t/n_train)*100, 'pct_states_val': (st_v/n_val)*100})
            nets_storage[method][aug_level][seed] = net

    # PLOTS
    df = pd.DataFrame(results)
    plot_sweep_results(df, title_suffix)
    print("Plotting TRAIN Amplitudes...")
    plot_amplitude_grid_unified(nets_storage, param_lists, train_tensor, "TRAIN", title_suffix)
    print("Plotting VALIDATION Amplitudes...")
    plot_amplitude_grid_unified(nets_storage, param_lists, tensor_x_test, "VAL", title_suffix)
    return df

# --- RUN ---
print("\n>>> RUN: CIFAR-10 Reduced (15k) with BATCH NORM + FULL REG")
indices = torch.randperm(len(tensor_x_train))[:15000]
x_sub = tensor_x_train[indices]
y_sub = tensor_y_train[indices]

df_res = run_experiment(x_sub, y_sub, "ConvNetBN_15k")