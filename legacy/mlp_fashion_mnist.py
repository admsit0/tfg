import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter
import os

# --- CONFIGURACIÓN ---
SAVE_MODELS = False
SAVE_PLOTS = True
SEEDS = [0,42,67]
BINS = 10        # 30 bins para buena resolución
EPOCHS = 10
BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if SAVE_MODELS or SAVE_PLOTS:
    os.makedirs("results", exist_ok=True)

# --- DATOS ---
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

tensor_x_train = torch.tensor(train_images, dtype=torch.float)
tensor_y_train = torch.tensor(train_labels, dtype=torch.long)
tensor_x_test = torch.tensor(test_images, dtype=torch.float)
tensor_y_test = torch.tensor(test_labels, dtype=torch.long)

full_train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
test_dataset = TensorDataset(tensor_x_test, tensor_y_test)
test_dataloader = DataLoader(test_dataset, batch_size=1000)

# --- MODELO ---
class Net(nn.Module):
    def __init__(self, dropout_p=0.0):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(64, 10)

    def hidden_layer(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- ENTRENAMIENTO ---
def train_network(net, reg_type, reg_val, train_loader, nepochs=EPOCHS, lr=0.01):
    optimizer = optim.SGD(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    net.to(DEVICE)
    net.train()
    
    for it in range(nepochs):
        for x, t in train_loader:
            x, t = x.to(DEVICE), t.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(outputs, t)

            if reg_type == 'L1':
                l1_term = torch.norm(net.fc1.weight, p=1) + torch.norm(net.fc2.weight, p=1)
                loss += reg_val * l1_term
            elif reg_type == 'L2':
                l2_term = torch.norm(net.fc1.weight, p=2) + torch.norm(net.fc2.weight, p=2)
                loss += reg_val * l2_term
            
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

# --- FUNCIONES DE ANÁLISIS DE ESTADOS ---

def compute_robust_bin_edges(activations, num_bins=BINS):
    """
    Calcula bordes de histograma ignorando outliers extremos.
    Usa el rango [min, percentil_99.9] para definir los bins.
    """
    # Aplanamos para ver la distribución global
    vals = activations.flatten()
    
    # Definir rango robusto: desde el mínimo real hasta el 99.9% de los datos
    # Esto evita que un solo valor de 1000 expanda los bins y deje todo lo demás en el bin 0
    v_min = np.min(vals)
    v_max = np.percentile(vals, 99.9) 
    
    # Si la red está muerta (todo 0), forzamos un rango pequeño artificial
    if v_max <= v_min:
        v_max = v_min + 1.0
        
    # Generar bordes lineales en ese rango efectivo
    bin_edges = np.linspace(v_min, v_max, num_bins + 1)
    
    # Asegurar que el último borde capture incluso los outliers superiores (extendiendo a infinito)
    # bin_edges[-1] = np.max(vals) + 0.01 # Opcional: extender al max real
    # Mejor estrategia del notebook: extender un poco el último
    bin_edges[-1] = np.max(vals) + 1e-3
    
    print(f"  -> Bin Edges (Robusto): Min={bin_edges[0]:.3f}, P99.9={v_max:.3f}, MaxReal={np.max(vals):.3f}")
    return bin_edges

_

# --- PLOTTING ---
def plot_sweep_results(df, title_suffix=''):
    if df.empty: return

    baseline_df = df[df['reg_type'] == 'Baseline']
    base_states_mean = baseline_df['num_states'].mean() if not baseline_df.empty else 0
    base_train_acc_mean = baseline_df['train_acc'].mean() if not baseline_df.empty else 0
    base_val_acc_mean = baseline_df['val_acc'].mean() if not baseline_df.empty else 0
    
    methods = ['L1', 'L2', 'Dropout']
    
    # FIG 1: Internal States
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
    for i, method in enumerate(methods):
        sub_df = df[df['reg_type'] == method]
        if sub_df.empty: continue
        ax = axes1[i]
        sns.lineplot(data=sub_df, x='reg_val', y='num_states', marker='o', ax=ax, errorbar='sd', label=method)
        ax.axhline(y=base_states_mean, color='gray', linestyle='--', label='Baseline')
        ax.set_title(f'Internal States vs {method}')
        
        if method in ['L1', 'L2']:
            ax.set_xscale('log')
            ax.set_xlabel(f'{method} $\lambda$ (log)')
        else:
            ax.set_xlabel(f'{method} Probability')
        ax.set_ylabel('Unique States')
        ax.legend()
    fig1.suptitle(f'Internal States Analysis - {title_suffix}', fontsize=16)
    if SAVE_PLOTS: fig1.savefig(f'results/states_sweep_{title_suffix}.png')
    
    # FIG 2: Accuracy
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, method in enumerate(methods):
        sub_df = df[df['reg_type'] == method]
        if sub_df.empty: continue
        ax = axes2[i]
        grp = sub_df.groupby('reg_val')[['train_acc', 'val_acc']].agg(['mean', 'std'])
        x_vals = grp.index
        
        y_t = grp['train_acc']['mean']
        y_t_std = grp['train_acc']['std']
        ax.plot(x_vals, y_t, label='Train', color='blue', marker='o')
        ax.fill_between(x_vals, y_t - y_t_std, y_t + y_t_std, color='blue', alpha=0.2)
        
        y_v = grp['val_acc']['mean']
        y_v_std = grp['val_acc']['std']
        ax.plot(x_vals, y_v, label='Validation', color='orange', marker='o')
        ax.fill_between(x_vals, y_v - y_v_std, y_v + y_v_std, color='orange', alpha=0.2)
        
        ax.axhline(y=base_train_acc_mean, color='blue', linestyle='--', alpha=0.5, label='Base Train')
        ax.axhline(y=base_val_acc_mean, color='orange', linestyle='--', alpha=0.5, label='Base Val')
        
        ax.set_title(f'Accuracy vs {method}')
        if method in ['L1', 'L2']:
            ax.set_xscale('log')
        ax.set_xlabel(f'{method} Param')
        ax.set_ylabel('Accuracy')
        ax.legend()
    fig2.suptitle(f'Accuracy (Train vs Val) - {title_suffix}', fontsize=16)
    if SAVE_PLOTS: fig2.savefig(f'results/accuracy_sweep_{title_suffix}.png')
    plt.show()

def plot_amplitude_grid_unified(nets_dict, param_lists, title_suffix='', seed_to_plot=0):
    methods = ['L1', 'L2', 'Dropout']
    n_cols = len(param_lists['L1']) 
    n_rows = 3
    
    subset_idx = np.random.choice(len(tensor_x_train), 200, replace=False)
    x_sample = tensor_x_train[subset_idx].to(DEVICE)
    
    fig_p, axes_p = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 3*n_rows), sharex=True, sharey=True)
    fig_m, axes_m = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 3*n_rows), sharex=True, sharey=True)

    for i, method in enumerate(methods):
        params = param_lists[method]
        for j in range(n_cols):
            if j >= len(params): continue
            val = params[j]
            ax_p = axes_p[i][j]
            ax_m = axes_m[i][j]
            
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
                
                # Plot Percentiles
                ax_p.fill_between(x_axis, p10_sg, p90_sg, color='C1', alpha=0.3)
                ax_p.plot(x_axis, median_sg, color='C1', label='Median')
                ax_p.set_title(f'{method} {val:.2g}\n$\\overline{{P_{{90}} - P_{{10}}}} = {amp_perc:.2f}$')
                
                # Plot Mean/Std
                ax_m.fill_between(x_axis, mean_sg - std_sg, mean_sg + std_sg, color='C2', alpha=0.3)
                ax_m.plot(x_axis, mean_sg, color='C2', label='Mean')
                ax_m.set_title(f'{method} {val:.2g}\n$2\\overline{{\\sigma}} = {amp_std:.2f}$')
            else:
                ax_p.text(0.5, 0.5, 'N/A', ha='center')
                ax_m.text(0.5, 0.5, 'N/A', ha='center')
            
            if j == 0:
                ax_p.set_ylabel(f'{method}\nAct (Z)')
                ax_m.set_ylabel(f'{method}\nAct (Z)')
            if i == n_rows - 1:
                ax_p.set_xlabel('Sample Idx')
                ax_m.set_xlabel('Sample Idx')

    fig_p.suptitle(f'Amplitudes (Percentiles) - {title_suffix}', fontsize=16)
    fig_m.suptitle(f'Amplitudes (Mean $\pm$ Std) - {title_suffix}', fontsize=16)
    plt.tight_layout()
    
    if SAVE_PLOTS:
        fig_p.savefig(f'results/amp_perc_unified_{title_suffix}.png')
        fig_m.savefig(f'results/amp_mean_unified_{title_suffix}.png')
    plt.show()

# --- EJECUCIÓN ---
def run_experiment(train_dataset_exp, title_suffix="FullData"):
    train_eval_loader = DataLoader(train_dataset_exp, batch_size=1000, shuffle=False)
    train_loader = DataLoader(train_dataset_exp, batch_size=BATCH_SIZE, shuffle=True)
    
    # 4 valores para L1/L2 y 4 valores para Dropout (Grid cuadrada)
    reg_vals_log = [1e-4, 1e-3, 1e-2, 1e-1, 5e-1, 1.0, 2.0]
    dropout_vals = [0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 1.0]
    
    param_lists = {
        'L1': reg_vals_log,
        'L2': reg_vals_log,
        'Dropout': dropout_vals
    }
    
    results = []
    nets_storage = {'L1': {}, 'L2': {}, 'Dropout': {}, 'Baseline': {0.0: {}}}
    for m in param_lists:
        for v in param_lists[m]:
            nets_storage[m][v] = {}

    print("Training Baseline & Computing ROBUST Bin Edges...")
    bin_edges = None
    
    for seed in SEEDS:
        torch.manual_seed(seed)
        net = Net(dropout_p=0.0)
        net = train_network(net, 'Baseline', 0.0, train_loader)
        
        # 1. Calcular Bins ROBUSTOS (una sola vez)
        if bin_edges is None:
            net.eval()
            with torch.no_grad():
                base_activations = net.hidden_layer(tensor_x_train.to(DEVICE)).cpu().numpy()
            bin_edges = compute_robust_bin_edges(base_activations, num_bins=BINS)

        # Evaluar
        states = get_num_internal_states_new(net, bin_edges)
        acc = evaluate_accuracy(net, test_dataloader)
        train_acc = evaluate_accuracy(net, train_eval_loader)
        
        results.append({'reg_type': 'Baseline', 'reg_val': 0.0, 'seed': seed, 
                        'val_acc': acc, 'train_acc': train_acc, 'num_states': states})
        nets_storage['Baseline'][0.0][seed] = net
        if SAVE_MODELS: torch.save(net.state_dict(), f'results/model_Base_s{seed}_{title_suffix}.pth')

    # 2. Barridos
    for method, p_list in param_lists.items():
        print(f"Training {method}...")
        for val in p_list:
            for seed in SEEDS:
                torch.manual_seed(seed)
                
                if method == 'Dropout':
                    net = Net(dropout_p=val)
                    r_type, r_val = 'Dropout', 0.0
                else:
                    net = Net(dropout_p=0.0)
                    r_type, r_val = method, val
                
                net = train_network(net, r_type, r_val, train_loader)
                acc = evaluate_accuracy(net, test_dataloader)
                train_acc = evaluate_accuracy(net, train_eval_loader)
                
                # Usamos los mismos bin edges robustos
                states = get_num_internal_states_new(net, bin_edges)
                
                results.append({'reg_type': method, 'reg_val': val, 'seed': seed, 
                                'val_acc': acc, 'train_acc': train_acc, 'num_states': states})
                nets_storage[method][val][seed] = net
                if SAVE_MODELS and seed == 0: torch.save(net.state_dict(), f'results/model_{method}_v{val:.3f}_{title_suffix}.pth')

    df = pd.DataFrame(results)
    plot_sweep_results(df, title_suffix)
    plot_amplitude_grid_unified(nets_storage, param_lists, title_suffix)
    return df

# --- RUN ---
print(">>> RUN 1: Full Dataset")
df_full = run_experiment(full_train_dataset, "Full")

print("\n>>> RUN 2: Reduced Dataset (Overfitting)")
indices = torch.randperm(len(tensor_x_train))[:1000]
reduced_train_dataset = TensorDataset(tensor_x_train[indices], tensor_y_train[indices])
df_reduced = run_experiment(reduced_train_dataset, "Reduced_Overfit")