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
SAVE_MODELS = True
SAVE_PLOTS = True
SEEDS = [0, 42, 67]
BINS = 30 
EPOCHS = 15 # Un valor intermedio razonable
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

# --- MODELO ORIGINAL (MLP) ---
class Net(nn.Module):
    def __init__(self, dropout_p=0.0, hidden_size=64, **kwargs):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_size, 10)

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

# --- MODELO MULTICAPA (Deep MLP) ---
class MultiLayerNet(nn.Module):
    def __init__(self, dropout_p=0.0, hidden_size=64, num_layers=4, **kwargs):
        super(MultiLayerNet, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_p)
        
        self.layers.append(nn.Linear(28*28, hidden_size))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            
        self.out_layer = nn.Linear(hidden_size, 10)

    def hidden_layer(self, x):
        x = x.view(-1, 28*28)
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.dropout(x)
        x = self.out_layer(x)
        return x

# --- NUEVO MODELO CONVOLUCIONAL (ConvNet) ---
class ConvNet(nn.Module):
    def __init__(self, dropout_p=0.0, hidden_size=64, **kwargs):
        super(ConvNet, self).__init__()
        # Entrada: 1 canal (gris), 28x28
        
        # Conv 1: 32 filtros
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Conv 2: 64 filtros
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Cálculo de tamaño aplanado:
        # 28x28 -> Conv1 -> 28x28 -> Pool -> 14x14
        # 14x14 -> Conv2 -> 14x14 -> Pool -> 7x7
        # Salida: 64 canales * 7 * 7
        self.flatten_size = 64 * 7 * 7
        
        # Capa densa "plana" oculta (la que mediremos)
        self.fc1 = nn.Linear(self.flatten_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_size, 10) # Salida (Softmax implícita en Loss)

    def hidden_layer(self, x):
        """
        Devuelve las activaciones de la última capa oculta densa (fc1) antes de la salida.
        """
        # Asegurar forma (Batch, 1, 28, 28)
        x = x.view(-1, 1, 28, 28)
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(-1, self.flatten_size) # Flatten
        x = F.relu(self.fc1(x))           # Activaciones FC1
        return x

    def forward(self, x):
        x = self.hidden_layer(x) # Obtener FC1 activada
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- ENTRENAMIENTO (ACTUALIZADO: Soporte para Conv2d) ---
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

            # Regularización (ahora incluye Conv2d)
            if reg_type in ['L1', 'L2']:
                reg_loss = 0
                for module in net.modules():
                    # Aplicamos reg a Linear y Conv2d
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

# --- ANÁLISIS DE ESTADOS ---

def compute_simple_bin_edges(activations, num_bins=BINS):
    vals = activations.flatten()
    _, bin_edges = np.histogram(vals, bins=num_bins)
    bin_edges = bin_edges.astype(float)
    bin_edges[-1] = bin_edges[-1] + 0.01 * (bin_edges[-1] - bin_edges[-2])
    print(f"  -> Bin Edges: Min={bin_edges[0]:.3f}, Max={bin_edges[-1]:.3f}")
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
    
    methods = ['L1', 'L2', 'Dropout']
    
    # FIG 1: Internal States %
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
    for i, method in enumerate(methods):
        sub_df = df[df['reg_type'] == method]
        if sub_df.empty: continue
        ax = axes1[i]
        
        grp = sub_df.groupby('reg_val')[['pct_states_train', 'pct_states_val']].agg(['mean', 'std'])
        x_vals = grp.index
        
        y_t = grp['pct_states_train']['mean']
        y_t_std = grp['pct_states_train']['std']
        ax.plot(x_vals, y_t, label='Train States %', color='blue', marker='o')
        ax.fill_between(x_vals, y_t - y_t_std, y_t + y_t_std, color='blue', alpha=0.1)
        
        y_v = grp['pct_states_val']['mean']
        y_v_std = grp['pct_states_val']['std']
        ax.plot(x_vals, y_v, label='Val States %', color='orange', marker='x', linestyle='--')
        ax.fill_between(x_vals, y_v - y_v_std, y_v + y_v_std, color='orange', alpha=0.1)

        ax.axhline(y=base_states_train, color='blue', linestyle=':', alpha=0.5, label='Base Train %')
        ax.axhline(y=base_states_val, color='orange', linestyle=':', alpha=0.5, label='Base Val %')
        
        ax.set_title(f'Internal States (%) vs {method}')
        if method in ['L1', 'L2']:
            ax.set_xscale('log')
            ax.set_xlabel(f'{method} $\lambda$ (log)')
        else:
            ax.set_xlabel(f'{method} Probability')
        
        ax.set_ylabel('Unique States / Total Samples (%)')
        ax.set_ylim(-5, 105)
        ax.legend()
        
    fig1.suptitle(f'Internal States Usage (%) - {title_suffix}', fontsize=16)
    if SAVE_PLOTS: fig1.savefig(f'results/states_pct_sweep_{title_suffix}.png')
    
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
        ax.plot(x_vals, y_t, label='Train Acc', color='blue', marker='o')
        ax.fill_between(x_vals, y_t - y_t_std, y_t + y_t_std, color='blue', alpha=0.2)
        
        y_v = grp['val_acc']['mean']
        y_v_std = grp['val_acc']['std']
        ax.plot(x_vals, y_v, label='Val Acc', color='orange', marker='o')
        ax.fill_between(x_vals, y_v - y_v_std, y_v + y_v_std, color='orange', alpha=0.2)
        
        ax.axhline(y=base_train_acc, color='blue', linestyle='--', alpha=0.5, label='Base Train')
        ax.axhline(y=base_val_acc, color='orange', linestyle='--', alpha=0.5, label='Base Val')
        
        ax.set_title(f'Accuracy vs {method}')
        if method in ['L1', 'L2']:
            ax.set_xscale('log')
        ax.set_xlabel(f'{method} Param')
        ax.set_ylabel('Accuracy')
        ax.legend()
    fig2.suptitle(f'Accuracy (Train vs Val) - {title_suffix}', fontsize=16)
    if SAVE_PLOTS: fig2.savefig(f'results/accuracy_sweep_{title_suffix}.png')
    plt.show()

def plot_amplitude_grid_unified(nets_dict, param_lists, input_tensor, dataset_name, title_suffix='', seed_to_plot=0):
    methods = ['L1', 'L2', 'Dropout']
    n_cols = len(param_lists['L1']) 
    n_rows = 3
    
    total_samples = len(input_tensor)
    subset_size = min(200, total_samples)
    subset_idx = np.random.choice(total_samples, subset_size, replace=False)
    x_sample = input_tensor[subset_idx].to(DEVICE)
    
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
                
                ax_p.fill_between(x_axis, p10_sg, p90_sg, color='C1', alpha=0.3)
                ax_p.plot(x_axis, median_sg, color='C1', label='Median')
                ax_p.set_title(f'{method} {val:.2g}\n$\\overline{{P_{{90}} - P_{{10}}}} = {amp_perc:.2f}$')
                
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

    fig_p.suptitle(f'{dataset_name}: Amplitudes (Percentiles) - {title_suffix}', fontsize=16)
    fig_m.suptitle(f'{dataset_name}: Amplitudes (Mean $\pm$ Std) - {title_suffix}', fontsize=16)
    plt.tight_layout()
    
    if SAVE_PLOTS:
        fig_p.savefig(f'results/amp_perc_{dataset_name}_{title_suffix}.png')
        fig_m.savefig(f'results/amp_mean_{dataset_name}_{title_suffix}.png')
    plt.show()

# --- EJECUCIÓN PRINCIPAL ---
def run_experiment(train_dataset_exp, title_suffix="FullData", net_class=Net, **model_kwargs):
    actual_train_tensor = train_dataset_exp.tensors[0]
    
    train_eval_loader = DataLoader(train_dataset_exp, batch_size=1000, shuffle=False)
    train_loader = DataLoader(train_dataset_exp, batch_size=BATCH_SIZE, shuffle=True)
    
    reg_vals_log = [1e-3, 1e-2, 1e-1, 5e-1, 1.0]
    dropout_vals = [0.1, 0.2, 0.4, 0.5, 1.0]
    
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

    print(f"Training Baseline ({net_class.__name__}) & Computing Bins...")
    bin_edges = None
    
    n_train_samples = len(actual_train_tensor)
    n_val_samples = len(tensor_x_test)
    
    for seed in SEEDS:
        torch.manual_seed(seed)
        net = net_class(dropout_p=0.0, **model_kwargs)
        net = train_network(net, 'Baseline', 0.0, train_loader)
        
        if bin_edges is None:
            net.eval()
            with torch.no_grad():
                base_activations = net.hidden_layer(tensor_x_train.to(DEVICE)).cpu().numpy()
            bin_edges = compute_simple_bin_edges(base_activations, num_bins=BINS)

        states_train = get_num_internal_states_new(net, bin_edges, actual_train_tensor)
        states_val   = get_num_internal_states_new(net, bin_edges, tensor_x_test)
        
        pct_train = (states_train / n_train_samples) * 100
        pct_val   = (states_val / n_val_samples) * 100

        acc = evaluate_accuracy(net, test_dataloader)
        train_acc = evaluate_accuracy(net, train_eval_loader)
        
        results.append({'reg_type': 'Baseline', 'reg_val': 0.0, 'seed': seed, 
                        'val_acc': acc, 'train_acc': train_acc, 
                        'pct_states_train': pct_train, 
                        'pct_states_val': pct_val})
        nets_storage['Baseline'][0.0][seed] = net
        if SAVE_MODELS: torch.save(net.state_dict(), f'results/model_Base_s{seed}_{title_suffix}.pth')

    for method, p_list in param_lists.items():
        print(f"Training {method}...")
        for val in p_list:
            for seed in SEEDS:
                torch.manual_seed(seed)
                
                if method == 'Dropout':
                    net = net_class(dropout_p=val, **model_kwargs)
                    r_type, r_val = 'Dropout', 0.0
                else:
                    net = net_class(dropout_p=0.0, **model_kwargs)
                    r_type, r_val = method, val
                
                net = train_network(net, r_type, r_val, train_loader)
                acc = evaluate_accuracy(net, test_dataloader)
                train_acc = evaluate_accuracy(net, train_eval_loader)
                
                states_train = get_num_internal_states_new(net, bin_edges, actual_train_tensor)
                states_val   = get_num_internal_states_new(net, bin_edges, tensor_x_test)
                
                pct_train = (states_train / n_train_samples) * 100
                pct_val   = (states_val / n_val_samples) * 100
                
                results.append({'reg_type': method, 'reg_val': val, 'seed': seed, 
                                'val_acc': acc, 'train_acc': train_acc, 
                                'pct_states_train': pct_train, 
                                'pct_states_val': pct_val})
                nets_storage[method][val][seed] = net
                if SAVE_MODELS and seed == 0: torch.save(net.state_dict(), f'results/model_{method}_v{val:.3f}_{title_suffix}.pth')

    df = pd.DataFrame(results)
    plot_sweep_results(df, title_suffix)
    
    print("Plotting TRAIN Amplitudes...")
    plot_amplitude_grid_unified(nets_storage, param_lists, actual_train_tensor, "TRAIN", title_suffix)
    
    print("Plotting VALIDATION Amplitudes...")
    plot_amplitude_grid_unified(nets_storage, param_lists, tensor_x_test, "VAL", title_suffix)
    
    return df

# --- RUN ---
print(">>> RUN 1: Full Dataset (60k, 64 neurons)")
df_full = run_experiment(full_train_dataset, "FullData", net_class=Net, hidden_size=64)

print("\n>>> RUN 2: Reduced Dataset (1k, 64 neurons - Overfitting)")
indices = torch.randperm(len(tensor_x_train))[:1000]
reduced_train_dataset = TensorDataset(tensor_x_train[indices], tensor_y_train[indices])
df_reduced = run_experiment(reduced_train_dataset, "Reduced_Overfit", net_class=Net, hidden_size=64)

print("\n>>> RUN 3: Reduced Dataset (1k, 128 neurons, 4 layers)")
df_deep = run_experiment(reduced_train_dataset, "DeepNet_Overfit", net_class=MultiLayerNet, hidden_size=128, num_layers=4)

print("\n>>> RUN 4: Reduced Dataset (1k, ConvNet)")
# Experimento 4 con ConvNet sobre dataset reducido
df_conv = run_experiment(reduced_train_dataset, "ConvNet_Overfit", net_class=ConvNet, hidden_size=128)
