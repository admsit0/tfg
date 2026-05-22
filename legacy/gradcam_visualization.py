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
NUM_IMAGES_TO_SHOW = 5 # Número de imágenes de muestra para el Grad-CAM

# --- CONFIGURACIÓN PARA EL EXPERIMENTO 2 (CUSTOM) ---
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
# MOTOR GRAD-CAM (INTERCEPTOR DE GRADIENTES)
# =========================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        
        output = self.model(input_tensor)
        predicted_class = output.argmax(dim=1).item()
        if target_class is None:
            target_class = predicted_class
            
        self.model.zero_grad()
        
        score = output[0, target_class]
        score.backward()
        
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        
        # PREVENCIÓN MATEMÁTICA: Las capas lineales (fc1) no tienen dimensiones espaciales 2D.
        # Grad-CAM no puede proyectarlas sobre la imagen original.
        if len(gradients.shape) == 1:
            return np.zeros((32, 32), dtype=np.float32), predicted_class
        
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max != 0:
            cam = cam / cam_max
            
        cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
        cam_resized = F.interpolate(cam_tensor, size=(32, 32), mode='bilinear', align_corners=False)
        
        return cam_resized.squeeze().numpy(), predicted_class

# =========================================================
# CARGA DE DATOS (DETERMINISTA)
# =========================================================
def get_validation_loader(dataset_name, batch_size=200):
    print(f"⬇️ Cargando Validación de {dataset_name} (Determinista)...")
    if dataset_name == 'CIFAR10':
        test_set = datasets.CIFAR10(root='./data_raw', train=False, download=True)
        x_test = torch.tensor(test_set.data).permute(0, 3, 1, 2).float() / 255.0
        y_test = torch.tensor(test_set.targets, dtype=torch.long)
    elif dataset_name == 'SVHN':
        test_set = datasets.SVHN(root='./data_raw', split='test', download=True)
        x_test = torch.tensor(test_set.data).float() / 255.0
        y_test = torch.tensor(test_set.labels, dtype=torch.long)
    elif dataset_name == 'FashionMNIST':
        test_set = datasets.FashionMNIST(root='./data_raw', train=False, download=True)
        x_test = F.pad(test_set.data.float() / 255.0, (2, 2, 2, 2)).unsqueeze(1).repeat(1, 3, 1, 1)
        y_test = torch.tensor(test_set.targets, dtype=torch.long)
    elif dataset_name == 'CIFAR100':
        test_set = datasets.CIFAR100(root='./data_raw', train=False, download=True)
        x_test = torch.tensor(test_set.data).permute(0, 3, 1, 2).float() / 255.0
        y_test = torch.tensor(test_set.targets, dtype=torch.long)
        
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    return test_loader, len(torch.unique(y_test))

# =========================================================
# OBTENCIÓN DE CAMPEONES
# =========================================================
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

# =========================================================
# MOTOR DEL EXPERIMENTO VISUAL
# =========================================================
def run_gradcam_section(pdf, champions, val_loader, num_classes, device, models_dir, dataset_name, exp_title, layer_name):
    method_names = list(champions.keys())
    
    # Obtenemos un único batch determinista de imágenes
    images, labels = next(iter(val_loader))
    images = images[:NUM_IMAGES_TO_SHOW]
    labels = labels[:NUM_IMAGES_TO_SHOW]
    
    heatmaps_results = {m: [] for m in method_names}
    predictions_results = {m: [] for m in method_names}
    
    for method_name, info in tqdm(champions.items(), desc=f"Generando Grad-CAMs ({layer_name})", colour='yellow'):
        val = info['val']
        src_method = info['source_method']
        
        filename = f"CNN_{src_method}_{val:.5f}_ep{TARGET_EPOCH}.pth"
        filepath = os.path.join(models_dir, filename)
        
        if not os.path.exists(filepath):
            for _ in range(NUM_IMAGES_TO_SHOW):
                heatmaps_results[method_name].append(np.zeros((32, 32)))
                predictions_results[method_name].append(-1)
            continue
            
        state_dict = torch.load(filepath, map_location=device)
        has_bn = any("bn1.weight" in k for k in state_dict.keys())
        net = ConvNet(reg_method=src_method, reg_val=val, num_classes=num_classes, use_bn=has_bn).to(device)
        net.load_state_dict(state_dict)
        
        # Enlazamos la capa dinámica ('conv1', 'conv3' o 'fc1')
        target_layer = getattr(net, layer_name)
        cam_generator = GradCAM(net, target_layer)
        
        for i in range(NUM_IMAGES_TO_SHOW):
            img_tensor = images[i].unsqueeze(0).to(device)
            target_class = labels[i].item()
            
            heatmap, pred_class = cam_generator.generate(img_tensor, target_class=target_class)
            heatmaps_results[method_name].append(heatmap)
            predictions_results[method_name].append(pred_class)

    fig = plt.figure(figsize=(4 + 3 * len(method_names), 3 * NUM_IMAGES_TO_SHOW))
    gs = gridspec.GridSpec(NUM_IMAGES_TO_SHOW, len(method_names) + 1, wspace=0.1, hspace=0.3)
    fig.suptitle(f"{exp_title}\nAnálisis de Saliencia Visual (Grad-CAM en {layer_name}) - Dataset: {dataset_name}", fontsize=18, fontweight='bold')
    
    for i in range(NUM_IMAGES_TO_SHOW):
        # Columna 0: Imagen Original
        ax_orig = fig.add_subplot(gs[i, 0])
        img_display = images[i].permute(1, 2, 0).cpu().numpy()
        ax_orig.imshow(img_display)
        ax_orig.axis('off')
        if i == 0:
            ax_orig.set_title("Input Original", fontweight='bold', fontsize=14)
        ax_orig.text(0.5, -0.15, f"Real: {labels[i].item()}", ha='center', va='center', transform=ax_orig.transAxes, fontsize=12, fontweight='bold')
        
        # Columnas posteriores: Heatmaps
        for j, method_name in enumerate(method_names):
            ax_map = fig.add_subplot(gs[i, j + 1])
            heatmap = heatmaps_results[method_name][i]
            pred = predictions_results[method_name][i]
            
            ax_map.imshow(img_display)
            ax_map.imshow(heatmap, cmap='jet', alpha=0.55)
            ax_map.axis('off')
            
            if i == 0:
                ax_map.set_title(method_name, fontweight='bold', fontsize=14)
                
            color = 'green' if pred == labels[i].item() else 'red'
            ax_map.text(0.5, -0.15, f"Pred: {pred}", ha='center', va='center', transform=ax_map.transAxes, color=color, fontsize=12, fontweight='bold')
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

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
    
    output_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gradcam_analysis', f'{dataset_name}')
    os.makedirs(output_base_dir, exist_ok=True)
    
    pdf_output_path = os.path.join(output_base_dir, f'GradCAM_Report_{dataset_name}.pdf')
    
    print(f"\n🧪 INICIANDO EXPERIMENTO GRAD-CAM (Dataset: {dataset_name})")
    val_loader, num_classes = get_validation_loader(dataset_name)
    
    layers_to_analyze = ['conv1', 'conv3', 'fc1']
    
    with PdfPages(pdf_output_path) as pdf:
        
        global_champions = get_global_champions(data_dir)
        custom_champions = get_custom_champions(data_dir, CUSTOM_TARGETS)
        
        for layer_name in layers_to_analyze:
            print("\n" + "="*50)
            print(f"▶ ANALIZANDO CAPA: {layer_name.upper()}")
            print("="*50)

            # --- SECCIÓN 1: ÓPTIMOS ---
            if 'Baseline' in global_champions:
                run_gradcam_section(pdf, global_champions, val_loader, num_classes, device, models_dir, dataset_name, f"Exp 1: Parámetros Óptimos - Capa {layer_name.upper()}", layer_name)

            # --- SECCIÓN 2: CUSTOM ---
            if 'Baseline' in custom_champions:
                run_gradcam_section(pdf, custom_champions, val_loader, num_classes, device, models_dir, dataset_name, f"Exp 2: Parámetros Custom - Capa {layer_name.upper()}", layer_name)

    print(f"\n🚀 ¡PDF completado con 6 páginas y guardado en: {pdf_output_path}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    main(args.dataset)
