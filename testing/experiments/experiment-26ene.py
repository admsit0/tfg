import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import transforms
import torchvision.datasets as datasets # Reemplazo de TF
import shutil
from tqdm import tqdm # Importamos TQDM para barras de progreso

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    'EPOCHS': 35,
    'BATCH_SIZE': 64,
    'SEEDS': [42],
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'OUTPUT_DIR': os.path.join(SCRIPT_DIR, 'outputs'),
    'DATA_LIMIT': 15000
}

def setup_directories():
    """Crea la estructura de carpetas limpia."""
    subdirs = ['models', 'plots', 'data']
    for sd in subdirs:
        os.makedirs(os.path.join(CONFIG['OUTPUT_DIR'], sd), exist_ok=True)
    print(f"✅ Directorios creados en {CONFIG['OUTPUT_DIR']}")

# --- 2. DEFINICIÓN DE DATASETS Y AUGMENTATION ---
def get_cifar_data():
    """Carga y preprocesa CIFAR-10 usando Torchvision (Sin TensorFlow)."""
    print("⬇️ Cargando datos CIFAR-10 con Torchvision...")
    
    # Descargar datos en carpeta local './data_raw' para no ensuciar
    train_set = datasets.CIFAR10(root='./data_raw', train=True, download=True)
    test_set = datasets.CIFAR10(root='./data_raw', train=False, download=True)
    
    # Extraer datos raw (numpy arrays uint8) y targets
    # CIFAR10.data es (N, H, W, C) igual que TF
    x_train = train_set.data 
    y_train = np.array(train_set.targets)
    
    x_test = test_set.data
    y_test = np.array(test_set.targets)
    
    # Normalizar
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Recortar dataset (opcional, para velocidad)
    if CONFIG['DATA_LIMIT']:
        print(f"⚠️ Limitando dataset a {CONFIG['DATA_LIMIT']} muestras.")
        x_train = x_train[:CONFIG['DATA_LIMIT']]
        y_train = y_train[:CONFIG['DATA_LIMIT']]
    
    # Convertir a Tensores (N, C, H, W) -> Permutamos igual que en tu código original
    tensor_x_train = torch.tensor(x_train).permute(0, 3, 1, 2)
    tensor_y_train = torch.tensor(y_train, dtype=torch.long) # Targets ya son 1D, no hace falta squeeze crítico
    tensor_x_test = torch.tensor(x_test).permute(0, 3, 1, 2)
    tensor_y_test = torch.tensor(y_test, dtype=torch.long)
    
    return (tensor_x_train, tensor_y_train), (tensor_x_test, tensor_y_test)

class AugmentedDataset(Dataset):
    """Dataset wrapper que aplica transformaciones al vuelo."""
    def __init__(self, x_tensor, y_tensor, transform=None):
        self.x = x_tensor
        self.y = y_tensor
        self.transform = transform
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        img = self.x[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.y[idx]

def get_augmentation_levels():
    """Define los niveles de Data Augmentation numéricos para el eje X."""
    t0 = None
    t1 = transforms.RandomHorizontalFlip(p=0.5)
    t2 = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4)
    ])
    t3 = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(15)
    ])
    
    return {0: t0, 1: t1, 2: t2, 3: t3}

# --- 3. DEFINICIÓN DE LA RED (CNN) ---
class ConvNet(nn.Module):
    def __init__(self, dropout_p=0.0):
        super(ConvNet, self).__init__()
        self.dropout_p = dropout_p
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.drop2d = nn.Dropout2d(p=dropout_p/2 if dropout_p > 0 else 0)
        self.drop1d = nn.Dropout(p=dropout_p)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.drop2d(F.relu(self.bn1(self.conv1(x)))))
        x = self.pool(self.drop2d(F.relu(self.bn2(self.conv2(x)))))
        x = self.pool(self.drop2d(F.relu(self.bn3(self.conv3(x)))))
        
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        if self.dropout_p > 0: 
            x = self.drop1d(x)
        x = self.fc2(x)
        return x

# --- 4. MOTOR DE ENTRENAMIENTO Y EVALUACIÓN ---
def evaluate(net, loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        # Usamos tqdm también para evaluación si es lenta, opcional. 
        # Aquí lo dejamos limpio para no saturar.
        for x, y in loader:
            x, y = x.to(CONFIG['DEVICE']), y.to(CONFIG['DEVICE'])
            out = net(x)
            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return correct / total

def train_experiment(net_name, reg_method, reg_val, train_loader, test_loader):
    """
    Entrena un modelo con TQDM integrado.
    """
    drop_val = reg_val if reg_method == 'Dropout' else 0.0
    
    if net_name == 'CNN':
        net = ConvNet(dropout_p=drop_val)
    else:
        raise ValueError("Tipo de red no soportado aún")
        
    net.to(CONFIG['DEVICE'])
    
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    
    # --- LOGGING VISUAL ---
    print(f"\n🚀 Iniciando Run: {net_name} | {reg_method} = {reg_val}")
    print("-" * 60)

    # Barra de progreso principal para las ÉPOCAS
    # position=0 y leave=True mantienen esta barra visible
    pbar_epoch = tqdm(range(1, CONFIG['EPOCHS'] + 1), desc="Progreso Entrenamiento", unit="epoch", leave=True, colour='green')

    for epoch in pbar_epoch:
        net.train()
        running_loss = 0.0
        
        # Barra de progreso interna para los BATCHES (opcionalmente leave=False para que desaparezca al acabar la epoch)
        # Usamos enumerate para tener el índice si quisiéramos
        pbar_batch = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, unit="batch", colour='cyan')
        
        for x, y in pbar_batch:
            x, y = x.to(CONFIG['DEVICE']), y.to(CONFIG['DEVICE'])
            optimizer.zero_grad()
            out = net(x)
            loss = criterion(out, y)
            
            if reg_method == 'L1' or reg_method == 'L2':
                reg_loss = 0
                for param in net.parameters():
                    if reg_method == 'L1':
                        reg_loss += torch.norm(param, 1)
                    else:
                        reg_loss += torch.norm(param, 2)
                loss += reg_val * reg_loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # Actualizar barra de batch con loss actual
            pbar_batch.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Evaluar
        train_acc = evaluate(net, train_loader) 
        val_acc = evaluate(net, test_loader)
        
        # Actualizar la barra de épocas con las métricas
        pbar_epoch.set_postfix({'Train Acc': f'{train_acc:.3f}', 'Val Acc': f'{val_acc:.3f}'})
        
        history.append({
            'epoch': epoch,
            'reg_method': reg_method,
            'reg_val': reg_val,
            'train_acc': train_acc,
            'val_acc': val_acc
        })
        
        model_name = f"{net_name}_{reg_method}_{reg_val:.4f}_ep{epoch}.pth"
        save_path = os.path.join(CONFIG['OUTPUT_DIR'], 'models', model_name)
        torch.save(net.state_dict(), save_path)
    
    return pd.DataFrame(history)

# --- 5. VISUALIZACIÓN 3D ---
def plot_3d_results(df, net_name, reg_method):
    modes = ['train_acc', 'val_acc']
    titles = ['Training Accuracy', 'Validation Accuracy']
    
    print(f"📊 Generando gráficas 3D para {reg_method}...")
    
    for mode, title in zip(modes, titles):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        xs = df['reg_val']
        ys = df['epoch']
        zs = df[mode]
        
        img = ax.scatter(xs, ys, zs, c=zs, cmap='viridis', s=50, alpha=0.8)
        
        ax.set_xlabel(f'{reg_method} Value')
        ax.set_ylabel('Epoch')
        ax.set_zlabel('Accuracy')
        ax.set_title(f'{net_name} - {reg_method}: {title}')
        
        fig.colorbar(img, ax=ax, label='Accuracy')
        
        filename = f"3D_{net_name}_{reg_method}_{mode}.png"
        path = os.path.join(CONFIG['OUTPUT_DIR'], 'plots', filename)
        plt.savefig(path)
        plt.close()
        # print(f"Gráfica guardada: {path}") # Comentado para reducir ruido

# --- 6. MACROBUCLE PRINCIPAL ---
def run_macro_experiment():
    print("""
    ===================================================
          INICIANDO MACRO EXPERIMENTO DE REGULARIZACIÓN
    ===================================================
    """)
    setup_directories()
    
    # 1. Datos (Ahora usa Torchvision)
    (tx_train, ty_train), (tx_test, ty_test) = get_cifar_data()
    
    # 2. Configuración de Experimentos
    NET_TYPES = ['CNN']
    
    EXPERIMENTS = {
        'Dropout': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.6],
        'DataAug': [0, 1, 2, 3] 
    }
    
    aug_transforms = get_augmentation_levels()
    
    # 3. Iteración
    for net_name in NET_TYPES:
        print(f"\n🔵 Evaluando Red: {net_name}")
        
        # Iterar sobre los métodos
        for reg_method, param_values in EXPERIMENTS.items():
            print(f"\n🔶 Método: {reg_method}")
            print("=" * 30)
            
            all_runs_df = pd.DataFrame()
            
            # Barra de progreso para los VALORES DEL PARÁMETRO
            pbar_params = tqdm(param_values, desc=f"Hyperparams ({reg_method})", unit="run", colour='yellow')
            
            for val in pbar_params:
                # Actualizar descripción de la barra
                pbar_params.set_description(f"Running {reg_method}={val}")
                
                if reg_method == 'DataAug':
                    current_transform = aug_transforms[val]
                    train_ds = AugmentedDataset(tx_train, ty_train, transform=current_transform)
                else:
                    train_ds = AugmentedDataset(tx_train, ty_train, transform=None)
                
                train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
                test_loader = DataLoader(TensorDataset(tx_test, ty_test), batch_size=1000)
                
                # Ejecutar entrenamiento
                run_df = train_experiment(net_name, reg_method, val, train_loader, test_loader)
                
                all_runs_df = pd.concat([all_runs_df, run_df], ignore_index=True)
            
            csv_name = f"data_{net_name}_{reg_method}.csv"
            all_runs_df.to_csv(os.path.join(CONFIG['OUTPUT_DIR'], 'data', csv_name), index=False)
            
            plot_3d_results(all_runs_df, net_name, reg_method)
            print(f"✅ Finalizado bloque {reg_method}. Datos guardados.")

if __name__ == "__main__":
    run_macro_experiment()