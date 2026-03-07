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
# DATASETS
# =========================================================

def get_dataset_loaders(dataset_name, batch_size=200, limit=25000):

    if dataset_name == 'CIFAR10':

        train_set = datasets.CIFAR10(root='./data_raw', train=True, download=True)
        test_set = datasets.CIFAR10(root='./data_raw', train=False, download=True)

        x_train = torch.tensor(train_set.data).permute(0,3,1,2).float()/255.
        y_train = torch.tensor(train_set.targets)

        x_test = torch.tensor(test_set.data).permute(0,3,1,2).float()/255.
        y_test = torch.tensor(test_set.targets)

        num_classes = 10

    elif dataset_name == 'SVHN':

        train_set = datasets.SVHN(root='./data_raw', split='train', download=True)
        test_set = datasets.SVHN(root='./data_raw', split='test', download=True)

        x_train = torch.tensor(train_set.data).float()/255.
        y_train = torch.tensor(train_set.labels)

        x_test = torch.tensor(test_set.data).float()/255.
        y_test = torch.tensor(test_set.labels)

        num_classes = 10

    elif dataset_name == 'FashionMNIST':

        train_set = datasets.FashionMNIST(root='./data_raw', train=True, download=True)
        test_set = datasets.FashionMNIST(root='./data_raw', train=False, download=True)

        x_train = F.pad(train_set.data.float()/255., (2,2,2,2)).unsqueeze(1).repeat(1,3,1,1)
        x_test = F.pad(test_set.data.float()/255., (2,2,2,2)).unsqueeze(1).repeat(1,3,1,1)

        y_train = train_set.targets
        y_test = test_set.targets

        num_classes = 10

    elif dataset_name == 'CIFAR100':

        train_set = datasets.CIFAR100(root='./data_raw', train=True, download=True)
        test_set = datasets.CIFAR100(root='./data_raw', train=False, download=True)

        x_train = torch.tensor(train_set.data).permute(0,3,1,2).float()/255.
        y_train = torch.tensor(train_set.targets)

        x_test = torch.tensor(test_set.data).permute(0,3,1,2).float()/255.
        y_test = torch.tensor(test_set.targets)

        num_classes = 100

    if limit:
        x_train = x_train[:limit]
        y_train = y_train[:limit]

    train_loader = DataLoader(TensorDataset(x_train,y_train),batch_size=batch_size,shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test,y_test),batch_size=batch_size,shuffle=False)

    return train_loader,test_loader,num_classes


# =========================================================
# UTILIDADES
# =========================================================

def extract_model_info(filename):

    pattern = r"CNN_(.*?)_([0-9\.]+)_ep(\d+)\.pth"

    m = re.search(pattern,filename)

    if m:
        return m.group(1),float(m.group(2)),int(m.group(3))

    return None,None,None


def get_best_epochs_and_acc(data_dir):

    best_models = {}

    if not os.path.exists(data_dir):
        return best_models

    for f in os.listdir(data_dir):

        if f.startswith("data_CNN_"):

            df = pd.read_csv(os.path.join(data_dir,f))

            method = f.replace("data_CNN_","").replace(".csv","")

            idx_max = df.groupby('reg_val')['val_acc'].idxmax()

            for _,row in df.loc[idx_max].iterrows():

                val = round(row['reg_val'],6)

                best_models[(method,val)] = {

                    'epoch':int(row['epoch']),
                    'val_acc':row['val_acc'],
                    'train_acc':row['train_acc']
                }

    return best_models


# =========================================================
# LIMITES GLOBALES
# =========================================================

def find_global_limits(device,base_models_dir,val_loader,num_classes,best_models_info):

    print("Calculando límites globales con modelo baseline")

    baseline_method=None
    best_ep=None

    for (m,v),info in best_models_info.items():
        if v==0.0:
            baseline_method=m
            best_ep=info['epoch']
            break

    model_files=[f for f in os.listdir(base_models_dir) if f.endswith(".pth")]

    for f in model_files:

        m,v,ep=extract_model_info(f)

        if m==baseline_method and v==0.0 and ep==best_ep:
            baseline_file=f
            break

    filepath=os.path.join(base_models_dir,baseline_file)

    state_dict=torch.load(filepath,map_location=device)

    has_bn=any("bn1.weight" in k for k in state_dict)

    net=ConvNet(num_classes=num_classes,use_bn=has_bn).to(device)

    net.load_state_dict(state_dict)
    net.eval()

    layers=['conv1','conv3','fc1']

    limits={l:{'min':float('inf'),'max':float('-inf')} for l in layers}

    activations={}

    def get_hook(name):
        def hook(m,i,o):

            batch_min=o.min().item()
            batch_max=o.max().item()

            limits[name]['min']=min(limits[name]['min'],batch_min)
            limits[name]['max']=max(limits[name]['max'],batch_max)

        return hook

    h1=net.conv1.register_forward_hook(get_hook('conv1'))
    h2=net.conv3.register_forward_hook(get_hook('conv3'))
    h3=net.fc1.register_forward_hook(get_hook('fc1'))

    with torch.no_grad():

        for x,_ in val_loader:
            net(x.to(device))

    h1.remove();h2.remove();h3.remove()

    for l in layers:
        if limits[l]['max']==limits[l]['min']:
            limits[l]['max']+=1e-6

    return limits


# =========================================================
# ESTADOS
# =========================================================

def compute_unique_states(net,loader,device,layers,limits):

    activations={l:None for l in layers}

    def get_hook(name):
        def hook(m,i,o):
            activations[name]=o.detach()
        return hook

    h1=net.conv1.register_forward_hook(get_hook('conv1'))
    h2=net.conv3.register_forward_hook(get_hook('conv3'))
    h3=net.fc1.register_forward_hook(get_hook('fc1'))

    unique_states={l:set() for l in layers}

    total_images=0

    with torch.no_grad():

        for x,_ in loader:

            x=x.to(device)

            net(x)

            batch_size=x.size(0)

            total_images+=batch_size

            for l in layers:

                act=activations[l]

                act=act.view(batch_size,-1)

                act=torch.clamp(act,limits[l]['min'],limits[l]['max'])

                for i in range(batch_size):

                    hist=torch.histc(
                        act[i],
                        bins=N_BINS,
                        min=limits[l]['min'],
                        max=limits[l]['max']
                    )

                    state=tuple(hist.cpu().numpy().astype(int))

                    unique_states[l].add(state)

    h1.remove();h2.remove();h3.remove()

    results={}

    for l in layers:
        results[l]=(len(unique_states[l])/total_images)*100

    return results,total_images


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["CIFAR10", "SVHN", "FashionMNIST", "CIFAR100"]
    )

    parser.add_argument("--batch_size", type=int, default=200)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    # =====================================================
    # DIRECTORIOS (estructura del proyecto)
    # =====================================================

    base_dir = f"./outputs_{args.dataset}"

    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    output_dir = base_dir

    print("Using directories:")
    print("data_dir   =", data_dir)
    print("models_dir =", models_dir)
    print("output_dir =", output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # =====================================================
    # DATASET
    # =====================================================

    train_loader, val_loader, num_classes = get_dataset_loaders(
        args.dataset,
        batch_size=args.batch_size
    )

    # =====================================================
    # BEST MODELS
    # =====================================================

    best_models = get_best_epochs_and_acc(data_dir)

    if len(best_models) == 0:
        raise RuntimeError(f"No best models found in {data_dir}")

    print("Found", len(best_models), "best models")

    # =====================================================
    # GLOBAL LIMITS
    # =====================================================

    limits = find_global_limits(
        device,
        models_dir,
        val_loader,
        num_classes,
        best_models
    )

    print("Global limits:")
    print(limits)

    # =====================================================
    # EVALUAR TODOS LOS MODELOS
    # =====================================================

    layers = ["conv1", "conv3", "fc1"]

    results = []

    for (method, val), info in tqdm(best_models.items()):

        epoch = info["epoch"]

        model_name = f"CNN_{method}_{val}_ep{epoch}.pth"

        path = os.path.join(models_dir, model_name)

        if not os.path.exists(path):
            print("Missing:", model_name)
            continue

        state_dict = torch.load(path, map_location=device)

        has_bn = any("bn1.weight" in k for k in state_dict)

        net = ConvNet(
            reg_method=method,
            reg_val=val,
            num_classes=num_classes,
            use_bn=has_bn
        ).to(device)

        net.load_state_dict(state_dict)
        net.eval()

        states_train, ntrain = compute_unique_states(
            net,
            train_loader,
            device,
            layers,
            limits
        )

        states_val, nval = compute_unique_states(
            net,
            val_loader,
            device,
            layers,
            limits
        )

        for layer in layers:

            results.append({
                "method": method,
                "reg_val": val,
                "epoch": epoch,
                "layer": layer,
                "states_train": states_train[layer],
                "states_val": states_val[layer],
                "train_acc": info["train_acc"],
                "val_acc": info["val_acc"]
            })

    df = pd.DataFrame(results)

    # =====================================================
    # GUARDAR RESULTADOS
    # =====================================================

    csv_path = os.path.join(output_dir, f"states_{args.dataset}.csv")

    df.to_csv(csv_path, index=False)

    print("Saved:", csv_path)

    # =====================================================
    # PDF
    # =====================================================

    pdf_path = os.path.join(output_dir, f"states_{args.dataset}.pdf")

    with PdfPages(pdf_path) as pdf:

        methods = df["method"].unique()

        for layer in layers:

            fig = plt.figure(figsize=(8,5))

            for method in methods:

                sub = df[(df.method == method) & (df.layer == layer)]

                sub = sub.sort_values("reg_val")

                plt.plot(
                    sub["reg_val"],
                    sub["states_val"],
                    marker="o",
                    label=method
                )

            plt.title(f"{args.dataset} — {layer}")
            plt.xlabel("Regularization value")
            plt.ylabel("% unique states (validation)")
            plt.legend()

            pdf.savefig(fig)
            plt.close()

    print("PDF saved:", pdf_path)