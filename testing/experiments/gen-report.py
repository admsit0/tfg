import matplotlib.pyplot as plt
import pandas as pd
from os import path
import numpy as np
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

# =========================================================
# CONFIG
# =========================================================
ZOOM_RANGE = False  # ← Set to True to enable zoomed reg_val view

BASE = 'outputs-higherGran/data'

dfs = {
    'Dropout': pd.read_csv(path.join(BASE, 'data_CNN_Dropout.csv')),
    'DataAug': pd.read_csv(path.join(BASE, 'data_CNN_DataAug.csv')),
    'L1': pd.read_csv(path.join(BASE, 'data_CNN_L1.csv')),
    'L2': pd.read_csv(path.join(BASE, 'data_CNN_L2.csv')),
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--"
})

max_epoch = max(df["epoch"].max() for df in dfs.values())
epoch_cmap = cm.viridis
epoch_norm = plt.Normalize(0, max_epoch)

all_train = np.concatenate([df['train_acc'].values for df in dfs.values()])
all_val   = np.concatenate([df['val_acc'].values   for df in dfs.values()])
acc_vmin = min(all_train.min(), all_val.min())
acc_vmax = max(all_train.max(), all_val.max())

with PdfPages("regularization_analysis_full2.pdf") as pdf:

    # =========================================================
    # PAGE 1 — LINE PLOTS
    # =========================================================
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, len(dfs), wspace=0.15, hspace=0.15)

    ax = np.array([
        [fig.add_subplot(gs[0, i]) for i in range(len(dfs))],
        [fig.add_subplot(gs[1, i]) for i in range(len(dfs))]
    ])

    for c, (name, df) in enumerate(dfs.items()):
        df = df.sort_values(["epoch", "reg_val"])

        for ep, g in df.groupby("epoch"):
            col = epoch_cmap(epoch_norm(ep))
            ax[0, c].plot(g["reg_val"], g["train_acc"], color=col, alpha=0.35)
            ax[1, c].plot(g["reg_val"], g["val_acc"],   color=col, alpha=0.35)

        ax[0, c].set_title(name)
        ax[1, c].set_xlabel("reg_val")

        if ZOOM_RANGE and name in ["L1", "L2"]:
            pos = df["reg_val"][df["reg_val"] > 0]
            for r in [0, 1]:
                ax[r, c].set_xscale("log")
                ax[r, c].set_xlim(pos.min()*0.8, pos.max()*1.2)

        elif name == "DataAug":
            ticks = sorted(df["reg_val"].unique())
            ax[0, c].set_xticks(ticks)
            ax[1, c].set_xticks(ticks)

    ax[0, 0].set_ylabel("Train accuracy")
    ax[1, 0].set_ylabel("Validation accuracy")

    sm = cm.ScalarMappable(norm=epoch_norm, cmap=epoch_cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.9).set_label("epoch")

    fig.suptitle("Accuracy vs Regularization Strength across Epochs", fontsize=15)
    pdf.savefig(fig)
    plt.close(fig)

    # =========================================================
    # PAGE 2 — SCATTER EPOCH vs REG
    # =========================================================
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, len(dfs), wspace=0.15, hspace=0.2)

    ax = np.array([
        [fig.add_subplot(gs[0, i]) for i in range(len(dfs))],
        [fig.add_subplot(gs[1, i]) for i in range(len(dfs))]
    ])

    for c, (name, df) in enumerate(dfs.items()):
        sc = ax[0, c].scatter(df["reg_val"], df["epoch"],
                              c=df["train_acc"],
                              vmin=acc_vmin, vmax=acc_vmax)
        ax[1, c].scatter(df["reg_val"], df["epoch"],
                         c=df["val_acc"],
                         vmin=acc_vmin, vmax=acc_vmax)

        ax[0, c].set_title(f"{name} – Train")
        ax[1, c].set_title(f"{name} – Val")
        ax[1, c].set_xlabel("reg_val")

        if ZOOM_RANGE and name in ["L1", "L2"]:
            pos = df["reg_val"][df["reg_val"] > 0]
            for r in [0, 1]:
                ax[r, c].set_xscale("log")
                ax[r, c].set_xlim(pos.min()*0.8, pos.max()*1.2)

    ax[0, 0].set_ylabel("epoch")
    ax[1, 0].set_ylabel("epoch")

    fig.colorbar(sc, ax=ax, shrink=0.9).set_label("accuracy")
    fig.suptitle("Training Dynamics: Epoch vs Regularization", fontsize=15)

    pdf.savefig(fig)
    plt.close(fig)

    # =========================================================
    # PAGE 3 — GENERALIZATION GAP
    # =========================================================
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, len(dfs), wspace=0.25)

    ax = [fig.add_subplot(gs[i]) for i in range(len(dfs))]

    for c, (name, df) in enumerate(dfs.items()):
        gap = df["train_acc"] - df["val_acc"]

        sc = ax[c].scatter(
            df["reg_val"], gap,
            c=df["epoch"],
            cmap=epoch_cmap,
            norm=epoch_norm,
            alpha=0.7
        )

        ax[c].axhline(0, color="black", ls="--")
        ax[c].set_title(name)
        ax[c].set_xlabel("reg_val")

        if ZOOM_RANGE and name in ["L1", "L2"]:
            pos = df["reg_val"][df["reg_val"] > 0]
            ax[c].set_xscale("log")
            ax[c].set_xlim(pos.min()*0.8, pos.max()*1.2)

    ax[0].set_ylabel("Generalization gap")

    fig.colorbar(sc, ax=ax, shrink=0.9).set_label("epoch")
    fig.suptitle("Generalization Gap (Train − Validation Accuracy)", fontsize=15)

    pdf.savefig(fig)
    plt.close(fig)
