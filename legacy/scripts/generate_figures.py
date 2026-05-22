"""
Generador de Figuras para el TFG
=================================
Genera todas las figuras del §5 y §6 desde los CSVs en results/.
Ejecutar desde el directorio raíz de "tfg ordered":

    python scripts/generate_figures.py

    # Directorio de salida personalizado (por defecto: thesis img/)
    python scripts/generate_figures.py --out-dir /ruta/img

    # Solo listar qué se generaría sin escribir archivos
    python scripts/generate_figures.py --dry-run

Requisitos: pandas matplotlib seaborn numpy
    pip install pandas matplotlib seaborn numpy

Figuras generadas (19) — nombres que \includegraphics{} espera en img/:
    fig01_accuracy_curvas_CIFAR10.pdf      → §5.1: Curvas train/val (CIFAR-10)
    fig01_accuracy_curvas_SVHN.pdf         → §5.1: Curvas train/val (SVHN)
    fig02_ranking_val_acc_CIFAR10.pdf      → §5.1: Ranking val_acc (CIFAR-10)
    fig02_ranking_val_acc_SVHN.pdf         → §5.1: Ranking val_acc (SVHN)
    fig03_entropia_vs_acc.pdf              → §5.2: Scatter entropía vs val_acc
    fig04_dispersion_vs_acc.pdf            → §5.2: Scatter dispersión vs val_acc
    fig05_temporal_uniquePctg_Custom.pdf   → §5.2: Trayectorias temporales (Custom)
    fig05_temporal_uniquePctg_Optimum.pdf  → §5.2: Trayectorias temporales (Optimum)
    fig06_robustez_datos_Custom.pdf        → §5.2: Ruido en datos (Custom)
    fig06_robustez_datos_Optimum.pdf       → §5.2: Ruido en datos (Optimum)
    fig07_robustez_pesos_conv1_Custom.pdf  → §5.2: Ruido en pesos conv1 (Custom)
    fig07_robustez_pesos_conv1_Optimum.pdf → §5.2: Ruido en pesos conv1 (Optimum)
    fig07b_robustez_pesos_conv3_Custom.pdf → §5.2: Ruido en pesos conv3 (Custom)
    fig07b_robustez_pesos_conv3_Optimum.pdf→ §5.2: Ruido en pesos conv3 (Optimum)
    fig07c_comparativa_capas_sigma15_Custom.pdf → §5.2: Bar chart 3 capas
    fig08_robustez_pesos_fc1_Custom.pdf    → §5.2: Ruido en pesos fc1 (Custom)
    fig08_robustez_pesos_fc1_Optimum.pdf   → §5.2: Ruido en pesos fc1 (Optimum)
    fig09_radar_multidimensional.pdf       → §6: Radar evaluación multidimensional
    fig10_svhn_unique_pctg_vs_acc.pdf      → §5.2: SVHN unique_pctg vs val_acc
"""

import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ─── CLI ───────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="Genera figuras del TFG desde CSVs.")
_parser.add_argument(
    "--out-dir",
    type=Path,
    default=None,
    help="Directorio de salida (por defecto: ../img)",
)
_parser.add_argument("--dry-run", action="store_true", help="Solo imprime qué se generaría.")
_ARGS, _UNKNOWN = _parser.parse_known_args()

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent  # raíz de "tfg ordered"
DATA_DIR = BASE_DIR / "results"

# Thesis img/ directory two levels up from "tfg ordered"
_DEFAULT_OUT = BASE_DIR.parent / "img"
OUT_DIR: Path = _ARGS.out_dir if _ARGS.out_dir else _DEFAULT_OUT

if not _ARGS.dry_run:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

# Paleta de colores por método (consistente en todas las figuras)
COLORS = {
    "Baseline":      "#666666",
    "L1":            "#E05C4A",
    "L2":            "#F0A03C",
    "Dropout":       "#5B8DB8",
    "BatchNorm":     "#7A5DB2",
    "DataAug":       "#3BAA6E",
    "GaussianNoise": "#C97AB2",
    "EarlyStopping": "#9B8A6E",
}

MARKERS = {
    "Baseline":      "o",
    "L1":            "s",
    "L2":            "D",
    "Dropout":       "^",
    "BatchNorm":     "P",
    "DataAug":       "*",
    "GaussianNoise": "X",
    "EarlyStopping": "v",
}

plt.rcParams.update({
    "font.family":  "serif",
    "font.size":    11,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi":   150,
    "savefig.bbox": "tight",
    "savefig.dpi":  300,
})


# ─── HELPERS ───────────────────────────────────────────────────────────────────

def save(fig, name):
    if _ARGS.dry_run:
        print(f"  [DRY] {name}")
        plt.close(fig)
        return
    path = OUT_DIR / name
    fig.savefig(path)
    print(f"  [OK]  {name}")
    plt.close(fig)


def ensure_baseline_csv(folder: Path, dataset: str) -> None:
    """Warn if the Baseline CSV is missing — figures will be incomplete without it."""
    baseline = folder / f"data_CNN_Baseline.csv"
    if not baseline.exists():
        print(f"  [WARN] No Baseline CSV in {folder} — some figures may be incomplete.")


def _safe_read(path: Path) -> "pd.DataFrame | None":
    if not path.exists():
        print(f"  [SKIP] Missing: {path.name}")
        return None
    return pd.read_csv(path)


def load_accuracy_csvs(dataset="CIFAR10"):
    """Carga todos los CSVs de accuracy y devuelve un DataFrame unificado."""
    folder = DATA_DIR / "accuracy" / dataset.lower()
    ensure_baseline_csv(folder, dataset)
    frames = []
    for csv in sorted(folder.glob("data_CNN_*.csv")):
        df = pd.read_csv(csv)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No CSVs found in {folder}")
    return pd.concat(frames, ignore_index=True)


def best_val_per_method(df):
    """Para cada método, retorna la fila con la mejor val_acc máxima."""
    idx = df.groupby("reg_method")["val_acc"].transform("max") == df["val_acc"]
    best = df[idx].drop_duplicates("reg_method").copy()
    return best


# ─── FIGURA 1: Curvas de accuracy (train y val) para cada método ───────────────

def fig01_accuracy_curvas(dataset="CIFAR10"):
    print(f"\n[Fig 01] Curvas accuracy {dataset}")
    df = load_accuracy_csvs(dataset)

    # Un subplot por método (los mejores hiperparámetros)
    methods = sorted(df["reg_method"].unique())
    n = len(methods)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5), sharey=True)
    axes = axes.flatten()

    for i, method in enumerate(methods):
        ax = axes[i]
        sub = df[df["reg_method"] == method]
        # Seleccionar el reg_val con mejor val_acc final
        best_rv = sub.groupby("reg_val")["val_acc"].max().idxmax()
        sub_best = sub[sub["reg_val"] == best_rv]
        color = COLORS.get(method, "steelblue")
        ax.plot(sub_best["epoch"], sub_best["train_acc"], "--", color=color, alpha=0.6, label="Train")
        ax.plot(sub_best["epoch"], sub_best["val_acc"], "-",  color=color, lw=2,   label="Val")
        ax.set_title(f"{method}\n(config={best_rv})", fontsize=10)
        ax.set_xlabel("Época")
        if i % cols == 0:
            ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Ocultar subplots vacíos
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Curvas de Accuracy por Método — {dataset}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, f"fig01_accuracy_curvas_{dataset}.pdf")


# ─── FIGURA 2: Ranking final val_acc ──────────────────────────────────────────

def fig02_ranking_val_acc(dataset="CIFAR10"):
    print(f"\n[Fig 02] Ranking val_acc {dataset}")
    df = load_accuracy_csvs(dataset)
    best = best_val_per_method(df).sort_values("val_acc", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLORS.get(m, "steelblue") for m in best["reg_method"]]
    bars = ax.barh(best["reg_method"], best["val_acc"], color=colors, edgecolor="white", height=0.6)

    for bar, v in zip(bars, best["val_acc"]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", ha="left", fontsize=10)

    ax.set_xlabel("Mejor val_acc (config óptima)")
    ax.set_title(f"Ranking de Técnicas de Regularización — {dataset}", fontweight="bold")
    ax.set_xlim(0, min(1.0, best["val_acc"].max() + 0.05))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save(fig, f"fig02_ranking_val_acc_{dataset}.pdf")


# ─── FIGURA 3: Entropía vs val_acc (scatter) ──────────────────────────────────

def fig03_entropia_vs_acc():
    print("\n[Fig 03] Entropía H vs val_acc (fc1, val, CIFAR-10)")
    csv = DATA_DIR / "internal_activations" / "cifar10" / "CIFAR10_30bins_entropy_states.csv"
    df = pd.read_csv(csv)
    sub = df[(df["layer"] == "fc1") & (df["split"] == "val")].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in sub["reg_method"].unique():
        m = sub[sub["reg_method"] == method]
        ax.scatter(m["entropy"], m["val_acc"],
                   color=COLORS.get(method, "gray"),
                   marker=MARKERS.get(method, "o"),
                   s=60, alpha=0.75, label=method, zorder=3)

    # Zona óptima
    ax.axvspan(1.1, 1.8, alpha=0.12, color="green", label="Zona óptima (H ∈ [1.1, 1.8])")
    ax.axvline(1.1, color="green", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(1.8, color="green", ls="--", lw=0.8, alpha=0.5)

    ax.set_xlabel("Entropía de Shannon H (fc1, validación)")
    ax.set_ylabel("val_acc")
    ax.set_title("Entropía de Activaciones vs Accuracy de Validación\n(CIFAR-10, 30 bins, capa fc1)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "fig03_entropia_vs_acc.pdf")


# ─── FIGURA 4: Dispersión vs val_acc (scatter) ────────────────────────────────

def fig04_dispersion_vs_acc():
    print("\n[Fig 04] Dispersión vs val_acc (fc1, val, CIFAR-10)")
    csv = DATA_DIR / "internal_activations" / "cifar10" / "CIFAR10_dispersion_ratio.csv"
    df = pd.read_csv(csv)
    sub = df[(df["layer"] == "fc1") & (df["split"] == "val")].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in sub["reg_method"].unique():
        m = sub[sub["reg_method"] == method]
        ax.scatter(m["dispersion_ratio"], m["val_acc"],
                   color=COLORS.get(method, "gray"),
                   marker=MARKERS.get(method, "o"),
                   s=60, alpha=0.75, label=method, zorder=3)

    # Zona óptima de dispersión
    ax.axvspan(0.27, 0.46, alpha=0.12, color="blue", label="Zona óptima (ratio ∈ [0.27, 0.46])")
    ax.axvline(0.27, color="blue", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(0.46, color="blue", ls="--", lw=0.8, alpha=0.5)

    ax.set_xlabel("Ratio de dispersión relativa al Baseline (fc1)")
    ax.set_ylabel("val_acc")
    ax.set_title("Dispersión de Activaciones vs Accuracy de Validación\n(CIFAR-10, capa fc1)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "fig04_dispersion_vs_acc.pdf")


# ─── FIGURA 5: Evolución temporal unique_pctg ─────────────────────────────────

def fig05_temporal_unique_pctg(series="Custom"):
    print(f"\n[Fig 05] Evolución temporal unique_pctg ({series})")
    csv = DATA_DIR / "temporal_evolution" / f"Bottleneck_Data_CIFAR10_{series}.csv"
    df = pd.read_csv(csv)
    # Bottleneck CSV uses 'method' (not 'reg_method')
    method_col = "method" if "method" in df.columns else "reg_method"
    sub = df[df["layer"] == "fc1"].copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    for method in sorted(sub[method_col].unique()):
        m = sub[sub[method_col] == method].sort_values("epoch")
        # Puede haber varias reg_val → plot cada una
        for rv, grp in m.groupby("reg_val"):
            color = COLORS.get(method, "gray")
            lw = 2 if method != "Baseline" else 1.5
            ls = "-" if method != "Baseline" else "--"
            label = f"{method} ({rv})" if len(m["reg_val"].unique()) > 1 else method
            ax.plot(grp["epoch"], grp["unique_pctg"], color=color, lw=lw, ls=ls,
                    marker=MARKERS.get(method, "o"), markersize=4, label=label, alpha=0.85)

    ax.set_xlabel("Época")
    ax.set_ylabel("% Estados Únicos en fc1")
    ax.set_title(f"Evolución Temporal de la Representación Interna — fc1\n(CIFAR-10, serie {series})", fontweight="bold")
    ax.set_ylim(-2, 105)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, f"fig05_temporal_uniquePctg_{series}.pdf")


# ─── FIGURA 6: Robustez a ruido en datos ─────────────────────────────────────

def fig06_robustez_datos(series="Custom"):
    print(f"\n[Fig 06] Robustez ruido datos ({series})")
    csv = DATA_DIR / "robustness_data_noise" / f"Data_Noise_Data_CIFAR10_{series}.csv"
    df = pd.read_csv(csv)

    # Normalizar: retención = acc / acc_sin_ruido (sigma=0)
    # Ambos CSVs de ruido usan 'method' y 'acc_mean' (en porcentaje o fracción)
    method_col = "method" if "method" in df.columns else "reg_method"
    acc_col = "acc_mean" if "acc_mean" in df.columns else "acc"
    sigma_col = "sigma" if "sigma" in df.columns else "noise_sigma"
    acc0 = df[df[sigma_col] == 0.0].groupby(method_col)[acc_col].mean().to_dict()
    df["retencion"] = df.apply(lambda r: r[acc_col] / acc0.get(r[method_col], 1.0), axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Izquierda: accuracy absoluto
    ax = axes[0]
    for method in sorted(df[method_col].unique()):
        m = df[df[method_col] == method].sort_values(sigma_col)
        ax.plot(m[sigma_col], m[acc_col], color=COLORS.get(method, "gray"),
                marker=MARKERS.get(method, "o"), ms=4, lw=2, label=method)
        if "acc_std" in df.columns:
            ax.fill_between(m[sigma_col],
                            m[acc_col] - m["acc_std"],
                            m[acc_col] + m["acc_std"],
                            color=COLORS.get(method, "gray"), alpha=0.1)
    ax.set_xlabel("Intensidad del ruido gaussiano (σ)")
    ax.set_ylabel("Accuracy de validación")
    ax.set_title("Accuracy vs Ruido en datos")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Derecha: retención relativa
    ax = axes[1]
    for method in sorted(df[method_col].unique()):
        m = df[df[method_col] == method].sort_values(sigma_col)
        ax.plot(m[sigma_col], m["retencion"] * 100, color=COLORS.get(method, "gray"),
                marker=MARKERS.get(method, "o"), ms=4, lw=2, label=method)
    ax.set_xlabel("Intensidad del ruido gaussiano (σ)")
    ax.set_ylabel("Retención de accuracy (%)")
    ax.set_title("Retención relativa vs Ruido en datos")
    ax.axhline(100, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(f"Robustez a Ruido Gaussiano en Datos de Entrada — CIFAR-10 ({series})", fontweight="bold")
    fig.tight_layout()
    save(fig, f"fig06_robustez_datos_{series}.pdf")


# ─── FIGURA 7 & 8: Robustez a ruido en pesos ─────────────────────────────────

_LAYER_FIG_PREFIX = {"conv1": "fig07", "conv3": "fig07b", "fc1": "fig08"}


def fig07_fig08_robustez_pesos(series="Custom"):
    print(f"\n[Fig 07-08] Robustez ruido pesos ({series})")

    for layer in ["conv1", "conv3", "fc1"]:
        csv = DATA_DIR / "robustness_weight_noise" / f"Flat_Minima_Data_CIFAR10_{layer}_{series}.csv"
        if not csv.exists():
            print(f"  MISSING: {csv.name}")
            continue
        df = pd.read_csv(csv)

        # Columnas esperadas: method, reg_val, sigma, acc_mean (o acc)
        acc_col = "acc_mean" if "acc_mean" in df.columns else "acc"
        sigma_col = "sigma" if "sigma" in df.columns else "noise_sigma"

        # Normalizar: retención = acc / acc_sin_ruido
        acc0 = df[df[sigma_col] == 0.0].groupby("method")[acc_col].mean().to_dict()
        df["retencion"] = df.apply(lambda r: r[acc_col] / acc0.get(r["method"], 1.0), axis=1)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        for method in sorted(df["method"].unique()):
            m = df[df["method"] == method].sort_values(sigma_col)
            ax.plot(m[sigma_col], m[acc_col], color=COLORS.get(method, "gray"),
                    marker=MARKERS.get(method, "o"), ms=4, lw=2, label=method)
        ax.set_xlabel(f"Intensidad del ruido en pesos de {layer} (σ)")
        ax.set_ylabel("Accuracy de validación")
        ax.set_title(f"Accuracy vs Ruido en pesos ({layer})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[1]
        for method in sorted(df["method"].unique()):
            m = df[df["method"] == method].sort_values(sigma_col)
            ax.plot(m[sigma_col], m["retencion"] * 100, color=COLORS.get(method, "gray"),
                    marker=MARKERS.get(method, "o"), ms=4, lw=2, label=method)
        ax.set_xlabel(f"Intensidad del ruido en pesos de {layer} (σ)")
        ax.set_ylabel("Retención de accuracy (%)")
        ax.set_title(f"Retención relativa vs Ruido en pesos ({layer})")
        ax.axhline(100, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        fig.suptitle(f"Robustez a Perturbaciones en Pesos — {layer} — CIFAR-10 ({series})", fontweight="bold")
        fig.tight_layout()
        prefix = _LAYER_FIG_PREFIX[layer]
        save(fig, f"{prefix}_robustez_pesos_{layer}_{series}.pdf")


# ─── FIGURA 9: Radar/tabla multidimensional (§7) ─────────────────────────────

def fig09_radar_multidimensional():
    """
    Figura de radar comparativa de los 6 regularizadores en las 3 dimensiones
    (accuracy, robustez datos, robustez pesos).
    Usa valores aproximados de las tablas ya compiladas.
    """
    print("\n[Fig 09] Radar multidimensional")

    # Valores normalizados [0,1] extraídos de las tablas del TFG
    # Columnas: [accuracy, robustez_datos, robustez_pesos_fc1]
    # (normalizado respecto al máximo observado)
    data = {
        # [accuracy, robustez_datos(ret σ=0.3 Custom), robustez_pesos_fc1(ret σ=0.1 Custom)]
        # Accuracy normalizado contra max (DataAug=0.785). Robustez: valores reales del CSV.
        # Verificado vs Data_Noise_Data_CIFAR10_Custom.csv y Flat_Minima_Data_CIFAR10_fc1_Custom.csv
        "DataAug":       [0.785/0.785, 0.314/0.80, 0.820/0.88],
        "BatchNorm":     [0.770/0.785, 0.264/0.80, 0.741/0.88],
        "Dropout":       [0.727/0.785, 0.395/0.80, 0.884/0.88],   # p=0.525 en accuracy, Custom=0.3 en robustez
        "L2":            [0.714/0.785, 0.551/0.80, 0.550/0.88],
        "GaussianNoise": [0.697/0.785, 0.796/0.80, 0.815/0.88],
        "Baseline":      [0.694/0.785, 0.534/0.80, 0.833/0.88],
    }
    categories = ["Accuracy\n(val_acc)", "Robustez\ndatos", "Robustez\npesos (fc1)"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # cierre

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for method, values in data.items():
        vals = values + values[:1]
        ax.plot(angles, vals, lw=2, color=COLORS.get(method, "gray"),
                marker=MARKERS.get(method, "o"), ms=6, label=method)
        ax.fill(angles, vals, color=COLORS.get(method, "gray"), alpha=0.07)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.set_title("Evaluación Multidimensional de los Regularizadores\n(valores normalizados respecto al máximo observado)",
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.tight_layout()
    save(fig, "fig09_radar_multidimensional.pdf")


# ─── FIGURA 10: SVHN unique_pctg zona óptima ─────────────────────────────────

def fig10_svhn_unique_pctg():
    print("\n[Fig 10] SVHN unique_pctg (30 bins) vs val_acc")
    csv = DATA_DIR / "internal_activations" / "svhn" / "SVHN_30bins_uniquePctg.csv"
    if not csv.exists():
        print("  MISSING SVHN_30bins_uniquePctg.csv")
        return
    df = pd.read_csv(csv)
    sub = df[(df["layer"] == "fc1") & (df["split"] == "val")].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in sub["reg_method"].unique():
        m = sub[sub["reg_method"] == method]
        ax.scatter(m["unique_pctg"], m["val_acc"],
                   color=COLORS.get(method, "gray"),
                   marker=MARKERS.get(method, "o"),
                   s=60, alpha=0.75, label=method, zorder=3)

    # Zona óptima (10-30% unique_pctg en SVHN)
    ax.axvspan(10, 30, alpha=0.12, color="green", label="Zona óptima (10–30%)")
    ax.axvline(10, color="green", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(30, color="green", ls="--", lw=0.8, alpha=0.5)

    ax.set_xlabel("% Estados Únicos en fc1 (SVHN, 30 bins, validación)")
    ax.set_ylabel("val_acc")
    ax.set_title("Estados Únicos vs Accuracy de Validación\n(SVHN, 30 bins, capa fc1)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "fig10_svhn_unique_pctg_vs_acc.pdf")


# ─── FIGURA 07c: Comparativa 3 capas (bar chart) ─────────────────────────────

def fig07c_comparativa_capas(sigma_target=0.15, series="Custom"):
    """
    Grouped bar chart: retencion de accuracy a sigma_target por capa (conv1, conv3, fc1)
    para cada metodo. Ilustra la heterogeneidad por capas (especialmente L2).
    Fuente: Flat_Minima_Data_CIFAR10_{layer}_{series}.csv
    """
    print(f"\n[Fig 07c] Comparativa retencion 3 capas sigma={sigma_target} ({series})")
    layers = ["conv1", "conv3", "fc1"]
    retention = {}  # method -> [ret_conv1, ret_conv3, ret_fc1]

    for layer in layers:
        csv = DATA_DIR / "robustness_weight_noise" / f"Flat_Minima_Data_CIFAR10_{layer}_{series}.csv"
        if not csv.exists():
            print(f"  MISSING: {csv.name}")
            return
        df = pd.read_csv(csv)
        acc_col = "acc_mean" if "acc_mean" in df.columns else "acc"
        sigma_col = "sigma" if "sigma" in df.columns else "noise_sigma"

        acc0 = df[df[sigma_col] == 0.0].groupby(["method", "reg_val"])[acc_col].mean().reset_index()
        acc_s = df[abs(df[sigma_col] - sigma_target) < 1e-9].groupby(["method", "reg_val"])[acc_col].mean().reset_index()

        for _, row0 in acc0.iterrows():
            m, rv, base = row0["method"], row0["reg_val"], row0[acc_col]
            s_row = acc_s[(acc_s["method"] == m) & (acc_s["reg_val"] == rv)]
            if len(s_row) > 0:
                ret = s_row[acc_col].values[0] / base * 100
                if m not in retention:
                    retention[m] = {}
                if layer not in retention[m]:
                    retention[m][layer] = ret

    methods = sorted(retention.keys())
    x = np.arange(len(methods))
    width = 0.25
    layer_labels = ["conv1", "conv3", "fc1"]
    offsets = [-width, 0, width]
    colors_bar = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (layer, offset, color) in enumerate(zip(layer_labels, offsets, colors_bar)):
        values = [retention.get(m, {}).get(layer, 0) for m in methods]
        bars = ax.bar(x + offset, values, width, label=layer, color=color, alpha=0.82, edgecolor="white")
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Retencion de accuracy (%)", fontsize=11)
    ax.set_xlabel("Metodo", fontsize=11)
    ax.set_ylim(0, 115)
    ax.axhline(100, color="gray", ls="--", lw=0.8, alpha=0.5, label="Sin degradacion (100%)")
    ax.legend(title="Capa perturbada", fontsize=9, title_fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(
        f"Heterogeneidad de la robustez por capa (sigma={sigma_target}, {series})\n"
        f"Nota: L2 robusto en conv1 (90%) pero fragil en conv3/fc1 (~40%)",
        fontweight="bold"
    )
    fig.tight_layout()
    save(fig, f"fig07c_comparativa_capas_sigma{int(sigma_target*100)}_{series}.pdf")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Generando figuras TFG ===")
    print(f"  Data dir : {DATA_DIR}")
    print(f"  Output   : {OUT_DIR}")
    print(f"  Dry-run  : {_ARGS.dry_run}\n")

    # §5 — Resultados generales
    fig01_accuracy_curvas("CIFAR10")
    fig01_accuracy_curvas("SVHN")
    fig02_ranking_val_acc("CIFAR10")
    fig02_ranking_val_acc("SVHN")

    # §6.1.1 — Análisis de activaciones internas
    fig03_entropia_vs_acc()
    fig04_dispersion_vs_acc()

    # §6.1.2 — Evolución temporal
    fig05_temporal_unique_pctg("Custom")
    fig05_temporal_unique_pctg("Optimum")

    # §6.1.3 — Robustez datos
    fig06_robustez_datos("Custom")
    fig06_robustez_datos("Optimum")

    # §6.1.4 — Robustez pesos (conv1, conv3, fc1)
    fig07_fig08_robustez_pesos("Custom")
    fig07_fig08_robustez_pesos("Optimum")
    fig07c_comparativa_capas(sigma_target=0.15, series="Custom")  # bar chart 3 capas

    # §7 — Evaluación multidimensional
    fig09_radar_multidimensional()

    # SVHN validación cruzada
    fig10_svhn_unique_pctg()

    if not _ARGS.dry_run:
        n = len(list(OUT_DIR.glob("fig*.pdf")))
        print(f"\n=== Completado: {n} figuras en {OUT_DIR} ===")
    else:
        print("\n=== Dry-run completado — ningún archivo escrito. ===")
