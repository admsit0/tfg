"""
fixed_plots.py
==============
Reproduce the two 2×2 summary figures:
  - combined_ranking_val_acc_2x2_fixed.png
  - combined_generalization_gap_ranking_2x2_fixed.png

Key change vs. the originals: all four subplots inside each figure share
the SAME x-axis range (computed from the global min/max across all datasets),
so bars can be compared at a glance.

Usage
-----
Place this file next to reproduce_thesis_outputs.py (same repo root) and run:

    python fixed_plots.py

Output goes to  <repo_root>/generated_outputs/figures/fixed/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]   # same anchor as the original
OUT_DIR = ROOT / "generated_outputs" / "figures" / "fixed"

DATASETS = ["CIFAR10", "SVHN", "CIFAR100", "FashionMNIST"]

# ── Shared style constants (copy-pasted from the original) ─────────────────────
METHOD_ORDER = ["DataAug", "BatchNorm", "Dropout", "L2",
                "Baseline", "GaussianNoise", "EarlyStopping", "L1"]

METHOD_LABELS = {
    "Baseline":      "Baseline",
    "DataAug":       "Aumento de datos",
    "BatchNorm":     "BatchNorm",
    "Dropout":       "Dropout",
    "GaussianNoise": "Ruido gaussiano",
    "EarlyStopping": "Parada temprana",
    "L1":            "L1",
    "L2":            "L2",
}

COLORS = {
    "Baseline":      "#4D4D4D",
    "DataAug":       "#D55E00",
    "BatchNorm":     "#0072B2",
    "Dropout":       "#009E73",
    "GaussianNoise": "#CC79A7",
    "EarlyStopping": "#E69F00",
    "L1":            "#999999",
    "L2":            "#56B4E9",
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def configure_style() -> None:
    plt.rcParams.update({
        "figure.dpi":              120,
        "savefig.dpi":             300,
        "font.family":             "DejaVu Sans",
        "font.size":               14,
        "axes.titlesize":          13,
        "axes.labelsize":          10,
        "axes.titleweight":        "bold",
        "axes.grid":               True,
        "grid.color":              "#D8D8D8",
        "grid.linewidth":          0.7,
        "grid.alpha":              0.75,
        "axes.spines.top":         False,
        "axes.spines.right":       False,
        "figure.constrained_layout.use": False,
    })


def ordered_methods(methods) -> list[str]:
    present   = list(dict.fromkeys(list(methods)))
    preferred = [m for m in METHOD_ORDER if m in present]
    remainder = sorted([m for m in present if m not in preferred])
    return preferred + remainder


def method_label(m: str) -> str:
    return METHOD_LABELS.get(m, m)


def final_accuracy_summary(dataset: str) -> pd.DataFrame:
    """Return one best-val_acc row per regularisation method."""
    acc_dir = ROOT / "accuracy" / dataset
    rows = []
    for csv_path in sorted(acc_dir.glob("data_CNN_*.csv")):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        idx = df["val_acc"].idxmax()
        row = df.loc[idx].copy()
        method = str(row["reg_method"])
        rows.append({
            "method":    method,
            "reg_val":   row["reg_val"],
            "epoch":     int(row["epoch"]),
            "train_acc": float(row["train_acc"]),
            "val_acc":   float(row["val_acc"]),
            "gap":       float(row["train_acc"] - row["val_acc"]),
        })

    summary = pd.DataFrame(rows)

    # Ensure a Baseline entry exists (fall back to reg_val == 0 if needed)
    if "Baseline" not in set(summary.get("method", [])):
        zero_rows = []
        for csv_path in sorted(acc_dir.glob("data_CNN_*.csv")):
            df = pd.read_csv(csv_path)
            if "reg_val" not in df.columns:
                continue
            zero = df[np.isclose(df["reg_val"].astype(float), 0.0)]
            if not zero.empty:
                zero_rows.append(zero.loc[zero["val_acc"].idxmax()])
        if zero_rows:
            zero_df = pd.DataFrame(zero_rows).reset_index(drop=True)
            base    = zero_df.iloc[int(zero_df["val_acc"].idxmax())]
            summary = pd.concat([
                summary,
                pd.DataFrame([{
                    "method":    "Baseline",
                    "reg_val":   0.0,
                    "epoch":     int(base["epoch"]),
                    "train_acc": float(base["train_acc"]),
                    "val_acc":   float(base["val_acc"]),
                    "gap":       float(base["train_acc"] - base["val_acc"]),
                }])
            ], ignore_index=True)

    return summary.sort_values("val_acc", ascending=False).reset_index(drop=True)


def load_all_summaries() -> dict[str, pd.DataFrame]:
    return {ds: final_accuracy_summary(ds) for ds in DATASETS}


# ── Figure 1 – Ranking val_acc  (2×2, UNIFIED x-axis per figure) ──────────────
def plot_ranking(summaries: dict[str, pd.DataFrame]) -> None:
    # Compute global x range across all datasets
    all_vals = pd.concat([df["val_acc"] for df in summaries.values()])
    x_min = max(0.0,  all_vals.min() - 0.05)
    x_max = min(1.0,  all_vals.max() + 0.08)

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    axes_flat = axes.flatten()

    for ax, dataset in zip(axes_flat, DATASETS):
        df   = summaries[dataset].sort_values("val_acc", ascending=True)
        labels = [method_label(m) for m in df["method"]]
        colors = [COLORS.get(m, "#333333") for m in df["method"]]
        bars   = ax.barh(labels, df["val_acc"],
                         color=colors, edgecolor="black", linewidth=0.5)
        for bar, (_, row) in zip(bars, df.iterrows()):
            ax.text(
                row["val_acc"] + (x_max - x_min) * 0.008,
                bar.get_y() + bar.get_height() / 2,
                f"{row['val_acc']:.3f}",
                va="center", fontsize=9,
            )
        ax.set_title(dataset)
        ax.set_xlabel("Mejor val_acc")
        ax.set_xlim(x_min, x_max)   # ← same for all four subplots

    fig.suptitle("Ranking final de accuracy por dataset",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "combined_ranking_val_acc_2x2_fixed.png"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Guardado: {path}")


# ── Figure 2 – Generalization gap  (2×2, UNIFIED x-axis per figure) ───────────
def plot_gap(summaries: dict[str, pd.DataFrame]) -> None:
    # Compute global x range
    all_gaps = pd.concat([df["gap"] for df in summaries.values()])
    x_min = 0.0
    x_max = all_gaps.max() * 1.10   # 10 % headroom

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    axes_flat = axes.flatten()

    for ax, dataset in zip(axes_flat, DATASETS):
        # Sort ascending so largest gap is at the top (matches original)
        df     = summaries[dataset].sort_values("gap", ascending=True)
        labels = [method_label(m) for m in df["method"]]
        colors = [COLORS.get(m, "#333333") for m in df["method"]]
        ax.barh(labels, df["gap"],
                color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(dataset)
        ax.set_xlabel("train_acc - val_acc")
        ax.set_xlim(x_min, x_max)   # ← same for all four subplots

    fig.suptitle("Brecha de generalizacion por dataset",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "combined_generalization_gap_ranking_2x2_fixed.png"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Guardado: {path}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    configure_style()
    summaries = load_all_summaries()
    plot_ranking(summaries)
    plot_gap(summaries)
    print("Listo.")

    