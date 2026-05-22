"""
fixed_internal_metric_correlations.py
=====================================
Creates two fixed 2x2 internal-metric figures per dataset:

1. Metric covariance/correlation:
   - entropy vs dispersion
   - entropy vs unique states
   - dispersion vs unique states
   - lower-right panel blank with a short correlation label

2. Metrics vs validation accuracy, with a strong zoom on the central/high-
   accuracy region:
   - val_acc vs entropy
   - val_acc vs unique states
   - val_acc vs dispersion
   - lower-right panel blank with a short thesis label

Each point is encoded by regularization method (marker/color). CSVs are never
modified.

Output:
generated_outputs/figures/fixed/internal_metrics/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "generated_outputs" / "figures" / "fixed" / "internal_metrics"

DATASETS = ["CIFAR10", "SVHN", "CIFAR100", "FashionMNIST"]

METHOD_ORDER = [
    "DataAug",
    "BatchNorm",
    "Dropout",
    "L2",
    "Baseline",
    "GaussianNoise",
    "EarlyStopping",
    "L1",
]

METHOD_LABELS = {
    "Baseline": "Baseline",
    "DataAug": "Aumento de datos",
    "BatchNorm": "BatchNorm",
    "Dropout": "Dropout",
    "GaussianNoise": "Ruido gaussiano",
    "EarlyStopping": "Parada temprana",
    "L1": "L1",
    "L2": "L2",
}

COLORS = {
    "Baseline": "#4D4D4D",
    "DataAug": "#D55E00",
    "BatchNorm": "#0072B2",
    "Dropout": "#009E73",
    "GaussianNoise": "#CC79A7",
    "EarlyStopping": "#E69F00",
    "L1": "#999999",
    "L2": "#56B4E9",
}

MARKERS = {
    "Baseline": "o",
    "DataAug": "s",
    "BatchNorm": "^",
    "Dropout": "D",
    "GaussianNoise": "v",
    "EarlyStopping": "P",
    "L1": "X",
    "L2": "h",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Aptos", "Calibri", "DejaVu Sans", "Arial"],
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.titleweight": "semibold",
            "axes.grid": True,
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#BDBDBD",
            "figure.constrained_layout.use": False,
        }
    )


def ordered_methods(methods) -> list[str]:
    present = list(dict.fromkeys(list(methods)))
    preferred = [m for m in METHOD_ORDER if m in present]
    remainder = sorted([m for m in present if m not in preferred])
    return preferred + remainder


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def states_path(dataset: str) -> Path:
    base = ROOT / "internal_activations" / dataset
    if dataset == "CIFAR10":
        return base / "CIFAR10_30bins_entropy_states.csv"
    if dataset == "SVHN" and (base / "SVHN_10bins_entropy_states.csv").exists():
        return base / "SVHN_10bins_entropy_states.csv"
    return base / "states_results.csv"


def dispersion_path(dataset: str) -> Path:
    base = ROOT / "internal_activations" / dataset
    if dataset == "CIFAR10":
        return base / "CIFAR10_dispersion_ratio.csv"
    if dataset == "SVHN" and (base / "SVHN_dispersion_ratio.csv").exists():
        return base / "SVHN_dispersion_ratio.csv"
    return base / "dispersion_ratio_results.csv"


def fc1_val(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["layer"] == "fc1") & (df["split"] == "val")].copy()


def add_baseline_from_zero(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Baseline" in set(df.get("reg_method", [])):
        return df.copy()
    zero = df[np.isclose(df["reg_val"].astype(float), 0.0)]
    if zero.empty:
        return df.copy()
    baseline = zero.loc[zero["val_acc"].idxmax()].copy()
    baseline["reg_method"] = "Baseline"
    return pd.concat([df, pd.DataFrame([baseline])], ignore_index=True)


def renormalize_dispersion_to_baseline(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df[df["reg_method"] == "Baseline"]
    if baseline.empty:
        return df
    anchor = float(baseline["dispersion_ratio"].iloc[0])
    if np.isclose(anchor, 0.0):
        return df
    out = df.copy()
    out["dispersion_ratio"] = out["dispersion_ratio"].astype(float) / anchor
    return out


def load_internal_metrics(dataset: str) -> pd.DataFrame:
    states = add_baseline_from_zero(fc1_val(pd.read_csv(states_path(dataset))))
    dispersion = add_baseline_from_zero(fc1_val(pd.read_csv(dispersion_path(dataset))))
    dispersion = renormalize_dispersion_to_baseline(dispersion)

    keep_state_cols = [
        "reg_method",
        "reg_val",
        "best_epoch",
        "val_acc",
        "train_acc",
        "unique_pctg",
        "entropy",
        "n_bins",
    ]
    states = states[[c for c in keep_state_cols if c in states.columns]]
    dispersion = dispersion[["reg_method", "reg_val", "dispersion_ratio"]]

    merged = states.merge(dispersion, on=["reg_method", "reg_val"], how="inner")
    merged = merged.dropna(subset=["unique_pctg", "entropy", "dispersion_ratio", "val_acc"])
    return merged.reset_index(drop=True)


def padded_limits(values: pd.Series, pad_frac: float = 0.08) -> tuple[float, float]:
    lo = float(values.min())
    hi = float(values.max())
    if np.isclose(lo, hi):
        return lo - 1.0, hi + 1.0
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


def zoom_limits(values: pd.Series, low_q: float, high_q: float, pad_frac: float = 0.10) -> tuple[float, float]:
    lo = float(values.quantile(low_q))
    hi = float(values.quantile(high_q))
    if np.isclose(lo, hi):
        lo = float(values.min())
        hi = float(values.max())
    if np.isclose(lo, hi):
        return lo - 1.0, hi + 1.0
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


def zoom_reference_points(df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    high_acc = df[df["val_acc"] >= df["val_acc"].quantile(0.40)].copy()
    if len(high_acc) < max(8, len(df) * 0.30):
        high_acc = df[df["val_acc"] >= df["val_acc"].quantile(0.25)].copy()

    x_lo = float(high_acc[x_col].quantile(0.08))
    x_hi = float(high_acc[x_col].quantile(0.92))
    central = high_acc[high_acc[x_col].between(x_lo, x_hi)].copy()
    if len(central) < max(8, len(df) * 0.22):
        return high_acc
    return central


def scatter_metric(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str, xlabel: str, ylabel: str, title: str) -> set[str]:
    seen: set[str] = set()
    for method in ordered_methods(df["reg_method"]):
        part = df[df["reg_method"] == method]
        if part.empty:
            continue
        seen.add(method)
        ax.scatter(
            part[x_col],
            part[y_col],
            s=58,
            marker=MARKERS.get(method, "o"),
            facecolor=COLORS.get(method, "#333333"),
            edgecolor="black",
            linewidth=0.45,
            alpha=0.92,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(*padded_limits(df[x_col]))
    ax.set_ylim(*padded_limits(df[y_col]))
    return seen


def scatter_metric_zoomed_accuracy(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    xlabel: str,
    title: str,
) -> set[str]:
    seen = scatter_metric(ax, df, x_col, "val_acc", xlabel, "Accuracy de validación", title)
    ref = zoom_reference_points(df, x_col)
    ax.set_xlim(*zoom_limits(ref[x_col], 0.02, 0.98, pad_frac=0.12))
    ax.set_ylim(*zoom_limits(ref["val_acc"], 0.05, 0.995, pad_frac=0.18))
    return seen


def legend_handles(methods_seen: set[str]) -> list:
    handles = []
    for method in METHOD_ORDER:
        if method not in methods_seen:
            continue
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=COLORS.get(method, "#333333"),
                marker=MARKERS.get(method, "o"),
                linestyle="None",
                markersize=7,
                label=method_label(method),
            )
        )
    return handles


def correlation_text(df: pd.DataFrame) -> str:
    return "Correlación entre\nmétricas internas"


def pearson_label(df: pd.DataFrame, left: str, right: str) -> str:
    corr = df[[left, right]].corr(method="pearson").iloc[0, 1]
    return f"r={corr:.2f}"


def covariance_title(df: pd.DataFrame, title: str, left: str, right: str) -> str:
    return f"{title} ({pearson_label(df, left, right)})"


def plot_covariance_figure(dataset: str, df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    seen: set[str] = set()

    seen |= scatter_metric(
        axes[0, 0],
        df,
        "dispersion_ratio",
        "entropy",
        "Ratio de dispersión en fc1",
        "Entropía H en fc1",
        covariance_title(df, "Entropía vs dispersión", "entropy", "dispersion_ratio"),
    )
    seen |= scatter_metric(
        axes[0, 1],
        df,
        "unique_pctg",
        "entropy",
        "Estados únicos en fc1 (%)",
        "Entropía H en fc1",
        covariance_title(df, "Entropía vs estados", "entropy", "unique_pctg"),
    )
    seen |= scatter_metric(
        axes[1, 0],
        df,
        "unique_pctg",
        "dispersion_ratio",
        "Estados únicos en fc1 (%)",
        "Ratio de dispersión en fc1",
        covariance_title(df, "Dispersión vs estados", "dispersion_ratio", "unique_pctg"),
    )

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.5,
        0.54,
        correlation_text(df),
        ha="center",
        va="center",
        fontsize=13,
        fontweight="normal",
        color="#333333",
        linespacing=1.35,
    )

    fig.legend(
        handles=legend_handles(seen),
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=8.5,
    )
    fig.suptitle(f"Comparación de covarianza entre métricas internas - {dataset}", fontsize=16, fontweight="semibold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])

    out = OUT_DIR / f"{dataset}_internal_metric_covariance_2x2_fixed.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_val_accuracy_figure(dataset: str, df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    seen: set[str] = set()

    seen |= scatter_metric_zoomed_accuracy(
        axes[0, 0],
        df,
        "entropy",
        "Entropía H en fc1",
        "Accuracy vs entropía",
    )
    seen |= scatter_metric_zoomed_accuracy(
        axes[0, 1],
        df,
        "unique_pctg",
        "Estados únicos en fc1 (%)",
        "Accuracy vs estados",
    )
    seen |= scatter_metric_zoomed_accuracy(
        axes[1, 0],
        df,
        "dispersion_ratio",
        "Ratio de dispersión en fc1",
        "Accuracy vs dispersión",
    )

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.5,
        0.54,
        "Zona intermedia\nfrente a accuracy",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="normal",
        color="#333333",
        linespacing=1.35,
    )

    fig.legend(
        handles=legend_handles(seen),
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=8.5,
    )
    fig.suptitle(f"Métricas internas frente a accuracy de validación con zoom - {dataset}", fontsize=16, fontweight="semibold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])

    out = OUT_DIR / f"{dataset}_internal_metrics_vs_val_acc_all_values_zoomed_2x2_fixed.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    written: list[Path] = []
    for dataset in DATASETS:
        df = load_internal_metrics(dataset)
        if df.empty:
            print(f"[WARN] Sin datos internos para {dataset}")
            continue
        written.append(plot_covariance_figure(dataset, df))
        written.append(plot_val_accuracy_figure(dataset, df))

    print("Generated:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
