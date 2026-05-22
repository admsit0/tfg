"""
fixed_dispersion_plot.py
========================
Reproduce the 2×2 val_acc vs. dispersion_ratio scatter figure with three fixes:

  1. Drop points where dispersion_ratio ≈ 0  (degenerate / collapsed activations).
     The raw CSVs are never modified.
  2. Zoom both axes to the data range of the *filtered* points (+ small padding),
     so the actual cloud shape becomes visible.
  3. Re-normalise dispersion_ratio so that the Baseline point lands exactly at
     x = 1.0, consistent with the reference line drawn at that position.
     Root cause: the CSV pre-computes dispersion_ratio against a reference that
     may not match the row selected as Baseline by add_baseline_from_zero
     (best val_acc among reg_val ≈ 0 rows), leaving Baseline at an arbitrary
     value ≠ 1.  Fix: divide all values in the column by Baseline's actual
     ratio, anchoring it to 1.0 by construction.

Usage
-----
Place next to reproduce_thesis_outputs.py (same repo root) and run:

    python fixed_dispersion_plot.py

Output → <repo_root>/generated_outputs/figures/fixed/
            combined_activation_val_acc_vs_dispersion_ratio_2x2_fixed.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "generated_outputs" / "figures" / "fixed"

DATASETS = ["CIFAR10", "SVHN", "CIFAR100", "FashionMNIST"]

# ── Style constants (identical to the original script) ─────────────────────────
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

MARKERS = {
    "Baseline":      "o",
    "DataAug":       "s",
    "BatchNorm":     "^",
    "Dropout":       "D",
    "GaussianNoise": "v",
    "EarlyStopping": "P",
    "L1":            "X",
    "L2":            "h",
}

# Threshold below which dispersion_ratio is considered degenerate/zero
ZERO_THRESHOLD = 0.05


# ── Helpers ────────────────────────────────────────────────────────────────────
def configure_style() -> None:
    plt.rcParams.update({
        "figure.dpi":              120,
        "savefig.dpi":             300,
        "font.family":             "DejaVu Sans",
        "font.size":               10,
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


def load_dispersion(dataset: str) -> pd.DataFrame:
    base = ROOT / "internal_activations" / dataset
    if dataset == "CIFAR10":
        path = base / "CIFAR10_dispersion_ratio.csv"
    elif dataset == "SVHN" and (base / "SVHN_dispersion_ratio.csv").exists():
        path = base / "SVHN_dispersion_ratio.csv"
    else:
        path = base / "dispersion_ratio_results.csv"
    return pd.read_csv(path)


def fc1_val(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[(df["layer"] == "fc1") & (df["split"] == "val")].copy()


def add_baseline_from_zero(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Baseline" in set(df.get("reg_method", [])):
        return df.copy()
    if "reg_val" not in df.columns:
        return df.copy()
    zero = df[np.isclose(df["reg_val"].astype(float), 0.0)]
    if zero.empty:
        return df.copy()
    base = zero.loc[zero["val_acc"].idxmax()].copy()
    base["reg_method"] = "Baseline"
    return pd.concat([df, pd.DataFrame([base])], ignore_index=True)


def drop_near_zero_dispersion(df: pd.DataFrame, threshold: float = ZERO_THRESHOLD) -> pd.DataFrame:
    """Remove rows whose dispersion_ratio is at or near zero (in-memory only)."""
    return df[df["dispersion_ratio"].astype(float) > threshold].copy()


def renormalize_to_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix 3 — anchor dispersion_ratio so Baseline lands exactly at x = 1.0.

    The CSV pre-computes dispersion_ratio against some reference run that may
    differ from the row chosen as Baseline by add_baseline_from_zero.  As a
    result, the Baseline scatter point can fall at x != 1.0 even though the
    vertical reference line is hardcoded at x = 1.0.

    Solution: divide every dispersion_ratio value by Baseline's actual value,
    making Baseline = 1.0 by construction and keeping all other ratios
    consistent relative to it.
    """
    baseline_rows = df[df["reg_method"] == "Baseline"]
    if baseline_rows.empty:
        return df                                    # no anchor available
    anchor = float(baseline_rows["dispersion_ratio"].iloc[0])
    if np.isclose(anchor, 0.0):
        return df                                    # avoid division by zero
    out = df.copy()
    out["dispersion_ratio"] = out["dispersion_ratio"].astype(float) / anchor
    return out


def padded_limits(values: pd.Series, pad_frac: float = 0.06) -> tuple[float, float]:
    lo, hi = float(values.min()), float(values.max())
    margin = (hi - lo) * pad_frac
    return lo - margin, hi + margin


def load_all(datasets: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for ds in datasets:
        raw = fc1_val(load_dispersion(ds))
        raw = add_baseline_from_zero(raw)
        raw = renormalize_to_baseline(raw)         # ← Fix 3 (before zero-filter)
        raw = drop_near_zero_dispersion(raw)       # ← Fix 1
        out[ds] = raw
    return out


# ── Build the legend handles (matches the original style) ─────────────────────
def build_legend_handles(methods_seen: set[str]) -> list:
    handles = []
    for m in METHOD_ORDER:
        if m not in methods_seen:
            continue
        handles.append(
            mlines.Line2D([], [],
                          color=COLORS.get(m, "#333333"),
                          marker=MARKERS.get(m, "o"),
                          linestyle="None",
                          markersize=7,
                          label=METHOD_LABELS.get(m, m))
        )
    # Baseline vertical line
    handles.append(
        mlines.Line2D([], [],
                      color="#555555",
                      linestyle="--",
                      linewidth=1.2,
                      label="Baseline (ratio = 1)")
    )
    return handles


# ── Main figure ────────────────────────────────────────────────────────────────
def generate_fixed_dispersion_figure(data: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes_flat = axes.flatten()

    methods_seen: set[str] = set()

    for ax, dataset in zip(axes_flat, DATASETS):
        df = data[dataset]
        if df.empty:
            ax.set_title(f"{dataset}  (sin datos)")
            continue

        # ── Fix 2: zoom axes to the filtered data ────────────────────────────
        x_lo, x_hi = padded_limits(df["dispersion_ratio"])
        y_lo, y_hi = padded_limits(df["val_acc"])

        for method in ordered_methods(df["reg_method"]):
            part = df[df["reg_method"] == method]
            methods_seen.add(method)
            ax.scatter(
                part["dispersion_ratio"],
                part["val_acc"],
                s=58,
                marker=MARKERS.get(method, "o"),
                facecolor=COLORS.get(method, "#333333"),
                edgecolor="black",
                linewidth=0.45,
                alpha=0.86,
                label=METHOD_LABELS.get(method, method),
                zorder=3,
            )

        # Baseline reference line (clipped to the zoomed x range)
        if 1.0 >= x_lo and 1.0 <= x_hi:
            ax.axvline(1.0, color="#555555", linestyle="--", linewidth=1.0, zorder=2)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(dataset)
        ax.set_xlabel("Ratio de dispersion en fc1")
        ax.set_ylabel("val_acc")

    # Shared legend at the bottom
    handles = build_legend_handles(methods_seen)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(
        "Dispersión de val_acc frente al ratio de dispersión en fc1\n"
        "(ratio ≈ 0 eliminados · ejes ajustados · ratio renormalizado a Baseline = 1)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "combined_activation_val_acc_vs_dispersion_ratio_2x2_fixed.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Guardado: {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    configure_style()
    data = load_all(DATASETS)
    generate_fixed_dispersion_figure(data)
    print("Listo.")