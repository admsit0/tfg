"""
fixed_temporal_plot.py
======================
Reproduce the 2×2 temporal evolution figure (fc1 unique_pctg over epochs)
and overlay the simultaneous val_acc evolution on a twin right axis.

Design choices
--------------
* One subplot per dataset (CIFAR10 / SVHN / CIFAR100 / FashionMNIST).
* Left  axis  (primary)   : unique_pctg in fc1  — solid lines, lw 1.9
* Right axis  (secondary) : val_acc             — dashed lines, lw 1.4, alpha 0.75
* Same color per method across both axes → the eye immediately links the two curves.
* A single two-row style legend at the bottom explains the convention once.
* For each method the *best hyperparameter configuration* is selected (the one
  whose maximum val_acc is highest) so both curves always refer to the same run.

Output
------
generated_outputs/figures/fixed/
    combined_temporal_fc1_unique_pctg_2x2_fixed.png

Usage
-----
Place next to reproduce_thesis_outputs.py and run:
    python fixed_temporal_plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "generated_outputs" / "figures" / "fixed"

DATASETS = ["CIFAR10", "SVHN", "CIFAR100", "FashionMNIST"]

# ── Style constants (identical to original script) ─────────────────────────────
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

# Line styles for the two metrics (same for all methods → distinction is
# conveyed purely by solid vs dashed, not by per-method linestyle)
LS_UNIQUE = "-"    # solid  → unique_pctg (left axis)
LS_ACC    = "--"   # dashed → val_acc     (right axis)

LW_UNIQUE = 1.9
LW_ACC    = 1.4
ALPHA_ACC = 0.78
MARKER_EVERY = 5   # place a marker every N epochs to avoid clutter


# ── Helpers ────────────────────────────────────────────────────────────────────
def configure_style() -> None:
    plt.rcParams.update({
        "figure.dpi":               120,
        "savefig.dpi":              300,
        "font.family":              "DejaVu Sans",
        "font.size":                10,
        "axes.titlesize":           13,
        "axes.labelsize":           14,
        "axes.titleweight":         "bold",
        "axes.grid":                True,
        "grid.color":               "#D8D8D8",
        "grid.linewidth":           0.7,
        "grid.alpha":               0.75,
        "axes.spines.top":          False,
        "legend.frameon":           True,
        "legend.framealpha":        0.95,
        "legend.edgecolor":         "#BDBDBD",
        "figure.constrained_layout.use": False,
    })


def ordered_methods(methods) -> list[str]:
    present   = list(dict.fromkeys(list(methods)))
    preferred = [m for m in METHOD_ORDER if m in present]
    remainder = sorted([m for m in present if m not in preferred])
    return preferred + remainder


# ── Data loaders ───────────────────────────────────────────────────────────────
def load_temporal_fc1(dataset: str) -> pd.DataFrame:
    path = ROOT / "temporal_evolution" / dataset / f"Bottleneck_Data_{dataset}_Optimum.csv"
    df   = pd.read_csv(path)
    return df[df["layer"] == "fc1"].copy()


def load_best_acc_curves(dataset: str) -> pd.DataFrame:
    acc_dir = ROOT / "accuracy" / dataset
    all_dfs: list[pd.DataFrame] = []

    for csv_path in sorted(acc_dir.glob("data_CNN_*.csv")):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame(columns=["method", "epoch", "val_acc"])

    full = pd.concat(all_dfs, ignore_index=True)

    # Normalise column names (original uses reg_method)
    if "reg_method" in full.columns and "method" not in full.columns:
        full = full.rename(columns={"reg_method": "method"})

    rows: list[pd.DataFrame] = []
    for method, grp in full.groupby("method"):
        if "reg_val" in grp.columns:
            # Find the reg_val whose *best* epoch has the highest val_acc
            best_idx   = grp["val_acc"].idxmax()
            best_rval  = grp.loc[best_idx, "reg_val"]
            best_curve = grp[np.isclose(grp["reg_val"].astype(float),
                                        float(best_rval))].copy()
        else:
            best_curve = grp.copy()

        best_curve = best_curve[["method", "epoch", "val_acc"]].sort_values("epoch")
        rows.append(best_curve)

    result = pd.concat(rows, ignore_index=True)

    # Add a synthetic Baseline entry from the zero-regularisation run if absent
    if "Baseline" not in set(result["method"]):
        zero_grp = full[np.isclose(full.get("reg_val", pd.Series(dtype=float))
                                   .astype(float), 0.0)]
        if not zero_grp.empty:
            best_rval = zero_grp.loc[zero_grp["val_acc"].idxmax(), "reg_val"]
            baseline_curve = zero_grp[np.isclose(zero_grp["reg_val"].astype(float),
                                                  float(best_rval))].copy()
            baseline_curve = baseline_curve[["method", "epoch", "val_acc"]].copy()
            baseline_curve["method"] = "Baseline"
            result = pd.concat([result, baseline_curve], ignore_index=True)

    return result


# ── Per-subplot drawing ────────────────────────────────────────────────────────
def draw_subplot(
    ax_left:   plt.Axes,
    dataset:   str,
    temp_df:   pd.DataFrame,
    acc_df:    pd.DataFrame,
) -> set[str]:

    ax_right = ax_left.twinx()

    ax_right.grid(False)
    ax_right.spines["top"].set_visible(False)

    plotted: set[str] = set()
    methods = ordered_methods(temp_df["method"])

    for method in methods:
        color = COLORS.get(method, "#333333")
        marker = MARKERS.get(method, "o")

        part_u = temp_df[temp_df["method"] == method].sort_values("epoch")
        if part_u.empty:
            continue

        epochs_u = part_u["epoch"].values
        markevery_u = max(1, len(epochs_u) // MARKER_EVERY)

        ax_left.plot(
            epochs_u,
            part_u["unique_pctg"].values,
            color=color,
            linestyle=LS_UNIQUE,
            linewidth=LW_UNIQUE,
            marker=marker,
            markersize=4.2,
            markevery=markevery_u,
        )

        part_a = acc_df[acc_df["method"] == method].sort_values("epoch")
        if part_a.empty:
            continue

        epochs_a = part_a["epoch"].values
        markevery_a = max(1, len(epochs_a) // MARKER_EVERY)

        ax_right.plot(
            epochs_a,
            part_a["val_acc"].values,
            color=color,
            linestyle=LS_ACC,
            linewidth=LW_ACC,
            marker=marker,
            markersize=3.5,
            alpha=ALPHA_ACC,
            markevery=markevery_a,
        )

        plotted.add(method)

    ax_left.set_title(dataset)
    ax_left.set_xlabel("Época")
    ax_left.set_ylabel("Estados únicos en fc1 (%)")
    ax_left.set_ylim(-2, 104)

    acc_vals = acc_df[acc_df["method"].isin(plotted)]["val_acc"]
    if not acc_vals.empty:
        ax_right.set_ylim(
            max(0.0, acc_vals.min() - 0.04),
            min(1.01, acc_vals.max() + 0.04),
        )

    ax_right.set_ylabel("val_acc", fontsize=9)
    ax_right.tick_params(axis="y", labelsize=8)

    return plotted


# ── Legend construction ────────────────────────────────────────────────────────
def build_combined_legend(methods_seen: set[str]) -> list:
    method_handles = []
    for m in METHOD_ORDER:
        if m not in methods_seen:
            continue
        method_handles.append(
            mlines.Line2D([], [],
                          color=COLORS.get(m, "#333333"),
                          marker=MARKERS.get(m, "o"),
                          linestyle=LS_UNIQUE,
                          linewidth=LW_UNIQUE,
                          markersize=6,
                          label=METHOD_LABELS.get(m, m))
        )

    spacer = mlines.Line2D([], [], color="none", label="")

    solid_handle = mlines.Line2D([], [],
                                 color="#555555",
                                 linestyle=LS_UNIQUE,
                                 linewidth=LW_UNIQUE,
                                 label="— Estados únicos fc1 (%)")

    dashed_handle = mlines.Line2D([], [],
                                  color="#555555",
                                  linestyle=LS_ACC,
                                  linewidth=LW_ACC,
                                  alpha=ALPHA_ACC,
                                  label="– – val_acc (eje derecho)")

    return method_handles + [spacer, solid_handle, dashed_handle]


# ── Main figure ────────────────────────────────────────────────────────────────
def generate_fixed_temporal_figure() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes_flat = axes.flatten()

    methods_seen: set[str] = set()

    for ax, dataset in zip(axes_flat, DATASETS):
        try:
            temp_df = load_temporal_fc1(dataset)
        except FileNotFoundError as e:
            ax.set_title(f"{dataset}  (sin datos)")
            print(f"  [AVISO] {e}")
            continue

        try:
            acc_df = load_best_acc_curves(dataset)
        except FileNotFoundError as e:
            acc_df = pd.DataFrame(columns=["method", "epoch", "val_acc"])
            print(f"  [AVISO] accuracy no disponible para {dataset}: {e}")

        seen = draw_subplot(ax, dataset, temp_df, acc_df)
        methods_seen |= seen

    handles = build_combined_legend(methods_seen)

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=11,              # ← más grande
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),  # ← más pegada al gráfico
        columnspacing=1.0,
        handlelength=2.0,
        labelspacing=0.4,
        borderpad=0.6,
    )

    fig.suptitle(
        "Evolución temporal de estados únicos en fc1 y val_acc por dataset",
        fontsize=15, fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.96])  # ← menos espacio inferior

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "combined_temporal_fc1_unique_pctg_2x2_fixed.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Guardado: {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    configure_style()
    generate_fixed_temporal_figure()
    print("Listo.")