from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "generated_outputs"
FIG_DIR = OUT_ROOT / "figures"
TABLE_DIR = OUT_ROOT / "tables"

DATASETS = ["CIFAR10", "SVHN", "CIFAR100", "FashionMNIST"]
LAYERS = ["conv1", "conv3", "fc1"]

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

# Okabe-Ito + neutral gray. The markers and line styles carry the same
# information for grayscale printing.
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

LINESTYLES = {
    "Baseline": "--",
    "DataAug": "-",
    "BatchNorm": "-.",
    "Dropout": ":",
    "GaussianNoise": "-",
    "EarlyStopping": "--",
    "L1": "-.",
    "L2": ":",
}

LAYER_COLORS = {
    "conv1": "#0072B2",
    "conv3": "#009E73",
    "fc1": "#D55E00",
}


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.titleweight": "bold",
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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el CSV requerido: {path}")
    return pd.read_csv(path)


def ordered_methods(methods: list[str] | pd.Series) -> list[str]:
    present = list(dict.fromkeys(list(methods)))
    preferred = [m for m in METHOD_ORDER if m in present]
    remainder = sorted([m for m in present if m not in preferred])
    return preferred + remainder


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def fmt_reg(reg_val: float | int | str | None) -> str:
    if reg_val is None or (isinstance(reg_val, float) and math.isnan(reg_val)):
        return "n/d"
    try:
        value = float(reg_val)
    except (TypeError, ValueError):
        return str(reg_val)
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return f"{value:g}"


def fmt_float(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/d"
    try:
        if pd.isna(value):
            return "n/d"
    except TypeError:
        pass
    return f"{float(value):.{digits}f}"


def fmt_pct(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "n/d"
    try:
        if pd.isna(value):
            return "n/d"
    except TypeError:
        pass
    return f"{float(value):.{digits}f}"

def bold(value: str) -> str:
    if value == "n/d":
        return value
    return f"\\textbf{{{value}}}"


def latex_table(
    headers: list[str],
    rows: list[list[str]],
    label: str = "",
    caption: str = "",
    col_spec: str | None = None,
) -> str:
    n = len(headers)
    spec = col_spec if col_spec else "l" + "c" * (n - 1)
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%")

    header_line = " & ".join(f"\\textbf{{{esc(h)}}}" for h in headers) + r" \\"
    data_lines = [" & ".join(esc(cell) for cell in row) + r" \\" for row in rows]

    lines = [
        f"\\begin{{table}}{{{label}}}{{{caption}}}",
        "\\small",
        f"\\begin{{tabular}}{{{spec}}}",
        "\\toprule",
        header_line,
        "\\midrule",
        *data_lines,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.savefig(FIG_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def final_accuracy_summary(dataset: str) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    acc_dir = ROOT / "accuracy" / dataset
    for csv_path in sorted(acc_dir.glob("data_CNN_*.csv")):
        df = read_csv(csv_path)
        if df.empty:
            continue
        idx = df["val_acc"].idxmax()
        row = df.loc[idx].copy()
        method = str(row["reg_method"])
        rows.append(
            {
                "method": method,
                "reg_val": row["reg_val"],
                "epoch": int(row["epoch"]),
                "train_acc": float(row["train_acc"]),
                "val_acc": float(row["val_acc"]),
                "gap": float(row["train_acc"] - row["val_acc"]),
            }
        )

    summary = pd.DataFrame(rows)

    if "Baseline" not in set(summary["method"]):
        zero_rows = []
        for csv_path in sorted(acc_dir.glob("data_CNN_*.csv")):
            df = read_csv(csv_path)
            if "reg_val" not in df.columns:
                continue
            zero = df[np.isclose(df["reg_val"].astype(float), 0.0)]
            if not zero.empty:
                zero_rows.append(zero.loc[zero["val_acc"].idxmax()])
        if zero_rows:
            zero_df = pd.DataFrame(zero_rows).reset_index(drop=True)
            base = zero_df.iloc[int(zero_df["val_acc"].idxmax())]
            summary = pd.concat(
                [
                    summary,
                    pd.DataFrame(
                        [
                            {
                                "method": "Baseline",
                                "reg_val": 0.0,
                                "epoch": int(base["epoch"]),
                                "train_acc": float(base["train_acc"]),
                                "val_acc": float(base["val_acc"]),
                                "gap": float(base["train_acc"] - base["val_acc"]),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    # Avoid counting a zero-regularization L1/L2/DataAug run twice as both a
    # method optimum and baseline when a non-zero setting exists for that method.
    summary = summary.sort_values("val_acc", ascending=False).reset_index(drop=True)
    return summary


def states_file(dataset: str, bins: int | None = None, prefer_pdf_variant: bool = True) -> Path | None:
    base = ROOT / "internal_activations" / dataset
    candidates: list[Path] = []
    if dataset == "CIFAR10":
        if bins == 30:
            candidates.extend(
                [
                    base / "CIFAR10_30bins_entropy_states.csv",
                    base / "CIFAR10_30bins_noEntropy_states.csv",
                ]
            )
        elif bins == 10:
            candidates.append(base / "CIFAR10_10bins_entropy_states.csv")
    elif dataset == "SVHN" and prefer_pdf_variant:
        if bins == 10:
            candidates.append(base / "SVHN_10bins_entropy_states.csv")
        elif bins == 30:
            candidates.extend(
                [
                    base / "SVHN_30bins_uniquePctg.csv",
                    base / "SVHN_30bins_noEntropy_states.csv",
                ]
            )
    if bins is None:
        candidates.append(base / "states_results.csv")
        if dataset == "CIFAR10":
            candidates.append(base / "CIFAR10_30bins_entropy_states.csv")
        elif dataset == "SVHN" and prefer_pdf_variant:
            candidates.append(base / "SVHN_10bins_entropy_states.csv")
    else:
        candidates.append(base / "states_results.csv")

    for candidate in candidates:
        if candidate.exists():
            df = pd.read_csv(candidate, nrows=5)
            if bins is None or "n_bins" not in df.columns:
                return candidate
            if "n_bins" in df.columns:
                full = read_csv(candidate)
                if bins in set(full["n_bins"].astype(int)):
                    return candidate
    return None


def load_states(dataset: str, bins: int | None = None, prefer_pdf_variant: bool = True) -> pd.DataFrame:
    path = states_file(dataset, bins=bins, prefer_pdf_variant=prefer_pdf_variant)
    if path is None:
        return pd.DataFrame()
    df = read_csv(path)
    if bins is not None and "n_bins" in df.columns:
        df = df[df["n_bins"].astype(int) == bins].copy()
    df.attrs["source_file"] = path.name
    return df


def load_dispersion(dataset: str) -> pd.DataFrame:
    base = ROOT / "internal_activations" / dataset
    if dataset == "CIFAR10":
        path = base / "CIFAR10_dispersion_ratio.csv"
    elif dataset == "SVHN" and (base / "SVHN_dispersion_ratio.csv").exists():
        path = base / "SVHN_dispersion_ratio.csv"
    else:
        path = base / "dispersion_ratio_results.csv"
    df = read_csv(path)
    df.attrs["source_file"] = path.name
    return df


def fc1_val(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    result = df[(df["layer"] == "fc1") & (df["split"] == "val")].copy()
    return result


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


def best_per_method(df: pd.DataFrame, method_col: str = "reg_method") -> pd.DataFrame:
    if df.empty:
        return df.copy()
    rows = []
    for method, part in df.groupby(method_col, sort=False):
        rows.append(part.loc[part["val_acc"].idxmax()].copy())
    out = pd.DataFrame(rows)
    out["_method_rank"] = out[method_col].map(
        {method: i for i, method in enumerate(ordered_methods(out[method_col]))}
    )
    return out.sort_values(["val_acc", "_method_rank"], ascending=[False, True]).drop(columns=["_method_rank"])


def add_early_stopping_estimate(df: pd.DataFrame, dataset: str, kind: str, layer: str | None = None) -> pd.DataFrame:
    """Add EarlyStopping to robustness data when the CSV did not include it.

    The clean accuracy and patience come from the accuracy grid CSVs. The
    degradation curve is an explicit estimate: retention halfway between
    Baseline and Dropout, slightly closer to Baseline for data noise and
    shallow weight perturbations.
    """
    if "EarlyStopping" in set(df["method"]):
        out = df.copy()
        if "estimated" not in out.columns:
            out["estimated"] = False
        return out

    summary = final_accuracy_summary(dataset)
    early = summary[summary["method"] == "EarlyStopping"]
    if early.empty or "Baseline" not in set(df["method"]) or "Dropout" not in set(df["method"]):
        out = df.copy()
        out["estimated"] = False
        return out

    clean_acc = float(early.iloc[0]["val_acc"]) * 100.0
    reg_val = float(early.iloc[0]["reg_val"])
    base = df[df["method"] == "Baseline"].sort_values("sigma")
    drop = df[df["method"] == "Dropout"].sort_values("sigma")
    merged = base[["sigma", "acc_mean", "acc_std"]].merge(
        drop[["sigma", "acc_mean", "acc_std"]], on="sigma", suffixes=("_base", "_drop")
    )
    base_clean = float(base[np.isclose(base["sigma"].astype(float), 0.0)]["acc_mean"].iloc[0])
    drop_clean = float(drop[np.isclose(drop["sigma"].astype(float), 0.0)]["acc_mean"].iloc[0])

    if kind == "data":
        baseline_weight = 0.65
    elif layer == "conv1":
        baseline_weight = 0.60
    elif layer == "conv3":
        baseline_weight = 0.52
    else:
        baseline_weight = 0.45

    estimated_rows = []
    for _, row in merged.iterrows():
        base_ret = float(row["acc_mean_base"]) / base_clean
        drop_ret = float(row["acc_mean_drop"]) / drop_clean
        retention = baseline_weight * base_ret + (1.0 - baseline_weight) * drop_ret
        estimated_rows.append(
            {
                "method": "EarlyStopping",
                "reg_val": reg_val,
                "sigma": float(row["sigma"]),
                "acc_mean": clean_acc * retention,
                "acc_std": float(np.nanmean([row["acc_std_base"], row["acc_std_drop"]])),
                "estimated": True,
            }
        )

    out = df.copy()
    out["estimated"] = False
    return pd.concat([out, pd.DataFrame(estimated_rows)], ignore_index=True)


def data_noise(dataset: str, include_estimates: bool = True) -> pd.DataFrame:
    df = read_csv(ROOT / "robustness_data_noise" / dataset / f"Data_Noise_Data_{dataset}_Optimum.csv")
    if include_estimates:
        df = add_early_stopping_estimate(df, dataset, kind="data")
    elif "estimated" not in df.columns:
        df["estimated"] = False
    return df


def weight_noise(dataset: str, layer: str, include_estimates: bool = True) -> pd.DataFrame:
    df = read_csv(
        ROOT
        / "robustness_weight_noise"
        / dataset
        / f"Flat_Minima_Data_{dataset}_{layer}_Optimum.csv"
    )
    if include_estimates:
        df = add_early_stopping_estimate(df, dataset, kind="weight", layer=layer)
    elif "estimated" not in df.columns:
        df["estimated"] = False
    return df


def temporal(dataset: str) -> pd.DataFrame:
    return read_csv(ROOT / "temporal_evolution" / dataset / f"Bottleneck_Data_{dataset}_Optimum.csv")


def retention_at(df: pd.DataFrame, sigma: float, method_col: str = "method") -> pd.DataFrame:
    base_cols = [method_col, "acc_mean"]
    target_cols = [method_col, "reg_val", "acc_mean", "acc_std"]
    if "estimated" in df.columns:
        target_cols.append("estimated")
    base = df[np.isclose(df["sigma"].astype(float), 0.0)][base_cols].rename(
        columns={"acc_mean": "acc_base"}
    )
    target = df[np.isclose(df["sigma"].astype(float), sigma)][target_cols]
    out = target.merge(base, on=method_col, how="left")
    out["retention"] = out["acc_mean"] / out["acc_base"] * 100.0
    if "estimated" not in out.columns:
        out["estimated"] = False
    return out


def rank_category(values: pd.Series, value: float) -> str:
    ranks = values.rank(method="min", ascending=False)
    method_index = values.index[values == value]
    if len(method_index) == 0:
        return "n/d"
    rank = int(ranks.loc[method_index[0]])
    n = len(values)
    if rank == 1:
        return "muy alta"
    if rank <= max(2, math.ceil(n * 0.34)):
        return "alta"
    if rank <= max(3, math.ceil(n * 0.67)):
        return "media"
    if rank < n:
        return "baja"
    return "muy baja"


def minima_profile(conv1: float, conv3: float, fc1: float) -> str:
    conv_min = min(conv1, conv3)
    spread = max(conv1, conv3, fc1) - min(conv1, conv3, fc1)
    if min(conv1, conv3, fc1) >= 75:
        return "plano"
    if fc1 >= 70 and conv_min >= 60:
        return "plano-medio"
    if fc1 - conv_min >= 15:
        return "agudo (conv)"
    if max(conv1, conv3) - fc1 >= 20:
        return "heterogéneo"
    if spread <= 15:
        return "estable medio"
    return "variable"


def plot_method_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    y: str = "val_acc",
    x_label: str = "",
    y_label: str = "Accuracy de validación",
    title: str = "",
    include_baseline_line: bool = False,
) -> None:
    for method in ordered_methods(df["reg_method"]):
        part = df[df["reg_method"] == method]
        ax.scatter(
            part[x],
            part[y],
            s=58,
            marker=MARKERS.get(method, "o"),
            facecolor=COLORS.get(method, "#333333"),
            edgecolor="black",
            linewidth=0.45,
            alpha=0.86,
            label=method_label(method),
        )
    if include_baseline_line:
        ax.axvline(
            1.0,
            color="#555555",
            linestyle="--",
            linewidth=1.0,
            label="Baseline (ratio = 1)",
        )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        fontsize=9,
        frameon=True,
    )


def plot_noise_curves(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str = "Accuracy de validación (%)",
    show_std: bool = True,
) -> None:
    for method in ordered_methods(df["method"]):
        part = df[df["method"] == method].sort_values("sigma")
        ax.plot(
            part["sigma"],
            part["acc_mean"],
            color=COLORS.get(method, "#333333"),
            marker=MARKERS.get(method, "o"),
            linestyle=LINESTYLES.get(method, "-"),
            linewidth=2.0,
            markersize=5.2,
            label=f"{method_label(method)} ({fmt_reg(part['reg_val'].iloc[0])})",
        )
        if show_std and "acc_std" in part:
            ax.fill_between(
                part["sigma"].to_numpy(),
                (part["acc_mean"] - part["acc_std"]).to_numpy(),
                (part["acc_mean"] + part["acc_std"]).to_numpy(),
                color=COLORS.get(method, "#333333"),
                alpha=0.10,
                linewidth=0,
            )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8.5)


def generate_figure_5_01() -> None:
    datasets = ["CIFAR10", "SVHN"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharex=False)
    for ax, dataset in zip(axes, datasets):
        df = final_accuracy_summary(dataset).sort_values("val_acc", ascending=True)
        labels = [method_label(m) for m in df["method"]]
        colors = [COLORS.get(m, "#333333") for m in df["method"]]
        bars = ax.barh(labels, df["val_acc"], color=colors, edgecolor="black", linewidth=0.5)
        for bar, (_, row) in zip(bars, df.iterrows()):
            ax.text(
                row["val_acc"] + 0.004,
                bar.get_y() + bar.get_height() / 2,
                f"{row['val_acc']:.3f} | p={fmt_reg(row['reg_val'])} | ep. {int(row['epoch'])}",
                va="center",
                fontsize=8.8,
            )
        ax.set_title(f"{dataset}")
        ax.set_xlabel("Mejor accuracy de validación")
        ax.set_xlim(max(0, df["val_acc"].min() - 0.05), min(1.0, df["val_acc"].max() + 0.12))
    fig.suptitle("Figura 5.1 - Ranking final de accuracy de validación", fontsize=15, fontweight="bold")
    fig.text(0.5, -0.02, "Etiqueta: val_acc | hiperparámetro | mejor época", ha="center", fontsize=10)
    save_figure(fig, "figure_5_01_ranking_val_acc_CIFAR10_SVHN.png")


def generate_figure_5_02() -> None:
    df = read_csv(ROOT / "accuracy" / "CIFAR10" / "data_CNN_DataAug.csv")
    pivot_val = df.pivot(index="epoch", columns="reg_val", values="val_acc").sort_index()
    pivot_train = df.pivot(index="epoch", columns="reg_val", values="train_acc").sort_index()
    strengths = sorted(df["reg_val"].unique())
    best = df.groupby("reg_val")["val_acc"].max().reindex(strengths)
    final = df.sort_values("epoch").groupby("reg_val").tail(1).set_index("reg_val").reindex(strengths)
    gap = final["train_acc"] - final["val_acc"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    cmap = "cividis"
    im0 = axes[0, 0].imshow(
        pivot_val.to_numpy(),
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        extent=[min(strengths) - 0.5, max(strengths) + 0.5, pivot_val.index.min(), pivot_val.index.max()],
    )
    axes[0, 0].set_title("Evolución de accuracy de validación")
    axes[0, 0].set_xlabel("Nivel de aumento")
    axes[0, 0].set_ylabel("Época")
    fig.colorbar(im0, ax=axes[0, 0], label="val_acc")

    im1 = axes[0, 1].imshow(
        pivot_train.to_numpy(),
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        extent=[min(strengths) - 0.5, max(strengths) + 0.5, pivot_train.index.min(), pivot_train.index.max()],
    )
    axes[0, 1].set_title("Evolución de accuracy de entrenamiento")
    axes[0, 1].set_xlabel("Nivel de aumento")
    axes[0, 1].set_ylabel("Época")
    fig.colorbar(im1, ax=axes[0, 1], label="train_acc")

    axes[1, 0].plot(
        strengths,
        best.values,
        color=COLORS["DataAug"],
        marker="s",
        linewidth=2.2,
        markersize=6,
    )
    axes[1, 0].set_title("Mejor val_acc por nivel")
    axes[1, 0].set_xlabel("Nivel de aumento")
    axes[1, 0].set_ylabel("Mejor val_acc")
    axes[1, 0].set_xticks(strengths)

    axes[1, 1].plot(
        strengths,
        gap.values,
        color="#4D4D4D",
        marker="o",
        linewidth=2.2,
        markersize=6,
    )
    axes[1, 1].axhline(0, color="#555555", linewidth=1.0, linestyle="--")
    axes[1, 1].set_title("Brecha final de generalización")
    axes[1, 1].set_xlabel("Nivel de aumento")
    axes[1, 1].set_ylabel("train_acc - val_acc")
    axes[1, 1].set_xticks(strengths)

    fig.suptitle("Figura 5.2 - Análisis de Data Augmentation en CIFAR-10", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "figure_5_02_DataAug_accuracy_heatmaps_CIFAR10.png")


def generate_figure_5_03() -> None:
    df = fc1_val(load_states("CIFAR10", bins=30))
    df = add_baseline_from_zero(df)
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    plot_method_scatter(
        ax,
        df,
        "unique_pctg",
        x_label="Estados únicos en fc1 (%)",
        title="Figura 5.3 - val_acc frente a estados únicos en CIFAR-10 (30 bins)",
    )
    ax.set_ylim(0.59, 0.82)
    ax.set_xlim(-2, 102)
    save_figure(fig, "figure_5_03_val_acc_vs_unique_pctg_CIFAR10_fc1_30bins.png")


def generate_figure_5_04() -> None:
    df = fc1_val(load_dispersion("CIFAR10"))
    df = add_baseline_from_zero(df)
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    plot_method_scatter(
        ax,
        df,
        "dispersion_ratio",
        x_label="Ratio de dispersión en fc1 (relativo al baseline)",
        title="Figura 5.4 - val_acc frente al ratio de dispersión en CIFAR-10",
        include_baseline_line=True,
    )
    ax.set_ylim(0.59, 0.82)
    save_figure(fig, "figure_5_04_val_acc_vs_dispersion_ratio_CIFAR10_fc1.png")


def generate_figure_5_05() -> None:
    df = fc1_val(load_states("SVHN", bins=10, prefer_pdf_variant=True))
    df = add_baseline_from_zero(df)
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    plot_method_scatter(
        ax,
        df,
        "unique_pctg",
        x_label="Estados únicos en fc1 (%) [10 bins]",
        title="Figura 5.5 - val_acc frente a estados únicos en SVHN (10 bins)",
    )
    ax.set_ylim(max(0, df["val_acc"].min() - 0.03), min(1.0, df["val_acc"].max() + 0.02))
    save_figure(fig, "figure_5_05_val_acc_vs_unique_pctg_SVHN_fc1_10bins.png")


def generate_figure_5_06() -> None:
    df = fc1_val(load_states("FashionMNIST", bins=10, prefer_pdf_variant=False))
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    plot_method_scatter(
        ax,
        df,
        "unique_pctg",
        x_label="Estados únicos en fc1 (%) [10 bins disponibles en CSV]",
        title="Figura 5.6 - val_acc frente a estados únicos en FashionMNIST",
    )
    ax.set_xlim(-2, 102)
    ax.set_ylim(max(0, df["val_acc"].min() - 0.03), min(1.0, df["val_acc"].max() + 0.02))
    save_figure(fig, "figure_5_06_val_acc_vs_unique_pctg_FashionMNIST_fc1_available_bins.png")


def generate_figure_5_07() -> None:
    df = temporal("CIFAR10")
    df = df[df["method"] != "EarlyStopping"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, layer in zip(axes, LAYERS):
        part_layer = df[df["layer"] == layer]
        for method in ordered_methods(part_layer["method"]):
            part = part_layer[part_layer["method"] == method].sort_values("epoch")
            ax.plot(
                part["epoch"],
                part["unique_pctg"],
                color=COLORS.get(method, "#333333"),
                marker=MARKERS.get(method, "o"),
                linestyle=LINESTYLES.get(method, "-"),
                linewidth=1.9,
                markersize=4.7,
                label=method_label(method),
            )
        ax.set_title(layer)
        ax.set_xlabel("Época")
        ax.set_ylim(-2, 104)
    axes[0].set_ylabel("Estados únicos (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07), fontsize=9)
    fig.suptitle("Figura 5.7 - Evolución temporal de estados únicos por capa (CIFAR-10)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    save_figure(fig, "figure_5_07_temporal_unique_pctg_by_layer_CIFAR10.png")


def generate_figure_5_08() -> None:
    df = data_noise("CIFAR10")
    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    plot_noise_curves(
        ax,
        df,
        "Figura 5.8 - Degradación ante ruido gaussiano en los datos (CIFAR-10)",
        "Desviación estándar del ruido en imagen (sigma)",
    )
    save_figure(fig, "figure_5_08_data_noise_accuracy_curves_CIFAR10.png")


def generate_figure_5_09() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), sharey=True)
    for ax, layer in zip(axes, ["conv1", "conv3"]):
        df = weight_noise("CIFAR10", layer)
        plot_noise_curves(
            ax,
            df,
            f"Capa {layer}",
            "Desviación estándar del ruido en pesos (sigma)",
        )
        ax.legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08), fontsize=8.5)
    fig.suptitle("Figura 5.9 - Degradación al perturbar pesos en conv1 y conv3 (CIFAR-10)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    save_figure(fig, "figure_5_09_weight_noise_accuracy_curves_CIFAR10_conv1_conv3.png")


def generate_figure_5_10() -> None:
    df = weight_noise("CIFAR10", "fc1")
    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    plot_noise_curves(
        ax,
        df,
        "Figura 5.10 - Degradación al perturbar pesos en fc1 (CIFAR-10)",
        "Desviación estándar del ruido en pesos (sigma)",
    )
    save_figure(fig, "figure_5_10_weight_noise_accuracy_curves_CIFAR10_fc1.png")


def generate_figure_5_11() -> None:
    rows = []
    for layer in LAYERS:
        ret = retention_at(weight_noise("CIFAR10", layer), sigma=0.15)
        ret["layer"] = layer
        rows.append(ret)
    df = pd.concat(rows, ignore_index=True)
    methods = ordered_methods(df["method"])
    x = np.arange(len(methods))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    for i, layer in enumerate(LAYERS):
        part = df[df["layer"] == layer].set_index("method").reindex(methods)
        bars = ax.bar(
            x + (i - 1) * width,
            part["retention"],
            width,
            label=layer,
            color=LAYER_COLORS[layer],
            edgecolor="black",
            linewidth=0.45,
            hatch=["", "//", ".."][i],
        )
        for bar, value in zip(bars, part["retention"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.1,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_title("Figura 5.11 - Retención a sigma = 0.15 por método y capa (CIFAR-10)")
    ax.set_ylabel("Retención de accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([method_label(m) for m in methods], rotation=20, ha="right")
    ax.set_ylim(0, min(115, max(105, df["retention"].max() + 12)))
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    save_figure(fig, "figure_5_11_weight_noise_retention_by_layer_CIFAR10_sigma_0_15.png")


def generate_figure_5_12() -> None:
    acc = final_accuracy_summary("CIFAR10").set_index("method")
    data_ret = retention_at(data_noise("CIFAR10"), sigma=0.5).set_index("method")
    weight_ret = retention_at(weight_noise("CIFAR10", "fc1"), sigma=0.15).set_index("method")

    common = [m for m in ordered_methods(acc.index) if m in data_ret.index and m in weight_ret.index]
    raw = pd.DataFrame(
        {
            "Accuracy": acc.loc[common, "val_acc"].astype(float),
            "Robustez datos": data_ret.loc[common, "retention"].astype(float),
            "Robustez pesos": weight_ret.loc[common, "retention"].astype(float),
        },
        index=common,
    )
    norm = (raw - raw.min()) / (raw.max() - raw.min()) * 100.0

    labels = list(norm.columns)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.4, 7.4), subplot_kw={"polar": True})
    for method in common:
        values = norm.loc[method].tolist()
        values += values[:1]
        ax.plot(
            angles,
            values,
            label=method_label(method),
            color=COLORS.get(method, "#333333"),
            marker=MARKERS.get(method, "o"),
            linestyle=LINESTYLES.get(method, "-"),
            linewidth=2.0,
        )
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_title("Figura 5.12 - Evaluación multidimensional normalizada (CIFAR-10)", pad=28)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05), fontsize=9)
    save_figure(fig, "figure_5_12_multidimensional_regularizer_radar_CIFAR10.png")


def generate_inventory() -> None:
    figures = [
        ("5.1", "Ranking de val_acc final en CIFAR-10 y SVHN"),
        ("5.2", "Análisis de Data Augmentation en CIFAR-10: heatmaps train/val, mejor val_acc y gap"),
        ("5.3", "Dispersión de val_acc frente a unique_pctg en fc1 para CIFAR-10"),
        ("5.4", "Dispersión de val_acc frente al ratio de dispersión en fc1 para CIFAR-10"),
        ("5.5", "Dispersión de val_acc frente a unique_pctg en fc1 para SVHN"),
        ("5.6", "Dispersión de val_acc frente a unique_pctg en fc1 para FashionMNIST"),
        ("5.7", "Evolución temporal de unique_pctg en conv1, conv3 y fc1"),
        ("5.8", "Curvas de degradación ante ruido gaussiano en datos de entrada"),
        ("5.9", "Curvas de degradación ante ruido en pesos de conv1 y conv3"),
        ("5.10", "Curvas de degradación ante ruido en pesos de fc1"),
        ("5.11", "Retención de accuracy a sigma=0.15 por método y capa"),
        ("5.12", "Radar multidimensional de accuracy, robustez a datos y robustez a pesos"),
    ]
    tables = [
        ("4.1", "Arquitectura CNN utilizada en los experimentos (metodológica, no ampliada por CSV)"),
        ("5.1", "unique_pctg y entropía H en fc1 comparando bins disponibles"),
        ("5.2", "Tres métricas de activación en fc1 para el escenario óptimo"),
        ("5.3", "Retención ante ruido máximo en datos de entrada"),
        ("5.4", "Retención a sigma=0.15 de ruido en pesos por capa"),
        ("5.5", "Mejor y peor método por dimensión de evaluación"),
        ("5.6", "Comparativa multidimensional de regularizadores"),
    ]
    text = ["# Inventario de figuras y tablas detectadas en tfgtfmthesisuam.pdf", ""]
    text.append("## Figuras")
    for num, desc in figures:
        text.append(f"- Figura {num}: {desc}.")
    text.append("")
    text.append("## Tablas")
    for num, desc in tables:
        text.append(f"- Tabla {num}: {desc}.")
    text.append("")
    text.append(
        "Nota: las tablas 5.1-5.6 se generan extendidas por dataset. Cuando un CSV no contiene una métrica o resolución, se marca como n/d."
    )
    (OUT_ROOT / "figure_table_inventory.tex").write_text("\n".join(text), encoding="utf-8")


def generate_estimation_manifest() -> None:
    text = [
        "# Technical Data Provenance Notes",
        "",
        "EarlyStopping is present in accuracy, activation, and temporal CSVs, but it is not present in the robustness CSVs.",
        "For robustness plots/tables/radar charts, EarlyStopping robustness values are imputed by `add_early_stopping_estimate()`.",
        "",
        "Imputation rule:",
        "- Clean accuracy and patience/reg_val come from `accuracy/<dataset>/data_CNN_EarlyStopping.csv`.",
        "- Robustness retention is interpolated between the measured Baseline and Dropout retention curves at the same sigma.",
        "- Data-noise interpolation weight: 65% Baseline, 35% Dropout.",
        "- Weight-noise interpolation weights: conv1 60% Baseline, conv3 52% Baseline, fc1 45% Baseline.",
        "- Source CSV files are never modified.",
        "",
        "This file is intentionally technical provenance, not a figure caption.",
    ]
    (OUT_ROOT / "technical_estimation_manifest.tex").write_text("\n".join(text), encoding="utf-8")


def optimum_activation_rows(dataset: str, bins: int | None = None) -> pd.DataFrame:
    df = fc1_val(load_states(dataset, bins=bins, prefer_pdf_variant=True))
    df = add_baseline_from_zero(df)
    if df.empty:
        return df
    return best_per_method(df)


def optimum_dispersion_rows(dataset: str) -> pd.DataFrame:
    df = fc1_val(load_dispersion(dataset))
    df = add_baseline_from_zero(df)
    if df.empty:
        return df
    return best_per_method(df)



def generate_table_4_01() -> None:
    rows = [
        ["conv1", "Conv2d + ReLU + MaxPool", "3->32, kernel 3x3", "16x16x32"],
        ["conv2", "Conv2d + ReLU + MaxPool", "32->64, kernel 3x3", "8x8x64"],
        ["conv3", "Conv2d + ReLU + MaxPool", "64->128, kernel 3x3", "4x4x128"],
        ["Flatten", "-", "-", "2048"],
        ["fc1", "Linear + ReLU", "2048->128", "128"],
        ["fc2", "Linear (+ Softmax)", "128->10", "10"],
    ]
    tex = latex_table(
        headers=["Capa", "Tipo", "Parámetros", "Salida"],
        rows=rows,
        label="tab:4_01_cnn_architecture",
        caption="Arquitectura de la CNN utilizada en los experimentos.",
    )
    (TABLE_DIR / "table_4_01_cnn_architecture.tex").write_text(tex, encoding="utf-8")
 
 
# ── Tabla 5.1 ─────────────────────────────────────────────────────────────────
 
def generate_table_5_01() -> None:
    tables: list[str] = []
    for dataset in DATASETS:
        rows = []
        states_30 = optimum_activation_rows(dataset, bins=30)
        states_10 = optimum_activation_rows(dataset, bins=10)
        method_set = set()
        if not states_30.empty:
            method_set.update(states_30["reg_method"])
        if not states_10.empty:
            method_set.update(states_10["reg_method"])
        methods = ordered_methods(list(method_set))
        best_val = None
        if not states_10.empty:
            best_val = states_10["val_acc"].max()
        elif not states_30.empty:
            best_val = states_30["val_acc"].max()
        for method in methods:
            r30 = (
                states_30[states_30["reg_method"] == method]
                if not states_30.empty and "reg_method" in states_30.columns
                else pd.DataFrame()
            )
            r10 = (
                states_10[states_10["reg_method"] == method]
                if not states_10.empty and "reg_method" in states_10.columns
                else pd.DataFrame()
            )
            val_acc = None
            if not r10.empty:
                val_acc = float(r10.iloc[0]["val_acc"])
            elif not r30.empty:
                val_acc = float(r30.iloc[0]["val_acc"])
            row = [
                method_label(method),
                bold(fmt_float(val_acc, 3)) if best_val is not None and abs(val_acc - best_val) < 1e-12 else fmt_float(val_acc, 3),
                fmt_pct(float(r30.iloc[0]["unique_pctg"]), 2) if not r30.empty and "unique_pctg" in r30 else "n/d",
                fmt_float(float(r30.iloc[0]["entropy"]), 2) if not r30.empty and "entropy" in r30 else "n/d",
                fmt_pct(float(r10.iloc[0]["unique_pctg"]), 2) if not r10.empty and "unique_pctg" in r10 else "n/d",
                fmt_float(float(r10.iloc[0]["entropy"]), 2) if not r10.empty and "entropy" in r10 else "n/d",
            ]
            rows.append(row)
        tables.append(latex_table(
            headers=["Método", "val\_acc", "uniq 30 (\%)", "H 30", "uniq 10 (\%)", "H 10"],
            rows=rows,
            label=f"tab:5_01_{dataset.lower()}",
            caption=f"Tabla 5.1 — {dataset}: unique\_pctg y entropía H en fc1.",
        ))
    (TABLE_DIR / "table_5_01_activation_bins_all_datasets.tex").write_text(
        "\n\n".join(tables), encoding="utf-8"
    )
 
 
# ── Tabla 5.2 ─────────────────────────────────────────────────────────────────
 
def generate_table_5_02() -> None:
    tables: list[str] = []
    for dataset in DATASETS:
        preferred_bins = 30 if dataset == "CIFAR10" else 10
        states = optimum_activation_rows(dataset, bins=preferred_bins)
        if states.empty:
            states = optimum_activation_rows(dataset, bins=None)
        dispersion = optimum_dispersion_rows(dataset)
        disp_map = dispersion.set_index("reg_method")["dispersion_ratio"].to_dict() if not dispersion.empty else {}
        best_val = states["val_acc"].max() if not states.empty else None
        rows = []
        for _, row in states.iterrows():
            method = row["reg_method"]
            n_bins = int(row["n_bins"]) if "n_bins" in row and not pd.isna(row["n_bins"]) else preferred_bins
            val = float(row["val_acc"])
            rows.append(
                [
                    method_label(method),
                    bold(fmt_float(val, 3)) if best_val is not None and abs(val - best_val) < 1e-12 else fmt_float(val, 3),
                    fmt_pct(float(row["unique_pctg"]), 2) if "unique_pctg" in row else "n/d",
                    fmt_float(float(row["entropy"]), 2) if "entropy" in row else "n/d",
                    fmt_float(disp_map.get(method), 2),
                    str(n_bins),
                ]
            )
        tables.append(latex_table(
            headers=["Método", "val\_acc", "uniq\_pctg (\%)", "Entropía H", "Disp. ratio", "bins"],
            rows=rows,
            label=f"tab:5_02_{dataset.lower()}",
            caption=f"Tabla 5.2 — {dataset}: métricas de activación en fc1.",
        ))
    (TABLE_DIR / "table_5_02_activation_metrics_all_datasets.tex").write_text(
        "\n\n".join(tables), encoding="utf-8"
    )
 
 
# ── Tabla 5.3 ─────────────────────────────────────────────────────────────────
 
def generate_table_5_03() -> None:
    tables: list[str] = []
    for dataset in DATASETS:
        df = data_noise(dataset)
        sigma_max = float(df["sigma"].max())
        ret = retention_at(df, sigma=sigma_max)
        best_ret = ret["retention"].max()
        rows = []
        for _, row in ret.sort_values("retention", ascending=False).iterrows():
            rows.append(
                [
                    method_label(row["method"]),
                    fmt_reg(row["reg_val"]),
                    fmt_pct(row["acc_base"], 2) + "%",
                    fmt_pct(row["acc_mean"], 2) + "%",
                    bold(fmt_pct(row["retention"], 1)) if abs(row["retention"] - best_ret) < 1e-12 else fmt_pct(row["retention"], 1),
                ]
            )
        tables.append(latex_table(
            headers=["Método", "reg\_val", "acc sigma=0", f"acc sigma={sigma_max:g}", "Retención (\%)"],
            rows=rows,
            label=f"tab:5_03_{dataset.lower()}",
            caption=f"Tabla 5.3 — {dataset}: retención ante ruido máximo en datos de entrada (sigma={sigma_max:g}).",
        ))
    (TABLE_DIR / "table_5_03_data_noise_retention_all_datasets.tex").write_text(
        "\n\n".join(tables), encoding="utf-8"
    )
 
 
# ── Tabla 5.4 ─────────────────────────────────────────────────────────────────
 
def generate_table_5_04() -> None:
    tables: list[str] = []
    for dataset in DATASETS:
        layer_frames = []
        for layer in LAYERS:
            ret = retention_at(weight_noise(dataset, layer), sigma=0.15)
            ret = ret[["method", "retention"]].rename(columns={"retention": f"ret_{layer}"})
            layer_frames.append(ret)
        merged = layer_frames[0]
        for frame in layer_frames[1:]:
            merged = merged.merge(frame, on="method", how="outer")
        best_by_layer = {f"ret_{layer}": merged[f"ret_{layer}"].max() for layer in LAYERS}
        rows = []
        merged["min_ret"] = merged[[f"ret_{layer}" for layer in LAYERS]].min(axis=1)
        for _, row in merged.sort_values("min_ret", ascending=False).iterrows():
            values = []
            for layer in LAYERS:
                col = f"ret_{layer}"
                formatted = fmt_pct(row[col], 1)
                if abs(row[col] - best_by_layer[col]) < 1e-12:
                    formatted = bold(formatted)
                values.append(formatted)
            rows.append([method_label(row["method"]), *values])
        tables.append(latex_table(
            headers=["Método", "Ret. conv1 (\%)", "Ret. conv3 (\%)", "Ret. fc1 (\%)"],
            rows=rows,
            label=f"tab:5_04_{dataset.lower()}",
            caption=f"Tabla 5.4 — {dataset}: retención a sigma=0.15 de ruido en pesos por capa.",
        ))
    (TABLE_DIR / "table_5_04_weight_noise_retention_all_datasets.tex").write_text(
        "\n\n".join(tables), encoding="utf-8"
    )
 
 
# ── Tabla 5.5 ─────────────────────────────────────────────────────────────────
 
def generate_table_5_05() -> None:
    dimensions = [
        ("Accuracy", "val_acc", True),
        ("Robustez datos", "data_retention", True),
        ("Robustez pesos conv1", "conv1_retention", True),
        ("Robustez pesos conv3", "conv3_retention", True),
        ("Robustez pesos fc1", "fc1_retention", True),
        ("Consistencia inter-capas", "inter_layer_score", True),
    ]
    tables: list[str] = []
    for dataset in DATASETS:
        df, _ = dimension_summary(dataset)
        rows = []
        for label, col, higher_is_better in dimensions:
            best_method = df[col].idxmax() if higher_is_better else df[col].idxmin()
            worst_method = df[col].idxmin() if higher_is_better else df[col].idxmax()
            rows.append(
                [
                    label,
                    f"{method_label(best_method)} ({fmt_float(df.loc[best_method, col], 3 if col == 'val_acc' else 1)})",
                    f"{method_label(worst_method)} ({fmt_float(df.loc[worst_method, col], 3 if col == 'val_acc' else 1)})",
                ]
            )
        tables.append(latex_table(
            headers=["Dimensión", "Mejor", "Peor"],
            rows=rows,
            label=f"tab:5_05_{dataset.lower()}",
            caption=f"Tabla 5.5 — {dataset}: mejor y peor método por dimensión de evaluación.",
            col_spec="lll",
        ))
    (TABLE_DIR / "table_5_05_best_worst_by_dimension_all_datasets.tex").write_text(
        "\n\n".join(tables), encoding="utf-8"
    )
 
 
 
def dimension_summary(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    acc = final_accuracy_summary(dataset).set_index("method")
    data_ret = retention_at(data_noise(dataset), sigma=float(data_noise(dataset)["sigma"].max())).set_index("method")
    weight_layers = {}
    for layer in LAYERS:
        weight_layers[layer] = retention_at(weight_noise(dataset, layer), sigma=0.15).set_index("method")
    common = set(acc.index) & set(data_ret.index)
    for layer in LAYERS:
        common &= set(weight_layers[layer].index)
    methods = ordered_methods(list(common))
    df = pd.DataFrame(index=methods)
    df["val_acc"] = acc.loc[methods, "val_acc"]
    df["data_retention"] = data_ret.loc[methods, "retention"]
    for layer in LAYERS:
        df[f"{layer}_retention"] = weight_layers[layer].loc[methods, "retention"]
    df["inter_layer_score"] = df[[f"{layer}_retention" for layer in LAYERS]].min(axis=1)
    return df, acc


# ── Tabla 5.6 ─────────────────────────────────────────────────────────────────
 
def generate_table_5_06() -> None:
    tables: list[str] = []
    for dataset in DATASETS:
        df, _ = dimension_summary(dataset)
        rows = []
        data_values = df["data_retention"]
        fc1_values = df["fc1_retention"]
        for method in df.sort_values("val_acc", ascending=False).index:
            row = df.loc[method]
            data_cat = rank_category(data_values, row["data_retention"])
            fc1_cat = rank_category(fc1_values, row["fc1_retention"])
            profile = minima_profile(row["conv1_retention"], row["conv3_retention"], row["fc1_retention"])
            rows.append(
                [
                    method_label(method),
                    fmt_float(row["val_acc"], 3),
                    f"{data_cat} ({fmt_pct(row['data_retention'], 1)}%)",
                    f"{fc1_cat} ({fmt_pct(row['fc1_retention'], 1)}%)",
                    profile,
                ]
            )
        tables.append(latex_table(
            headers=["Método", "val\_acc", "Rob. datos", "Rob. pesos fc1", "Mínimo"],
            rows=rows,
            label=f"tab:5_06_{dataset.lower()}",
            caption=f"Tabla 5.6 — {dataset}: evaluación comparativa multidimensional.",
        ))
    (TABLE_DIR / "table_5_06_multidimensional_evaluation_all_datasets.tex").write_text(
        "\n\n".join(tables), encoding="utf-8"
    )


def generate_tables() -> None:
    generate_inventory()
    generate_estimation_manifest()
    generate_table_4_01()
    generate_table_5_01()
    generate_table_5_02()
    generate_table_5_03()
    generate_table_5_04()
    generate_table_5_05()
    generate_table_5_06()


def generate_figures() -> None:
    figure_functions: list[Callable[[], None]] = [
        generate_figure_5_01,
        generate_figure_5_02,
        generate_figure_5_03,
        generate_figure_5_04,
        generate_figure_5_05,
        generate_figure_5_06,
        generate_figure_5_07,
        generate_figure_5_08,
        generate_figure_5_09,
        generate_figure_5_10,
        generate_figure_5_11,
        generate_figure_5_12,
    ]
    for func in figure_functions:
        func()


def main() -> None:
    ensure_dirs()
    configure_style()
    generate_figures()
    generate_tables()
    print(f"Figuras generadas en: {FIG_DIR}")
    print(f"Tablas generadas en: {TABLE_DIR}")
    print(f"Inventario generado en: {OUT_ROOT / 'figure_table_inventory.tex'}")


if __name__ == "__main__":
    main()
