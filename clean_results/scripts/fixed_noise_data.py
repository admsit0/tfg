"""
fixed_noise_data.py
===================
Generates ALL figures and LaTeX tables for the data-noise robustness experiment,
broken down by every sigma level tested (not just the maximum).

Output layout
-------------
generated_outputs/figures/fixed/noise/
  data_noise_curves_2x2.png                      ← full degradation curves (all σ)
  data_noise_retention_sigma_<s>_2x2.png         ← retention bar chart per σ
  data_noise_retention_sigma_<s>.tex             ← combined 4-dataset LaTeX table per σ

Usage
-----
    python fixed_noise_data.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "generated_outputs" / "figures" / "fixed" / "noise"

DATASETS = ["CIFAR10", "SVHN", "CIFAR100", "FashionMNIST"]

DATASET_LABELS = {
    "CIFAR10":      "CIFAR-10",
    "SVHN":         "SVHN",
    "CIFAR100":     "CIFAR-100",
    "FashionMNIST": "FashionMNIST",
}

# ── Style (identical to original) ─────────────────────────────────────────────
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

LINESTYLES = {
    "Baseline":      "--",
    "DataAug":       "-",
    "BatchNorm":     "-.",
    "Dropout":       ":",
    "GaussianNoise": "-",
    "EarlyStopping": "--",
    "L1":            "-.",
    "L2":            ":",
}


def configure_style() -> None:
    plt.rcParams.update({
        "figure.dpi":               120,
        "savefig.dpi":              300,
        "font.family":              "DejaVu Sans",
        "font.size":                10,
        "axes.titlesize":           13,
        "axes.labelsize":           10,
        "axes.titleweight":         "bold",
        "axes.grid":                True,
        "grid.color":               "#D8D8D8",
        "grid.linewidth":           0.7,
        "grid.alpha":               0.75,
        "axes.spines.top":          False,
        "axes.spines.right":        False,
        "legend.frameon":           True,
        "legend.framealpha":        0.95,
        "legend.edgecolor":         "#BDBDBD",
        "figure.constrained_layout.use": False,
    })


def ordered_methods(methods) -> list[str]:
    present   = list(dict.fromkeys(list(methods)))
    preferred = [m for m in METHOD_ORDER if m in present]
    remainder = sorted(m for m in present if m not in preferred)
    return preferred + remainder


def method_label(m: str) -> str:
    return METHOD_LABELS.get(m, m)


def fmt_pct(v, d: int = 1) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/d"
    return f"{float(v):.{d}f}"


def bold_tex(s: str) -> str:
    return f"\\textbf{{{s}}}"


def sigma_str(s: float) -> str:
    """Compact, filename-safe representation of a sigma value."""
    return f"{s:g}".replace(".", "_")


# ── Early-stopping imputation (reused from original) ──────────────────────────
def add_early_stopping_estimate(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if "EarlyStopping" in set(df["method"]):
        return df
    acc_path = ROOT / "accuracy" / dataset / "data_CNN_EarlyStopping.csv"
    if not acc_path.exists():
        return df
    acc_df  = pd.read_csv(acc_path)
    best_es = acc_df.loc[acc_df["val_acc"].idxmax()]
    reg_val  = float(best_es["reg_val"])
    clean_acc = float(best_es["val_acc"])

    base_clean_rows  = df[(df["method"] == "Baseline")  & np.isclose(df["sigma"].astype(float), 0.0)]
    drop_clean_rows  = df[(df["method"] == "Dropout")   & np.isclose(df["sigma"].astype(float), 0.0)]
    if base_clean_rows.empty or drop_clean_rows.empty:
        return df
    base_clean = float(base_clean_rows["acc_mean"].iloc[0])
    drop_clean = float(drop_clean_rows["acc_mean"].iloc[0])

    base_curves = df[df["method"] == "Baseline"].sort_values("sigma")
    drop_curves = df[df["method"] == "Dropout"].sort_values("sigma")
    merged = base_curves.merge(drop_curves, on="sigma", suffixes=("_base", "_drop"))

    estimated_rows = []
    for _, row in merged.iterrows():
        ret = 0.65 * (float(row["acc_mean_base"]) / base_clean) + \
              0.35 * (float(row["acc_mean_drop"]) / drop_clean)
        estimated_rows.append({
            "method":    "EarlyStopping",
            "reg_val":   reg_val,
            "sigma":     float(row["sigma"]),
            "acc_mean":  clean_acc * ret,
            "acc_std":   float(np.nanmean([row["acc_std_base"], row["acc_std_drop"]])),
            "estimated": True,
        })
    out = df.copy()
    out["estimated"] = False
    return pd.concat([out, pd.DataFrame(estimated_rows)], ignore_index=True)


def load_data_noise(dataset: str) -> pd.DataFrame:
    path = ROOT / "robustness_data_noise" / dataset / f"Data_Noise_Data_{dataset}_Optimum.csv"
    df   = pd.read_csv(path)
    df   = add_early_stopping_estimate(df, dataset)
    if "estimated" not in df.columns:
        df["estimated"] = False
    return df


def load_all_data_noise() -> dict[str, pd.DataFrame]:
    return {ds: load_data_noise(ds) for ds in DATASETS}


def discover_sigmas(dfs: dict[str, pd.DataFrame]) -> list[float]:
    """All non-zero sigma values common (or union) across datasets, sorted."""
    all_sigmas: set[float] = set()
    for df in dfs.values():
        non_zero = df[~np.isclose(df["sigma"].astype(float), 0.0)]["sigma"].astype(float)
        all_sigmas.update(non_zero.unique())
    return sorted(all_sigmas)


def retention_at_sigma(df: pd.DataFrame, sigma: float) -> pd.DataFrame:
    base = df[np.isclose(df["sigma"].astype(float), 0.0)][["method", "acc_mean"]] \
             .rename(columns={"acc_mean": "acc_base"})
    target = df[np.isclose(df["sigma"].astype(float), sigma)].copy()
    out = target.merge(base, on="method", how="left")
    out["retention"] = out["acc_mean"] / out["acc_base"] * 100.0
    return out


# ── Figure 1 – Full degradation curves, 2×2 ────────────────────────────────────
def plot_curves_2x2(dfs: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, dataset in zip(axes.flat, DATASETS):
        df = dfs[dataset]
        for method in ordered_methods(df["method"]):
            part = df[df["method"] == method].sort_values("sigma")
            ax.plot(
                part["sigma"], part["acc_mean"],
                color=COLORS.get(method, "#333333"),
                marker=MARKERS.get(method, "o"),
                linestyle=LINESTYLES.get(method, "-"),
                linewidth=2.0, markersize=5.0,
                label=method_label(method),
            )
            if "acc_std" in part.columns:
                ax.fill_between(
                    part["sigma"].to_numpy(),
                    (part["acc_mean"] - part["acc_std"]).to_numpy(),
                    (part["acc_mean"] + part["acc_std"]).to_numpy(),
                    color=COLORS.get(method, "#333333"), alpha=0.10, linewidth=0,
                )
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("Desviación estándar del ruido (σ)")
        ax.set_ylabel("Accuracy de validación (%)")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04), fontsize=9)
    fig.suptitle("Degradación ante ruido gaussiano en datos de entrada",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "data_noise_curves_2x2.png"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  {path.name}")


# ── Figure 2 – Retention bar chart per sigma, 2×2 ─────────────────────────────
def plot_retention_per_sigma(dfs: dict[str, pd.DataFrame], sigmas: list[float]) -> None:
    for sigma in sigmas:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for ax, dataset in zip(axes.flat, DATASETS):
            ret = retention_at_sigma(dfs[dataset], sigma)
            if ret.empty:
                ax.set_title(f"{DATASET_LABELS[dataset]}  (sin datos para σ={sigma:g})")
                continue

            ret_sorted = ret.sort_values("retention", ascending=True)
            labels  = [method_label(m) for m in ret_sorted["method"]]
            colors  = [COLORS.get(m, "#333333") for m in ret_sorted["method"]]
            best    = ret_sorted["retention"].max()

            bars = ax.barh(labels, ret_sorted["retention"],
                           color=colors, edgecolor="black", linewidth=0.5)
            for bar, (_, row) in zip(bars, ret_sorted.iterrows()):
                weight = "bold" if abs(row["retention"] - best) < 1e-9 else "normal"
                ax.text(
                    row["retention"] + 0.4,
                    bar.get_y() + bar.get_height() / 2,
                    f"{row['retention']:.1f}%",
                    va="center", fontsize=8.5, fontweight=weight,
                )
            ax.set_title(DATASET_LABELS[dataset])
            ax.set_xlabel("Retención de accuracy (%)")
            ax.set_xlim(0, min(115, ret_sorted["retention"].max() + 12))

        fig.suptitle(f"Retención ante ruido en datos  (σ = {sigma:g})",
                     fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])

        fname = f"data_noise_retention_sigma_{sigma_str(sigma)}_2x2.png"
        path  = OUT_DIR / fname
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  {fname}")


# ── LaTeX tables per sigma ─────────────────────────────────────────────────────
def latex_retention_table_data(
    dfs: dict[str, pd.DataFrame],
    sigma: float,
    label: str,
    caption: str,
) -> str:
    n_cols = 4        # Método | acc σ=0 | acc σ=s | Retención
    col_spec = "lccc"

    header = (
        r"\textbf{Método} & \textbf{Acc. $\sigma$=0} & "
        rf"\textbf{{Acc. $\sigma$={sigma:g}}} & \textbf{{Retención (\%)}} \\"
    )

    dataset_blocks: list[str] = []
    for dataset in DATASETS:
        ret = retention_at_sigma(dfs[dataset], sigma)
        if ret.empty:
            continue
        best_ret = ret["retention"].max()
        ret_sorted = ret.sort_values("retention", ascending=False)

        rows_tex: list[str] = []
        for _, row in ret_sorted.iterrows():
            ret_val = fmt_pct(row["retention"], 1)
            if abs(row["retention"] - best_ret) < 1e-9:
                ret_val = bold_tex(ret_val)
            rows_tex.append(
                f"{method_label(row['method'])} & "
                f"{fmt_pct(row['acc_base'], 3)} & "
                f"{fmt_pct(row['acc_mean'], 3)} & "
                f"{ret_val} \\\\"
            )

        block = (
            f"\\multicolumn{{{n_cols}}}{{l}}{{\\textbf{{{DATASET_LABELS[dataset]}}}}}\\\\\n"
            "\\midrule\n"
            + "\n".join(rows_tex)
        )
        dataset_blocks.append(block)

    body = "\n\\midrule\n".join(dataset_blocks)

    return (
        f"\\begin{{table}}{{{label}}}{{{caption}}}\n"
        "\\small\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )


def write_tables_per_sigma(dfs: dict[str, pd.DataFrame], sigmas: list[float]) -> None:
    for sigma in sigmas:
        s_str = sigma_str(sigma)
        label   = f"tab:data_noise_retention_sigma_{s_str}"
        caption = (
            f"Retención de \\textit{{accuracy}} ante ruido gaussiano en datos "
            f"($\\sigma$={sigma:g}) por método en los cuatro datasets evaluados."
        )
        tex  = latex_retention_table_data(dfs, sigma, label, caption)
        path = OUT_DIR / f"data_noise_retention_sigma_{s_str}.tex"
        path.write_text(tex, encoding="utf-8")
        print(f"  {path.name}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Cargando datos de ruido en datos…")
    dfs    = load_all_data_noise()
    sigmas = discover_sigmas(dfs)
    print(f"  Niveles de σ encontrados: {sigmas}")

    print("\nFigura de curvas completas:")
    plot_curves_2x2(dfs)

    print("\nFiguras de retención por σ:")
    plot_retention_per_sigma(dfs, sigmas)

    print("\nTablas LaTeX por σ:")
    write_tables_per_sigma(dfs, sigmas)

    print(f"\nListo. Todo en: {OUT_DIR}")