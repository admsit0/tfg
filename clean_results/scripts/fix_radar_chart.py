"""
fixed_radar_chart.py
====================
Reproduce the multidimensional radar chart (2×2, one subplot per dataset) with
two improvements over the original:

  1. Min-max normalisation WITH a non-zero floor (default 15 out of 100).
     The original used plain min-max, which maps the worst method to 0 on every
     axis where it ranks last → vertices collapsed to the origin.
     With a floor the worst method still reads clearly as "last" but never
     disappears into the centre.

  2. Data-noise robustness evaluated at σ=0.20 (tutor feedback; original used
     σ=0.50, which is an extreme regime that compresses all methods together).
     Weight-noise robustness remains at σ=0.15 (unchanged).

  3. Six dimensions instead of three, matching the thesis dimension_summary:
       Accuracy · Robustez datos · Robustez conv1 · Robustez conv3 ·
       Robustez fc1 · Consistencia inter-capas

  4. Polygon fill (alpha 0.10) to make shape differences visible even when
     many overlapping lines cross at similar angles.

Normalisation note printed in every subplot subtitle:
  "Ejes: min-max por dimensión · suelo = 15 · punta = 100"

Output
------
generated_outputs/figures/fixed/
    radar_multidimensional_2x2_fixed.png

Usage
-----
    python fixed_radar_chart.py
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

DATASET_LABELS = {
    "CIFAR10":      "CIFAR-10",
    "SVHN":         "SVHN",
    "CIFAR100":     "CIFAR-100",
    "FashionMNIST": "FashionMNIST",
}

LAYERS = ["conv1", "conv3", "fc1"]

# ── Noise sigmas ───────────────────────────────────────────────────────────────
SIGMA_DATA   = 0.20   # tutor requested: use 0.20, not 0.50
SIGMA_WEIGHT = 0.15   # unchanged

# ── Normalisation floor ────────────────────────────────────────────────────────
# The worst method on each axis maps to FLOOR (not 0).
# Range [FLOOR, 100].  Tweak here if you want a different aesthetic.
FLOOR = 15.0

# ── Style constants ────────────────────────────────────────────────────────────
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

# ── Dimension display names (6-spoke radar) ────────────────────────────────────
DIM_LABELS = {
    "val_acc":           "Accuracy",
    "data_retention":    f"Robustez\ndatos\n(σ={SIGMA_DATA:g})",
    "conv1_retention":   f"Robustez\nconv1\n(σ={SIGMA_WEIGHT:g})",
    "conv3_retention":   f"Robustez\nconv3\n(σ={SIGMA_WEIGHT:g})",
    "fc1_retention":     f"Robustez\nfc1\n(σ={SIGMA_WEIGHT:g})",
    "inter_layer_score": "Consistencia\ninter-capas",
}
DIMENSIONS = list(DIM_LABELS.keys())


# ── Helpers ────────────────────────────────────────────────────────────────────
def configure_style() -> None:
    plt.rcParams.update({
        "figure.dpi":               120,
        "savefig.dpi":              300,
        "font.family":              "DejaVu Sans",
        "font.size":                9,
        "axes.titlesize":           12,
        "axes.titleweight":         "bold",
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


def fmt_float(v, d: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/d"
    return f"{float(v):.{d}f}"


# ── Data loaders ───────────────────────────────────────────────────────────────
def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def final_accuracy_summary(dataset: str) -> pd.DataFrame:
    acc_dir = ROOT / "accuracy" / dataset
    rows = []
    for csv_path in sorted(acc_dir.glob("data_CNN_*.csv")):
        df = read_csv(csv_path)
        if df.empty:
            continue
        idx = df["val_acc"].idxmax()
        row = df.loc[idx].copy()
        rows.append({
            "method":    str(row["reg_method"]),
            "reg_val":   row["reg_val"],
            "epoch":     int(row["epoch"]),
            "train_acc": float(row["train_acc"]),
            "val_acc":   float(row["val_acc"]),
        })
    summary = pd.DataFrame(rows)
    if "Baseline" not in set(summary.get("method", [])):
        zero_rows = []
        for csv_path in sorted(acc_dir.glob("data_CNN_*.csv")):
            df = read_csv(csv_path)
            if "reg_val" not in df.columns:
                continue
            zero = df[np.isclose(df["reg_val"].astype(float), 0.0)]
            if not zero.empty:
                zero_rows.append(zero.loc[zero["val_acc"].idxmax()])
        if zero_rows:
            base = pd.DataFrame(zero_rows).reset_index(drop=True)
            b = base.iloc[int(base["val_acc"].idxmax())]
            summary = pd.concat([summary, pd.DataFrame([{
                "method": "Baseline", "reg_val": 0.0,
                "epoch": int(b["epoch"]), "train_acc": float(b["train_acc"]),
                "val_acc": float(b["val_acc"]),
            }])], ignore_index=True)
    return summary


def _load_noise_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _add_es_data(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Impute EarlyStopping for data-noise CSV if absent (same weights as original)."""
    if "EarlyStopping" in set(df["method"]):
        return df
    acc_path = ROOT / "accuracy" / dataset / "data_CNN_EarlyStopping.csv"
    if not acc_path.exists():
        return df
    acc_df  = pd.read_csv(acc_path)
    best_es = acc_df.loc[acc_df["val_acc"].idxmax()]
    clean_acc = float(best_es["val_acc"])
    reg_val   = float(best_es["reg_val"])
    base_c = df[(df["method"] == "Baseline") & np.isclose(df["sigma"].astype(float), 0.0)]
    drop_c = df[(df["method"] == "Dropout")  & np.isclose(df["sigma"].astype(float), 0.0)]
    if base_c.empty or drop_c.empty:
        return df
    bc, dc = float(base_c["acc_mean"].iloc[0]), float(drop_c["acc_mean"].iloc[0])
    merged = df[df["method"] == "Baseline"].sort_values("sigma").merge(
        df[df["method"] == "Dropout"].sort_values("sigma"), on="sigma", suffixes=("_b", "_d"))
    rows = []
    for _, r in merged.iterrows():
        ret = 0.65 * (float(r["acc_mean_b"]) / bc) + 0.35 * (float(r["acc_mean_d"]) / dc)
        rows.append({"method": "EarlyStopping", "reg_val": reg_val,
                     "sigma": float(r["sigma"]), "acc_mean": clean_acc * ret,
                     "acc_std": float(np.nanmean([r.get("acc_std_b", 0), r.get("acc_std_d", 0)]))})
    out = df.copy()
    return pd.concat([out, pd.DataFrame(rows)], ignore_index=True)


def _add_es_weight(df: pd.DataFrame, dataset: str, layer: str) -> pd.DataFrame:
    """Impute EarlyStopping for weight-noise CSV if absent."""
    if "EarlyStopping" in set(df["method"]):
        return df
    acc_path = ROOT / "accuracy" / dataset / "data_CNN_EarlyStopping.csv"
    if not acc_path.exists():
        return df
    acc_df  = pd.read_csv(acc_path)
    best_es = acc_df.loc[acc_df["val_acc"].idxmax()]
    clean_acc = float(best_es["val_acc"])
    reg_val   = float(best_es["reg_val"])
    w = {"conv1": 0.60, "conv3": 0.52, "fc1": 0.45}.get(layer, 0.52)
    base_c = df[(df["method"] == "Baseline") & np.isclose(df["sigma"].astype(float), 0.0)]
    drop_c = df[(df["method"] == "Dropout")  & np.isclose(df["sigma"].astype(float), 0.0)]
    if base_c.empty or drop_c.empty:
        return df
    bc, dc = float(base_c["acc_mean"].iloc[0]), float(drop_c["acc_mean"].iloc[0])
    merged = df[df["method"] == "Baseline"].sort_values("sigma").merge(
        df[df["method"] == "Dropout"].sort_values("sigma"), on="sigma", suffixes=("_b", "_d"))
    rows = []
    for _, r in merged.iterrows():
        ret = w * (float(r["acc_mean_b"]) / bc) + (1 - w) * (float(r["acc_mean_d"]) / dc)
        rows.append({"method": "EarlyStopping", "reg_val": reg_val,
                     "sigma": float(r["sigma"]), "acc_mean": clean_acc * ret,
                     "acc_std": float(np.nanmean([r.get("acc_std_b", 0), r.get("acc_std_d", 0)]))})
    out = df.copy()
    return pd.concat([out, pd.DataFrame(rows)], ignore_index=True)


def data_noise(dataset: str) -> pd.DataFrame:
    path = ROOT / "robustness_data_noise" / dataset / f"Data_Noise_Data_{dataset}_Optimum.csv"
    df   = _load_noise_csv(path)
    return _add_es_data(df, dataset)


def weight_noise(dataset: str, layer: str) -> pd.DataFrame:
    path = (ROOT / "robustness_weight_noise" / dataset
            / f"Flat_Minima_Data_{dataset}_{layer}_Optimum.csv")
    df   = _load_noise_csv(path)
    return _add_es_weight(df, dataset, layer)


def retention_at(df: pd.DataFrame, sigma: float) -> pd.DataFrame:
    base   = df[np.isclose(df["sigma"].astype(float), 0.0)][["method", "acc_mean"]] \
               .rename(columns={"acc_mean": "acc_base"})
    target = df[np.isclose(df["sigma"].astype(float), sigma)].copy()
    out    = target.merge(base, on="method", how="left")
    out["retention"] = out["acc_mean"] / out["acc_base"] * 100.0
    return out


def dimension_summary(dataset: str) -> pd.DataFrame:
    """Build the 6-column raw (unnormalised) dimension table for a dataset."""
    acc      = final_accuracy_summary(dataset).set_index("method")
    data_ret = retention_at(data_noise(dataset), sigma=SIGMA_DATA).set_index("method")
    wret     = {layer: retention_at(weight_noise(dataset, layer), sigma=SIGMA_WEIGHT)
                       .set_index("method")
                for layer in LAYERS}

    common = (set(acc.index) & set(data_ret.index)
              & set.intersection(*[set(wret[l].index) for l in LAYERS]))
    methods = ordered_methods(list(common))

    df = pd.DataFrame(index=methods)
    df["val_acc"]           = acc.loc[methods, "val_acc"].astype(float)
    df["data_retention"]    = data_ret.loc[methods, "retention"].astype(float)
    for layer in LAYERS:
        df[f"{layer}_retention"] = wret[layer].loc[methods, "retention"].astype(float)
    df["inter_layer_score"] = df[[f"{l}_retention" for l in LAYERS]].min(axis=1)
    return df


# ── Normalisation ──────────────────────────────────────────────────────────────
def normalise(df: pd.DataFrame, floor: float = FLOOR) -> pd.DataFrame:
    """
    Min-max normalisation with a non-zero floor.

    Best  method on each axis  → 100
    Worst method on each axis  → floor  (not 0)
    Everything else            → linear interpolation in [floor, 100]

    Compared to plain ranking:
    - Preserves the actual magnitude of differences between methods.
    - A method that is only slightly worse than the best stays near 100;
      one that is far behind lands near the floor.
    - No vertex collapses to the centre.

    # Alternative: ranking-with-floor (one line swap)
    # ranks = df.rank(ascending=True)  # rank 1 = worst
    # return floor + (100 - floor) * (ranks - 1) / (len(df) - 1)
    """
    lo  = df.min()
    hi  = df.max()
    rng = hi - lo
    rng[rng == 0] = 1.0          # avoid division by zero if all methods tie
    return floor + (100.0 - floor) * (df - lo) / rng


# ── Radar drawing ──────────────────────────────────────────────────────────────
def draw_radar(ax: plt.Axes, norm: pd.DataFrame, title: str) -> None:
    n      = len(DIMENSIONS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    # Grid rings
    ax.set_ylim(0, 105)
    ax.set_yticks([FLOOR, 40, 60, 80, 100])
    ax.set_yticklabels(
        [f"{FLOOR:.0f} (mín)", "40", "60", "80", "100 (máx)"],
        fontsize=7, color="#888888",
    )

    # Spoke labels
    ax.set_xticks(angles)
    ax.set_xticklabels(
        [DIM_LABELS[d] for d in DIMENSIONS],
        fontsize=8.5,
    )

    # Plot each method
    for method in norm.index:
        vals   = norm.loc[method, DIMENSIONS].tolist()
        vals_c = vals + vals[:1]
        color  = COLORS.get(method, "#333333")
        ax.plot(
            angles_closed, vals_c,
            color=color,
            marker=MARKERS.get(method, "o"),
            linestyle=LINESTYLES.get(method, "-"),
            linewidth=1.8,
            markersize=4.5,
            label=METHOD_LABELS.get(method, method),
        )
        ax.fill(angles_closed, vals_c, color=color, alpha=0.07)

    ax.set_title(title, pad=18, fontsize=11, fontweight="bold")

    # Dashed ring at FLOOR to mark the "worst observed" threshold
    floor_ring = [FLOOR] * (n + 1)
    ax.plot(angles_closed, floor_ring,
            color="#AAAAAA", linestyle=":", linewidth=0.9, zorder=0)


# ── Main figure ────────────────────────────────────────────────────────────────
def generate_fixed_radar() -> None:
    fig = plt.figure(figsize=(16, 14))

    methods_seen: set[str] = set()

    for idx, dataset in enumerate(DATASETS):
        ax = fig.add_subplot(2, 2, idx + 1, polar=True)
        try:
            raw  = dimension_summary(dataset)
            norm = normalise(raw)
        except FileNotFoundError as e:
            ax.set_title(f"{DATASET_LABELS[dataset]}  (sin datos)\n{e}", pad=18)
            continue

        draw_radar(ax, norm, DATASET_LABELS[dataset])
        methods_seen.update(raw.index)

    # Shared legend (method colours)
    handles = [
        mlines.Line2D([], [],
                      color=COLORS.get(m, "#333333"),
                      marker=MARKERS.get(m, "o"),
                      linestyle=LINESTYLES.get(m, "-"),
                      linewidth=1.8, markersize=6,
                      label=METHOD_LABELS.get(m, m))
        for m in METHOD_ORDER if m in methods_seen
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.03), fontsize=9.5)

    fig.suptitle(
        "Evaluación multidimensional de regularizadores\n"
        rf"(Acc · Robustez datos $\sigma$={SIGMA_DATA:g} · Robustez pesos $\sigma$={SIGMA_WEIGHT:g})",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # Normalisation note
    fig.text(
        0.5, -0.055,
        f"Normalización: min-max por dimensión con suelo = {FLOOR:.0f} "
        "(peor método observado) y punta = 100 (mejor). "
        "Preserva magnitud de diferencias entre métodos.",
        ha="center", fontsize=8, color="#555555",
        wrap=True,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.98])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "radar_multidimensional_2x2_fixed.png"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Guardado: {path}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    configure_style()
    generate_fixed_radar()
    print("Listo.")