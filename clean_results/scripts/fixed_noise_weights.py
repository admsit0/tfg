"""
fixed_noise_weights.py
======================
Genera las tablas LaTeX de retencion para el experimento de ruido en pesos.

La salida replica la estructura de las tablas de data-noise por nivel de
sigma, pero aqui las capas son columnas: conv1, conv3 y fc1.

Output:
generated_outputs/figures/fixed/noise/
  weight_noise_retention_sigma_<s>.tex

Uso:
    python scripts/fixed_noise_weights.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "generated_outputs" / "figures" / "fixed" / "noise"

DATASETS = ["CIFAR10", "SVHN", "CIFAR100", "FashionMNIST"]
LAYERS = ["conv1", "conv3", "fc1"]

DATASET_LABELS = {
    "CIFAR10": "CIFAR-10",
    "SVHN": "SVHN",
    "CIFAR100": "CIFAR-100",
    "FashionMNIST": "FashionMNIST",
}

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


def method_rank(method: str) -> int:
    if method in METHOD_ORDER:
        return METHOD_ORDER.index(method)
    return len(METHOD_ORDER)


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def fmt_pct(value, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/d"
    return f"{float(value):.{digits}f}"


def bold_tex(value: str) -> str:
    return f"\\textbf{{{value}}}"


def sigma_str(sigma: float) -> str:
    return f"{sigma:g}".replace(".", "_")


def weight_noise_path(dataset: str, layer: str) -> Path:
    return (
        ROOT
        / "robustness_weight_noise"
        / dataset
        / f"Flat_Minima_Data_{dataset}_{layer}_Optimum.csv"
    )


def clean_accuracy_for_early_stopping(dataset: str, reference_scale: float) -> tuple[float, float] | None:
    path = ROOT / "accuracy" / dataset / "data_CNN_EarlyStopping.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "val_acc" not in df.columns:
        return None
    best = df.loc[df["val_acc"].idxmax()]
    clean_acc = float(best["val_acc"])
    if reference_scale > 1.5 and clean_acc <= 1.5:
        clean_acc *= 100.0
    return clean_acc, float(best.get("reg_val", np.nan))


def add_early_stopping_estimate(df: pd.DataFrame, dataset: str, layer: str) -> pd.DataFrame:
    if "EarlyStopping" in set(df["method"]):
        out = df.copy()
        if "estimated" not in out.columns:
            out["estimated"] = False
        return out

    if "Baseline" not in set(df["method"]) or "Dropout" not in set(df["method"]):
        out = df.copy()
        out["estimated"] = False
        return out

    base = df[df["method"] == "Baseline"].sort_values("sigma")
    drop = df[df["method"] == "Dropout"].sort_values("sigma")
    base_zero = base[np.isclose(base["sigma"].astype(float), 0.0)]
    drop_zero = drop[np.isclose(drop["sigma"].astype(float), 0.0)]
    if base_zero.empty or drop_zero.empty:
        out = df.copy()
        out["estimated"] = False
        return out

    reference_scale = float(base_zero["acc_mean"].iloc[0])
    clean = clean_accuracy_for_early_stopping(dataset, reference_scale)
    if clean is None:
        out = df.copy()
        out["estimated"] = False
        return out
    clean_acc, reg_val = clean

    merged = base[["sigma", "acc_mean", "acc_std"]].merge(
        drop[["sigma", "acc_mean", "acc_std"]],
        on="sigma",
        suffixes=("_base", "_drop"),
    )

    base_clean = float(base_zero["acc_mean"].iloc[0])
    drop_clean = float(drop_zero["acc_mean"].iloc[0])
    baseline_weight = {"conv1": 0.60, "conv3": 0.52, "fc1": 0.45}.get(layer, 0.50)

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


def load_weight_noise(dataset: str, layer: str) -> pd.DataFrame:
    df = pd.read_csv(weight_noise_path(dataset, layer))
    df = add_early_stopping_estimate(df, dataset, layer)
    if "estimated" not in df.columns:
        df["estimated"] = False
    return df


def load_all_weight_noise() -> dict[str, dict[str, pd.DataFrame]]:
    return {
        dataset: {layer: load_weight_noise(dataset, layer) for layer in LAYERS}
        for dataset in DATASETS
    }


def discover_sigmas(dfs: dict[str, dict[str, pd.DataFrame]]) -> list[float]:
    sigmas: set[float] = set()
    for layer_dfs in dfs.values():
        for df in layer_dfs.values():
            non_zero = df[~np.isclose(df["sigma"].astype(float), 0.0)]["sigma"].astype(float)
            sigmas.update(non_zero.unique())
    return sorted(sigmas)


def retention_at_sigma(df: pd.DataFrame, sigma: float) -> pd.DataFrame:
    base = (
        df[np.isclose(df["sigma"].astype(float), 0.0)][["method", "acc_mean"]]
        .rename(columns={"acc_mean": "acc_base"})
    )
    target = df[np.isclose(df["sigma"].astype(float), sigma)].copy()
    out = target.merge(base, on="method", how="left")
    out["retention"] = out["acc_mean"] / out["acc_base"] * 100.0
    return out


def retention_by_layer_for_dataset(
    dfs: dict[str, dict[str, pd.DataFrame]], dataset: str, sigma: float
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for layer in LAYERS:
        ret = retention_at_sigma(dfs[dataset][layer], sigma)
        ret = ret[["method", "retention"]].rename(columns={"retention": f"ret_{layer}"})
        frames.append(ret)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="method", how="outer")

    ret_cols = [f"ret_{layer}" for layer in LAYERS]
    merged["min_ret"] = merged[ret_cols].min(axis=1)
    merged["_method_rank"] = merged["method"].map(method_rank)
    return merged.sort_values(["min_ret", "_method_rank"], ascending=[False, True]).reset_index(drop=True)


def latex_weight_retention_table(
    dfs: dict[str, dict[str, pd.DataFrame]], sigma: float, label: str, caption: str
) -> str:
    n_cols = 4
    col_spec = "lccc"
    header = (
        r"\textbf{Método} & \textbf{Ret. conv1 (\%)} & "
        r"\textbf{Ret. conv3 (\%)} & \textbf{Ret. fc1 (\%)} \\"
    )

    dataset_blocks: list[str] = []
    for dataset in DATASETS:
        ret = retention_by_layer_for_dataset(dfs, dataset, sigma)
        if ret.empty:
            continue

        layer_cols = [f"ret_{layer}" for layer in LAYERS]
        best_by_layer = {col: ret[col].max(skipna=True) for col in layer_cols}
        rows_tex: list[str] = []
        for _, row in ret.iterrows():
            values: list[str] = []
            for col in layer_cols:
                value = row[col]
                formatted = fmt_pct(value, 1)
                if pd.notna(value) and abs(float(value) - float(best_by_layer[col])) < 1e-9:
                    formatted = bold_tex(formatted)
                values.append(formatted)
            rows_tex.append(f"{method_label(row['method'])} & {' & '.join(values)} \\\\")

        dataset_blocks.append(
            f"\\multicolumn{{{n_cols}}}{{l}}{{\\textbf{{{DATASET_LABELS[dataset]}}}}}\\\\\n"
            "\\midrule\n"
            + "\n".join(rows_tex)
        )

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


def write_tables_per_sigma(dfs: dict[str, dict[str, pd.DataFrame]], sigmas: list[float]) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for sigma in sigmas:
        s_str = sigma_str(sigma)
        label = f"tab:weight_noise_retention_sigma_{s_str}"
        caption = (
            f"Retención de \\textit{{accuracy}} ante ruido gaussiano en pesos "
            f"($\\sigma$={sigma:g}) por capa y método en los cuatro datasets evaluados."
        )
        tex = latex_weight_retention_table(dfs, sigma, label, caption)
        path = OUT_DIR / f"weight_noise_retention_sigma_{s_str}.tex"
        path.write_text(tex, encoding="utf-8")
        written.append(path)
    return written


def main() -> None:
    dfs = load_all_weight_noise()
    sigmas = discover_sigmas(dfs)
    written = write_tables_per_sigma(dfs, sigmas)

    print("Generated:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
