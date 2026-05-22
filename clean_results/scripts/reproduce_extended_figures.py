from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import reproduce_thesis_outputs as base


ROOT = base.ROOT
FIG_DIR = base.FIG_DIR
COMBINED_DIR = FIG_DIR / "combined"
DATASETS = base.DATASETS
LAYERS = base.LAYERS


def dataset_dir(dataset: str) -> Path:
    return FIG_DIR / dataset


def ensure_extended_dirs() -> None:
    base.ensure_dirs()
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        dataset_dir(dataset).mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_name(dataset: str, slug: str) -> str:
    return f"{dataset}_{slug}.png"


def method_file(dataset: str, method: str) -> Path | None:
    if method == "Baseline":
        candidate = ROOT / "accuracy" / dataset / "data_CNN_Baseline.csv"
        return candidate if candidate.exists() else None
    return ROOT / "accuracy" / dataset / f"data_CNN_{method}.csv"


def best_curve(dataset: str, method: str) -> pd.DataFrame:
    summary = base.final_accuracy_summary(dataset)
    selected = summary[summary["method"] == method]
    if selected.empty:
        return pd.DataFrame()
    reg_val = float(selected.iloc[0]["reg_val"])

    path = method_file(dataset, method)
    if path is not None and path.exists():
        df = base.read_csv(path)
        return df[np.isclose(df["reg_val"].astype(float), reg_val)].sort_values("epoch")

    zero_runs = []
    for csv_path in sorted((ROOT / "accuracy" / dataset).glob("data_CNN_*.csv")):
        df = base.read_csv(csv_path)
        zero = df[np.isclose(df["reg_val"].astype(float), 0.0)].copy()
        if not zero.empty:
            zero_runs.append(zero)
    if not zero_runs:
        return pd.DataFrame()
    all_zero = pd.concat(zero_runs, ignore_index=True)
    # If there are several zero-regularization runs, keep the one whose peak
    # validation accuracy matches the selected baseline best.
    grouped = []
    for _, part in all_zero.groupby("reg_method"):
        grouped.append(part.loc[part["val_acc"].idxmax()])
    grouped_df = pd.DataFrame(grouped).reset_index(drop=True)
    best_source = grouped_df.iloc[int(grouped_df["val_acc"].idxmax())]
    source_method = best_source["reg_method"]
    return all_zero[all_zero["reg_method"] == source_method].sort_values("epoch")


def plot_ranking_ax(ax: plt.Axes, dataset: str, annotate: bool = True) -> None:
    df = base.final_accuracy_summary(dataset).sort_values("val_acc", ascending=True)
    labels = [base.method_label(m) for m in df["method"]]
    colors = [base.COLORS.get(m, "#333333") for m in df["method"]]
    bars = ax.barh(labels, df["val_acc"], color=colors, edgecolor="black", linewidth=0.45)
    if annotate:
        for bar, (_, row) in zip(bars, df.iterrows()):
            ax.text(
                row["val_acc"] + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{row['val_acc']:.3f}",
                va="center",
                fontsize=8,
            )
    ax.set_title(dataset)
    ax.set_xlabel("Mejor val_acc")
    ax.set_xlim(max(0, df["val_acc"].min() - 0.06), min(1.0, df["val_acc"].max() + 0.10))


def plot_accuracy_curves_ax(ax: plt.Axes, dataset: str) -> None:
    summary = base.final_accuracy_summary(dataset)
    for method in base.ordered_methods(summary["method"]):
        curve = best_curve(dataset, method)
        if curve.empty:
            continue
        ax.plot(
            curve["epoch"],
            curve["val_acc"],
            color=base.COLORS.get(method, "#333333"),
            marker=base.MARKERS.get(method, "o"),
            linestyle=base.LINESTYLES.get(method, "-"),
            linewidth=1.8,
            markersize=3.8,
            markevery=max(1, len(curve) // 6),
            label=base.method_label(method),
        )
    ax.set_title(dataset)
    ax.set_xlabel("Epoca")
    ax.set_ylabel("val_acc")


def plot_generalization_gap_ax(ax: plt.Axes, dataset: str) -> None:
    df = base.final_accuracy_summary(dataset).sort_values("gap", ascending=True)
    colors = [base.COLORS.get(m, "#333333") for m in df["method"]]
    ax.barh([base.method_label(m) for m in df["method"]], df["gap"], color=colors, edgecolor="black", linewidth=0.45)
    ax.set_title(dataset)
    ax.set_xlabel("train_acc - val_acc")


def plot_data_aug_summary_ax(ax: plt.Axes, dataset: str) -> None:
    path = ROOT / "accuracy" / dataset / "data_CNN_DataAug.csv"
    df = base.read_csv(path)
    strengths = sorted(df["reg_val"].unique())
    best = df.groupby("reg_val")["val_acc"].max().reindex(strengths)
    final = df.sort_values("epoch").groupby("reg_val").tail(1).set_index("reg_val").reindex(strengths)
    gap = final["train_acc"] - final["val_acc"]
    ax.plot(strengths, best.values, color=base.COLORS["DataAug"], marker="s", linewidth=2.0, label="Mejor val_acc")
    ax2 = ax.twinx()
    ax2.plot(strengths, gap.values, color="#4D4D4D", marker="o", linestyle="--", linewidth=1.8, label="Gap final")
    ax.set_title(dataset)
    ax.set_xlabel("Nivel de aumento")
    ax.set_ylabel("Mejor val_acc")
    ax2.set_ylabel("Gap final")
    ax.set_xticks(strengths)


def plot_data_aug_heatmap_ax(ax: plt.Axes, dataset: str, metric: str) -> None:
    df = base.read_csv(ROOT / "accuracy" / dataset / "data_CNN_DataAug.csv")
    pivot = df.pivot(index="epoch", columns="reg_val", values=metric).sort_index()
    strengths = sorted(df["reg_val"].unique())
    im = ax.imshow(
        pivot.to_numpy(),
        aspect="auto",
        origin="lower",
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
        extent=[min(strengths) - 0.5, max(strengths) + 0.5, pivot.index.min(), pivot.index.max()],
    )
    ax.set_title(f"{dataset} - {metric}")
    ax.set_xlabel("Nivel de aumento")
    ax.set_ylabel("Epoca")
    return im


def activation_df(dataset: str, bins: int | None = None) -> pd.DataFrame:
    df = base.fc1_val(base.load_states(dataset, bins=bins, prefer_pdf_variant=True))
    if df.empty and bins is not None:
        df = base.fc1_val(base.load_states(dataset, bins=None, prefer_pdf_variant=True))
    return base.add_baseline_from_zero(df)


def preferred_activation_df(dataset: str) -> pd.DataFrame:
    if dataset in {"CIFAR10", "SVHN"}:
        bins = 30 if dataset == "CIFAR10" else 10
        return activation_df(dataset, bins=bins)
    return activation_df(dataset, bins=10)


def plot_unique_scatter_ax(ax: plt.Axes, dataset: str) -> None:
    df = preferred_activation_df(dataset)
    base.plot_method_scatter(
        ax,
        df,
        "unique_pctg",
        x_label="Estados unicos en fc1 (%)",
        y_label="val_acc",
        title=dataset,
    )
    legend = ax.get_legend()
    if legend:
        legend.remove()


def plot_entropy_scatter_ax(ax: plt.Axes, dataset: str) -> None:
    df = preferred_activation_df(dataset)
    if "entropy" not in df.columns:
        ax.text(0.5, 0.5, "Sin entropia en CSV", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(dataset)
        return
    base.plot_method_scatter(
        ax,
        df,
        "entropy",
        x_label="Entropia H en fc1",
        y_label="val_acc",
        title=dataset,
    )
    legend = ax.get_legend()
    if legend:
        legend.remove()


def plot_dispersion_scatter_ax(ax: plt.Axes, dataset: str) -> None:
    df = base.fc1_val(base.load_dispersion(dataset))
    df = base.add_baseline_from_zero(df)
    base.plot_method_scatter(
        ax,
        df,
        "dispersion_ratio",
        x_label="Ratio de dispersion en fc1",
        y_label="val_acc",
        title=dataset,
        include_baseline_line=True,
    )
    legend = ax.get_legend()
    if legend:
        legend.remove()


def plot_unique_vs_dispersion_ax(ax: plt.Axes, dataset: str) -> None:
    states = preferred_activation_df(dataset)
    disp = base.fc1_val(base.load_dispersion(dataset))
    disp = base.add_baseline_from_zero(disp)
    merged = states.merge(
        disp[["reg_method", "reg_val", "dispersion_ratio"]],
        on=["reg_method", "reg_val"],
        how="inner",
    )
    for method in base.ordered_methods(merged["reg_method"]):
        part = merged[merged["reg_method"] == method]
        ax.scatter(
            part["unique_pctg"],
            part["dispersion_ratio"],
            s=52,
            marker=base.MARKERS.get(method, "o"),
            color=base.COLORS.get(method, "#333333"),
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
            label=base.method_label(method),
        )
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_title(dataset)
    ax.set_xlabel("Estados unicos en fc1 (%)")
    ax.set_ylabel("Ratio de dispersion")


def plot_temporal_layers(dataset: str) -> None:
    df = base.temporal(dataset)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, layer in zip(axes, LAYERS):
        part_layer = df[df["layer"] == layer]
        for method in base.ordered_methods(part_layer["method"]):
            part = part_layer[part_layer["method"] == method].sort_values("epoch")
            ax.plot(
                part["epoch"],
                part["unique_pctg"],
                color=base.COLORS.get(method, "#333333"),
                marker=base.MARKERS.get(method, "o"),
                linestyle=base.LINESTYLES.get(method, "-"),
                linewidth=1.8,
                markersize=4.0,
                label=base.method_label(method),
            )
        ax.set_title(layer)
        ax.set_xlabel("Epoca")
        ax.set_ylim(-2, 104)
    axes[0].set_ylabel("Estados unicos (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=8.5)
    fig.suptitle(f"Evolucion temporal por capa - {dataset}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    save(fig, dataset_dir(dataset) / figure_name(dataset, "temporal_unique_pctg_by_layer"))


def plot_temporal_fc1_ax(ax: plt.Axes, dataset: str) -> None:
    plot_temporal_layer_ax(ax, dataset, "fc1")


def plot_temporal_layer_ax(ax: plt.Axes, dataset: str, layer: str) -> None:
    df = base.temporal(dataset)
    part_layer = df[df["layer"] == layer]
    for method in base.ordered_methods(part_layer["method"]):
        part = part_layer[part_layer["method"] == method].sort_values("epoch")
        ax.plot(
            part["epoch"],
            part["unique_pctg"],
            color=base.COLORS.get(method, "#333333"),
            marker=base.MARKERS.get(method, "o"),
            linestyle=base.LINESTYLES.get(method, "-"),
            linewidth=1.6,
            markersize=3.5,
            label=base.method_label(method),
        )
    ax.set_title(dataset)
    ax.set_xlabel("Epoca")
    ax.set_ylabel(f"Estados unicos {layer} (%)")
    ax.set_ylim(-2, 104)


def plot_noise_dataset(dataset: str) -> None:
    fig, ax = plt.subplots(figsize=(10.3, 6.0))
    base.plot_noise_curves(
        ax,
        base.data_noise(dataset),
        f"Robustez ante ruido en datos - {dataset}",
        "Desviacion estandar del ruido en imagen (sigma)",
    )
    save(fig, dataset_dir(dataset) / figure_name(dataset, "data_noise_accuracy_curves"))


def plot_data_noise_combined_ax(ax: plt.Axes, dataset: str) -> None:
    base.plot_noise_curves(
        ax,
        base.data_noise(dataset),
        dataset,
        "sigma datos",
        y_label="Accuracy (%)",
    )
    legend = ax.get_legend()
    if legend:
        legend.remove()


def plot_weight_noise_dataset(dataset: str, layer: str) -> None:
    fig, ax = plt.subplots(figsize=(10.3, 6.0))
    base.plot_noise_curves(
        ax,
        base.weight_noise(dataset, layer),
        f"Robustez ante ruido en pesos ({layer}) - {dataset}",
        "Desviacion estandar del ruido en pesos (sigma)",
    )
    save(fig, dataset_dir(dataset) / figure_name(dataset, f"weight_noise_accuracy_curves_{layer}"))


def plot_weight_noise_combined_ax(ax: plt.Axes, dataset: str, layer: str) -> None:
    base.plot_noise_curves(
        ax,
        base.weight_noise(dataset, layer),
        dataset,
        "sigma pesos",
        y_label="Accuracy (%)",
    )
    legend = ax.get_legend()
    if legend:
        legend.remove()


def plot_weight_retention_ax(ax: plt.Axes, dataset: str, show_legend: bool = False) -> None:
    rows = []
    for layer in LAYERS:
        ret = base.retention_at(base.weight_noise(dataset, layer), sigma=0.15)
        ret["layer"] = layer
        rows.append(ret)
    df = pd.concat(rows, ignore_index=True)
    methods = base.ordered_methods(df["method"])
    x = np.arange(len(methods))
    width = 0.23
    for i, layer in enumerate(LAYERS):
        part = df[df["layer"] == layer].set_index("method").reindex(methods)
        ax.bar(
            x + (i - 1) * width,
            part["retention"],
            width,
            label=layer,
            color=base.LAYER_COLORS[layer],
            edgecolor="black",
            linewidth=0.35,
            hatch=["", "//", ".."][i],
        )
    ax.set_title(dataset)
    ax.set_ylabel("Retencion (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([base.method_label(m) for m in methods], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, min(115, max(105, df["retention"].max() + 10)))
    if show_legend:
        ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.25), fontsize=8)


def radar_data(dataset: str) -> pd.DataFrame:
    acc = base.final_accuracy_summary(dataset).set_index("method")
    data_ret = base.retention_at(base.data_noise(dataset), sigma=float(base.data_noise(dataset)["sigma"].max())).set_index("method")
    weight_ret = base.retention_at(base.weight_noise(dataset, "fc1"), sigma=0.15).set_index("method")
    common = [m for m in base.ordered_methods(acc.index) if m in data_ret.index and m in weight_ret.index]
    raw = pd.DataFrame(
        {
            "Accuracy": acc.loc[common, "val_acc"].astype(float),
            "Robustez datos": data_ret.loc[common, "retention"].astype(float),
            "Robustez pesos": weight_ret.loc[common, "retention"].astype(float),
        },
        index=common,
    )
    return (raw - raw.min()) / (raw.max() - raw.min()) * 100.0


def plot_radar_ax(ax: plt.Axes, dataset: str, legend: bool = False) -> None:
    norm = radar_data(dataset)
    labels = list(norm.columns)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    for method in norm.index:
        values = norm.loc[method].tolist()
        values += values[:1]
        ax.plot(
            angles,
            values,
            label=base.method_label(method),
            color=base.COLORS.get(method, "#333333"),
            marker=base.MARKERS.get(method, "o"),
            linestyle=base.LINESTYLES.get(method, "-"),
            linewidth=1.8,
            markersize=3.5,
        )
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_title(dataset, pad=16)
    if legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05), fontsize=8)


def plot_hyperparameter_profiles(dataset: str) -> None:
    paths = sorted((ROOT / "accuracy" / dataset).glob("data_CNN_*.csv"))
    n = len(paths)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.5 * rows), squeeze=False)
    for ax, path in zip(axes.ravel(), paths):
        df = base.read_csv(path)
        method = str(df["reg_method"].iloc[0])
        profile = df.groupby("reg_val")["val_acc"].max().reset_index().sort_values("reg_val")
        ax.plot(
            profile["reg_val"].astype(float),
            profile["val_acc"],
            color=base.COLORS.get(method, "#333333"),
            marker=base.MARKERS.get(method, "o"),
            linewidth=1.8,
        )
        if method in {"L1", "L2"}:
            ax.set_xscale("symlog", linthresh=1e-4)
        ax.set_title(base.method_label(method))
        ax.set_xlabel("reg_val")
        ax.set_ylabel("Mejor val_acc")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"Perfiles de busqueda de hiperparametros - {dataset}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, dataset_dir(dataset) / figure_name(dataset, "hyperparameter_search_profiles"))


def plot_hyperparameter_overlay_ax(ax: plt.Axes, dataset: str) -> None:
    for path in sorted((ROOT / "accuracy" / dataset).glob("data_CNN_*.csv")):
        df = base.read_csv(path)
        method = str(df["reg_method"].iloc[0])
        if method == "Baseline":
            continue
        profile = df.groupby("reg_val")["val_acc"].max().reset_index().sort_values("reg_val")
        x_values = np.arange(len(profile))
        ax.plot(
            x_values,
            profile["val_acc"],
            color=base.COLORS.get(method, "#333333"),
            marker=base.MARKERS.get(method, "o"),
            linestyle=base.LINESTYLES.get(method, "-"),
            linewidth=1.5,
            markersize=3.5,
            label=base.method_label(method),
        )
    ax.set_title(dataset)
    ax.set_xlabel("Indice de hiperparametro probado")
    ax.set_ylabel("Mejor val_acc")


def render_histogram_pdf(dataset: str) -> list[Path]:
    pdf = ROOT / "histograms" / dataset / f"activation_histograms_multilayer_{dataset}.pdf"
    if not pdf.exists():
        return []
    prefix = dataset_dir(dataset) / f"{dataset}_activation_histograms_multilayer"
    try:
        subprocess.run(
            ["pdftoppm", "-png", "-r", "180", str(pdf), str(prefix)],
            check=True,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"No se pudo renderizar {pdf}: {exc}")
        return []
    return sorted(dataset_dir(dataset).glob(f"{dataset}_activation_histograms_multilayer*.png"))


def plot_dataset_figures(dataset: str) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    plot_ranking_ax(ax, dataset)
    fig.suptitle(f"Ranking final de accuracy - {dataset}", fontsize=15, fontweight="bold")
    save(fig, dataset_dir(dataset) / figure_name(dataset, "ranking_val_acc"))

    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    plot_accuracy_curves_ax(ax, dataset)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.suptitle(f"Curvas de accuracy de validacion - {dataset}", fontsize=15, fontweight="bold")
    save(fig, dataset_dir(dataset) / figure_name(dataset, "best_validation_accuracy_curves"))

    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    plot_generalization_gap_ax(ax, dataset)
    fig.suptitle(f"Brecha de generalizacion - {dataset}", fontsize=15, fontweight="bold")
    save(fig, dataset_dir(dataset) / figure_name(dataset, "generalization_gap_ranking"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    im0 = plot_data_aug_heatmap_ax(axes[0, 0], dataset, "val_acc")
    im1 = plot_data_aug_heatmap_ax(axes[0, 1], dataset, "train_acc")
    fig.colorbar(im0, ax=axes[0, 0], label="val_acc")
    fig.colorbar(im1, ax=axes[0, 1], label="train_acc")
    plot_data_aug_summary_ax(axes[1, 0], dataset)
    axes[1, 1].axis("off")
    axes[1, 1].text(0.5, 0.5, "Resumen de DataAug\n(mejor val_acc y gap final)", ha="center", va="center")
    fig.suptitle(f"Analisis de Data Augmentation - {dataset}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, dataset_dir(dataset) / figure_name(dataset, "data_aug_heatmaps_and_gap"))

    scatter_specs = [
        ("activation_val_acc_vs_unique_pctg", plot_unique_scatter_ax, "val_acc frente a estados unicos"),
        ("activation_val_acc_vs_entropy", plot_entropy_scatter_ax, "val_acc frente a entropia"),
        ("activation_val_acc_vs_dispersion_ratio", plot_dispersion_scatter_ax, "val_acc frente a dispersion"),
        ("activation_unique_vs_dispersion_ratio", plot_unique_vs_dispersion_ax, "Estados unicos frente a dispersion"),
    ]
    for slug, fn, title in scatter_specs:
        fig, ax = plt.subplots(figsize=(9.2, 6.0))
        fn(ax, dataset)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8)
        fig.suptitle(f"{title} - {dataset}", fontsize=15, fontweight="bold")
        save(fig, dataset_dir(dataset) / figure_name(dataset, slug))

    plot_temporal_layers(dataset)
    plot_noise_dataset(dataset)
    for layer in LAYERS:
        plot_weight_noise_dataset(dataset, layer)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    plot_weight_retention_ax(ax, dataset, show_legend=True)
    fig.suptitle(f"Retencion ante ruido en pesos por capa - {dataset}", fontsize=15, fontweight="bold")
    save(fig, dataset_dir(dataset) / figure_name(dataset, "weight_noise_retention_by_layer_sigma_0_15"))

    fig, ax = plt.subplots(figsize=(7.2, 7.2), subplot_kw={"polar": True})
    plot_radar_ax(ax, dataset, legend=True)
    fig.suptitle(f"Radar multidimensional - {dataset}", fontsize=15, fontweight="bold")
    save(fig, dataset_dir(dataset) / figure_name(dataset, "multidimensional_regularizer_radar"))

    plot_hyperparameter_profiles(dataset)
    render_histogram_pdf(dataset)


def combined_grid(slug: str, title: str, plot_fn, figsize=(13.0, 9.0), sharex=False, sharey=False) -> None:
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=sharex, sharey=sharey)
    for ax, dataset in zip(axes.ravel(), DATASETS):
        plot_fn(ax, dataset)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=8.5)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04 if handles else 0, 1, 0.94])
    save(fig, COMBINED_DIR / f"combined_{slug}_2x2.png")


def combined_data_aug_heatmaps(metric: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    last_im = None
    for ax, dataset in zip(axes.ravel(), DATASETS):
        last_im = plot_data_aug_heatmap_ax(ax, dataset, metric)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), label=metric, shrink=0.88)
    fig.suptitle(f"DataAug - mapas de calor de {metric} (2x2 datasets)", fontsize=16, fontweight="bold")
    save(fig, COMBINED_DIR / f"combined_data_aug_{metric}_heatmaps_2x2.png")


def combined_temporal_fc1() -> None:
    for layer in LAYERS:
        combined_grid(
            f"temporal_{layer}_unique_pctg",
            f"Evolucion temporal de estados unicos en {layer}",
            lambda ax, dataset, layer=layer: plot_temporal_layer_ax(ax, dataset, layer),
            figsize=(13.5, 8.5),
            sharey=True,
        )


def combined_weight_retention() -> None:
    combined_grid(
        "weight_noise_retention_by_layer_sigma_0_15",
        "Retencion ante ruido en pesos por capa (sigma=0.15)",
        lambda ax, dataset: plot_weight_retention_ax(ax, dataset, show_legend=False),
        figsize=(13.5, 8.8),
    )


def combined_radar() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={"polar": True})
    for ax, dataset in zip(axes.ravel(), DATASETS):
        plot_radar_ax(ax, dataset, legend=False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), fontsize=8)
    fig.suptitle("Radar multidimensional por dataset", fontsize=16, fontweight="bold")
    save(fig, COMBINED_DIR / "combined_multidimensional_regularizer_radar_2x2.png")


def combined_histograms() -> None:
    images = []
    for dataset in DATASETS:
        rendered = sorted(dataset_dir(dataset).glob(f"{dataset}_activation_histograms_multilayer*.png"))
        if rendered:
            images.append((dataset, rendered[0]))
    if not images:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, item in zip(axes.ravel(), images):
        dataset, image_path = item
        ax.imshow(mpimg.imread(image_path))
        ax.set_title(dataset)
        ax.axis("off")
    for ax in axes.ravel()[len(images):]:
        ax.axis("off")
    fig.suptitle("Histogramas de activaciones - primera pagina por dataset", fontsize=16, fontweight="bold")
    save(fig, COMBINED_DIR / "combined_activation_histograms_first_page_2x2.png")


def plot_combined_figures() -> None:
    combined_grid("ranking_val_acc", "Ranking final de accuracy por dataset", plot_ranking_ax, figsize=(13.5, 9))
    combined_grid(
        "best_validation_accuracy_curves",
        "Curvas de accuracy de validacion por dataset",
        plot_accuracy_curves_ax,
        figsize=(13.5, 8.8),
    )
    combined_grid(
        "generalization_gap_ranking",
        "Brecha de generalizacion por dataset",
        plot_generalization_gap_ax,
        figsize=(13.5, 9),
    )
    combined_grid(
        "hyperparameter_search_overlay",
        "Perfiles de busqueda de hiperparametros por dataset",
        plot_hyperparameter_overlay_ax,
        figsize=(13.5, 8.8),
    )
    combined_grid(
        "data_aug_best_val_acc_and_gap",
        "DataAug: mejor val_acc y gap final por dataset",
        plot_data_aug_summary_ax,
        figsize=(13.5, 8.5),
    )
    combined_data_aug_heatmaps("val_acc")
    combined_data_aug_heatmaps("train_acc")
    combined_grid(
        "activation_val_acc_vs_unique_pctg",
        "val_acc frente a estados unicos en fc1",
        plot_unique_scatter_ax,
        figsize=(13.5, 9),
    )
    combined_grid(
        "activation_val_acc_vs_entropy",
        "val_acc frente a entropia en fc1",
        plot_entropy_scatter_ax,
        figsize=(13.5, 9),
    )
    combined_grid(
        "activation_val_acc_vs_dispersion_ratio",
        "val_acc frente a ratio de dispersion en fc1",
        plot_dispersion_scatter_ax,
        figsize=(13.5, 9),
    )
    combined_grid(
        "activation_unique_vs_dispersion_ratio",
        "Estados unicos frente a ratio de dispersion en fc1",
        plot_unique_vs_dispersion_ax,
        figsize=(13.5, 9),
    )
    combined_temporal_fc1()
    combined_grid(
        "data_noise_accuracy_curves",
        "Robustez ante ruido en datos por dataset",
        plot_data_noise_combined_ax,
        figsize=(14.5, 9),
    )
    for layer in LAYERS:
        combined_grid(
            f"weight_noise_accuracy_curves_{layer}",
            f"Robustez ante ruido en pesos {layer} por dataset",
            lambda ax, dataset, layer=layer: plot_weight_noise_combined_ax(ax, dataset, layer),
            figsize=(14.5, 9),
        )
    combined_weight_retention()
    combined_radar()
    combined_histograms()


def main() -> None:
    base.configure_style()
    ensure_extended_dirs()
    # Regenerate the original thesis figures/tables too; this keeps the legacy
    # outputs in sync with the EarlyStopping estimates.
    base.generate_figures()
    base.generate_tables()
    for dataset in DATASETS:
        plot_dataset_figures(dataset)
    plot_combined_figures()
    print(f"Figuras por dataset en: {FIG_DIR}")
    print(f"Figuras combinadas 2x2 en: {COMBINED_DIR}")


if __name__ == "__main__":
    main()
