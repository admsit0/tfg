import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

# =========================================================
# CONFIG
# =========================================================

BASE_BN = "outputs-higherGran/data"
BASE_NO_BN = "outputs-higherGran-noBatchnorm/data"

OUTPUT_PDF = "regularization_analysis_comparison_v2.pdf"

METHODS = ["Dropout", "DataAug", "L1", "L2"]

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--"
})

# =========================================================
# DATA LOADING
# =========================================================

def load_experiment(base_path):
    dfs = {}
    for m in METHODS:
        path = os.path.join(base_path, f"data_CNN_{m}.csv")
        dfs[m] = pd.read_csv(path)
    return dfs


# =========================================================
# SUMMARY COMPUTATION
# =========================================================

def compute_summary(df):
    """
    Returns:
    - df_best_per_reg
    - df_final_epoch
    - global_best_row
    """
    best_rows = []
    final_rows = []

    for reg_val, group in df.groupby("reg_val"):
        group = group.sort_values("epoch")

        # Best validation
        best_idx = group["val_acc"].idxmax()
        best_row = group.loc[best_idx].copy()
        best_row["gap"] = best_row["train_acc"] - best_row["val_acc"]
        best_rows.append(best_row)

        # Final epoch
        final_row = group.iloc[-1].copy()
        final_row["gap"] = final_row["train_acc"] - final_row["val_acc"]
        final_rows.append(final_row)

    df_best = pd.DataFrame(best_rows)
    df_final = pd.DataFrame(final_rows)

    global_best = df_best.loc[df_best["val_acc"].idxmax()]

    return df_best, df_final, global_best


# =========================================================
# HEATMAP
# =========================================================

def plot_heatmap(ax, df, metric, title):
    pivot = df.pivot(index="epoch", columns="reg_val", values=metric)
    im = ax.imshow(pivot.values,
                   aspect="auto",
                   origin="lower",
                   interpolation="nearest")

    ax.set_title(title)
    ax.set_xlabel("reg_val index")
    ax.set_ylabel("epoch")

    return im


# =========================================================
# BLOCK GENERATION
# =========================================================

def generate_block(pdf, title, base_path, global_vmin, global_vmax):

    dfs = load_experiment(base_path)

    # -----------------------------
    # Executive summary page
    # -----------------------------
    summary_rows = []

    for m, df in dfs.items():
        df_best, df_final, global_best = compute_summary(df)

        summary_rows.append({
            "Method": m,
            "Best Val Acc": global_best["val_acc"],
            "Reg Val": global_best["reg_val"],
            "Epoch": global_best["epoch"],
            "Gap": global_best["gap"]
        })

    summary_df = pd.DataFrame(summary_rows)

    fig = plt.figure(figsize=(12, 6))
    plt.axis("off")
    plt.title(f"{title} — Executive Summary", fontsize=14)

    summary_display = summary_df.copy()

    # Format numeric columns safely (no deprecated applymap)
    for col in summary_display.columns:
        if pd.api.types.is_numeric_dtype(summary_display[col]):
            summary_display[col] = summary_display[col].map(lambda x: f"{x:.4f}")

    table = plt.table(
        cellText=summary_display.values,
        colLabels=summary_display.columns,
        loc="center"
    )


    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    pdf.savefig(fig)
    plt.close(fig)

    # -----------------------------
    # Per-method analysis
    # -----------------------------
    for m, df in dfs.items():

        df_best, df_final, global_best = compute_summary(df)

        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.3)

        # Heatmap val
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = plot_heatmap(ax1, df, "val_acc", "Val Accuracy")
        im1.set_clim(global_vmin, global_vmax)
        fig.colorbar(im1, ax=ax1)

        # Heatmap train
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = plot_heatmap(ax2, df, "train_acc", "Train Accuracy")
        im2.set_clim(global_vmin, global_vmax)
        fig.colorbar(im2, ax=ax2)

        # Best per reg
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(df_best["reg_val"], df_best["val_acc"], marker="o")
        ax3.set_title("Best Val Acc per reg_val")
        ax3.set_xlabel("reg_val")
        ax3.set_ylabel("val_acc")

        if m in ["L1", "L2"]:
            ax3.set_xscale("log")

        # Gap final
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(df_final["reg_val"], df_final["gap"], marker="o")
        ax4.axhline(0, linestyle="--")
        ax4.set_title("Final Generalization Gap")
        ax4.set_xlabel("reg_val")
        ax4.set_ylabel("gap")

        if m in ["L1", "L2"]:
            ax4.set_xscale("log")

        fig.suptitle(f"{title} — {m}", fontsize=14)
        pdf.savefig(fig)
        plt.close(fig)


# =========================================================
# CROSS COMPARISON BN vs NO BN
# =========================================================

def generate_cross_comparison(pdf, dfs_bn, dfs_no_bn):

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.3)

    for i, m in enumerate(METHODS):
        ax = fig.add_subplot(gs[i // 2, i % 2])

        df_best_bn, _, _ = compute_summary(dfs_bn[m])
        df_best_no_bn, _, _ = compute_summary(dfs_no_bn[m])

        ax.plot(df_best_bn["reg_val"],
                df_best_bn["val_acc"],
                marker="o",
                label="BatchNorm")

        ax.plot(df_best_no_bn["reg_val"],
                df_best_no_bn["val_acc"],
                marker="s",
                label="No BatchNorm")

        if m in ["L1", "L2"]:
            ax.set_xscale("log")

        ax.set_title(m)
        ax.set_xlabel("reg_val")
        ax.set_ylabel("Best val_acc")
        ax.legend()

    fig.suptitle("Direct Comparison — BatchNorm vs No BatchNorm", fontsize=14)
    pdf.savefig(fig)
    plt.close(fig)


# =========================================================
# MAIN
# =========================================================

def main():

    dfs_bn = load_experiment(BASE_BN)
    dfs_no_bn = load_experiment(BASE_NO_BN)

    # Global accuracy limits for fair comparison
    all_vals = []
    for dfs in [dfs_bn, dfs_no_bn]:
        for df in dfs.values():
            all_vals.append(df["train_acc"].values)
            all_vals.append(df["val_acc"].values)

    all_vals = np.concatenate(all_vals)
    global_vmin = all_vals.min()
    global_vmax = all_vals.max()

    with PdfPages(OUTPUT_PDF) as pdf:

        # BLOCK 1 — BN
        generate_block(
            pdf,
            "WITH BATCHNORM",
            BASE_BN,
            global_vmin,
            global_vmax
        )

        # BLOCK 2 — NO BN
        generate_block(
            pdf,
            "WITHOUT BATCHNORM",
            BASE_NO_BN,
            global_vmin,
            global_vmax
        )

        # BLOCK 3 — Direct comparison
        generate_cross_comparison(pdf, dfs_bn, dfs_no_bn)

    print(f"PDF generado: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
