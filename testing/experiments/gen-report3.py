import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

# =========================================================
# CONFIGURACIÓN
# =========================================================

# Directorio principal a analizar (Apunta a tu nuevo experimento)
DIR_MAIN = "outputs_moreMethods/data"
OUTPUT_PDF = "outputs_moreMethods/comprehensive_regularization_report.pdf"

# Modo Comparativa Cruzada (Toggle)
COMPARE_MODE = False
DIR_COMPARE = "outputs_moreMethods_noBN/data" # Solo se usa si COMPARE_MODE = True

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--"
})

# =========================================================
# AUTO-DESCUBRIMIENTO Y CARGA DE DATOS
# =========================================================

def get_available_methods(base_path):
    """Detecta automáticamente qué métodos se han evaluado leyendo los CSVs."""
    methods = []
    if not os.path.exists(base_path):
        print(f"⚠️  Ruta no encontrada: {base_path}")
        return methods
        
    for f in os.listdir(base_path):
        if f.startswith("data_CNN_") and f.endswith(".csv"):
            m = f.replace("data_CNN_", "").replace(".csv", "")
            methods.append(m)
    return sorted(methods)

def load_experiment(base_path, methods):
    dfs = {}
    for m in methods:
        path = os.path.join(base_path, f"data_CNN_{m}.csv")
        if os.path.exists(path):
            dfs[m] = pd.read_csv(path)
    return dfs

# =========================================================
# CÁLCULO DE RESÚMENES
# =========================================================

def compute_summary(df):
    """Retorna los mejores datos por parámetro, la época final y el mejor global."""
    best_rows = []
    final_rows = []

    for reg_val, group in df.groupby("reg_val"):
        group = group.sort_values("epoch")

        # Mejor Validación
        best_idx = group["val_acc"].idxmax()
        best_row = group.loc[best_idx].copy()
        best_row["gap"] = best_row["train_acc"] - best_row["val_acc"]
        best_rows.append(best_row)

        # Última época
        final_row = group.iloc[-1].copy()
        final_row["gap"] = final_row["train_acc"] - final_row["val_acc"]
        final_rows.append(final_row)

    df_best = pd.DataFrame(best_rows).sort_values("reg_val")
    df_final = pd.DataFrame(final_rows).sort_values("reg_val")
    
    global_best = df_best.loc[df_best["val_acc"].idxmax()]

    return df_best, df_final, global_best

# =========================================================
# GENERACIÓN DE GRÁFICOS
# =========================================================

def plot_heatmap(ax, df, metric, title):
    """Dibuja un heatmap mapeando correctamente reg_val y epochs a los ejes."""
    # Pivotar datos: Filas=epoch, Columnas=reg_val
    pivot = df.pivot(index="epoch", columns="reg_val", values=metric)
    
    # Manejar caso de valor único (ej. BatchNorm)
    if len(pivot.columns) <= 1:
        ax.plot(pivot.index, pivot.iloc[:, 0], marker='o', color='indigo')
        ax.set_title(f"{title} (Evolución 1D)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        return None # No devuelve image map
        
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap='viridis')
    ax.set_title(title)
    
    # Configurar Eje X (reg_val)
    ax.set_xticks(np.arange(len(pivot.columns)))
    # Formatear números para que sean legibles (ej. 0.0001 -> 1e-4)
    x_labels = [f"{x:g}" if x == 0 else (f"{x:.1e}" if x < 0.01 else f"{x:.3g}") for x in pivot.columns]
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlabel("reg_val")
    
    # Configurar Eje Y (epoch) - Mostrar un subconjunto para no saturar
    y_ticks = np.linspace(0, len(pivot.index) - 1, num=min(10, len(pivot.index)), dtype=int)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(pivot.index[y_ticks])
    ax.set_ylabel("epoch")

    return im

def setup_log_scale_safe(ax, method):
    """Aplica escala logarítmica de forma segura soportando el valor 0.0"""
    if method in ["L1", "L2", "GaussianNoise"]:
        # symlog permite escalar logarítmicamente manteniendo un vecindario lineal alrededor de 0
        ax.set_xscale("symlog", linthresh=1e-5) 

def generate_block(pdf, title, base_path, methods, global_vmin, global_vmax):
    dfs = load_experiment(base_path, methods)
    if not dfs: return

    # -----------------------------
    # PÁGINA 1: Resumen Ejecutivo
    # -----------------------------
    summary_rows = []
    for m in methods:
        if m in dfs:
            _, _, global_best = compute_summary(dfs[m])
            summary_rows.append({
                "Method": m,
                "Best Val Acc": global_best["val_acc"],
                "Reg Val": global_best["reg_val"],
                "Epoch": int(global_best["epoch"]),
                "Gap": global_best["gap"]
            })

    summary_df = pd.DataFrame(summary_rows).sort_values(by="Best Val Acc", ascending=False)

    fig = plt.figure(figsize=(12, 6))
    plt.axis("off")
    plt.title(f"{title} — Executive Summary (Sorted by Val Acc)", fontsize=16, fontweight='bold', y=0.9)

    summary_display = summary_df.copy()
    for col in ["Best Val Acc", "Gap", "Reg Val"]:
        if col in summary_display:
            summary_display[col] = summary_display[col].apply(lambda x: f"{x:.5g}")

    table = plt.table(
        cellText=summary_display.values,
        colLabels=summary_display.columns,
        loc="center",
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Estilizar encabezados de tabla
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        elif row % 2 == 0:
            cell.set_facecolor('#f2f2f2')

    pdf.savefig(fig)
    plt.close(fig)

    # -----------------------------
    # PÁGINAS 2..N: Análisis por Método
    # -----------------------------
    for m in methods:
        if m not in dfs: continue
        
        df = dfs[m]
        df_best, df_final, _ = compute_summary(df)

        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.3)

        # 1. Heatmap Validación
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = plot_heatmap(ax1, df, "val_acc", "Val Accuracy Evolution")
        if im1:
            im1.set_clim(global_vmin, global_vmax)
            fig.colorbar(im1, ax=ax1)

        # 2. Heatmap Entrenamiento
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = plot_heatmap(ax2, df, "train_acc", "Train Accuracy Evolution")
        if im2:
            im2.set_clim(global_vmin, global_vmax)
            fig.colorbar(im2, ax=ax2)

        # 3. Best Val Acc per reg_val
        ax3 = fig.add_subplot(gs[1, 0])
        if len(df_best) > 1:
            ax3.plot(df_best["reg_val"], df_best["val_acc"], marker="o", color='darkorange', linewidth=2)
            setup_log_scale_safe(ax3, m)
        else:
            ax3.bar(["Valor Único"], df_best["val_acc"], color='darkorange', width=0.3)
            
        ax3.set_title("Best Val Acc vs Regularization Strength")
        ax3.set_xlabel("reg_val")
        ax3.set_ylabel("Max val_acc")

        # 4. Final Gap
        ax4 = fig.add_subplot(gs[1, 1])
        if len(df_final) > 1:
            ax4.plot(df_final["reg_val"], df_final["gap"], marker="s", color='crimson', linewidth=2)
            setup_log_scale_safe(ax4, m)
        else:
            ax4.bar(["Valor Único"], df_final["gap"], color='crimson', width=0.3)
            
        ax4.axhline(0, linestyle="--", color='black', alpha=0.5)
        ax4.set_title("Final Generalization Gap (Train - Val)")
        ax4.set_xlabel("reg_val")
        ax4.set_ylabel("Gap")

        fig.suptitle(f"Method Analysis: {m}", fontsize=16, fontweight='bold')
        pdf.savefig(fig)
        plt.close(fig)

# =========================================================
# COMPARATIVA CRUZADA (Opcional)
# =========================================================
def generate_cross_comparison(pdf, base_main, base_compare, methods):
    dfs_main = load_experiment(base_main, methods)
    dfs_compare = load_experiment(base_compare, methods)

    # Solo comparar métodos que existen en ambos
    common_methods = [m for m in methods if m in dfs_main and m in dfs_compare]
    if not common_methods: return

    # Título separador
    fig_sep = plt.figure(figsize=(12, 6))
    plt.axis("off")
    plt.text(0.5, 0.5, "Cross-Directory Comparison", ha='center', va='center', fontsize=20, fontweight='bold')
    pdf.savefig(fig_sep)
    plt.close(fig_sep)

    fig = plt.figure(figsize=(14, 4 * ((len(common_methods) + 1) // 2)))
    gs = gridspec.GridSpec((len(common_methods) + 1) // 2, 2, wspace=0.3, hspace=0.4)

    for i, m in enumerate(common_methods):
        ax = fig.add_subplot(gs[i // 2, i % 2])

        df_best_m, _, _ = compute_summary(dfs_main[m])
        df_best_c, _, _ = compute_summary(dfs_compare[m])

        ax.plot(df_best_m["reg_val"], df_best_m["val_acc"], marker="o", label=f"Dir Main")
        ax.plot(df_best_c["reg_val"], df_best_c["val_acc"], marker="s", label=f"Dir Compare")

        setup_log_scale_safe(ax, m)

        ax.set_title(m)
        ax.set_xlabel("reg_val")
        ax.set_ylabel("Best val_acc")
        ax.legend()

    pdf.savefig(fig)
    plt.close(fig)

# =========================================================
# MAIN
# =========================================================
def main():
    print("🔍 Escaneando carpeta en busca de resultados...")
    methods = get_available_methods(DIR_MAIN)
    
    if not methods:
        print("❌ No se encontraron archivos de datos. Verifica la ruta.")
        return

    print(f"✅ Métodos detectados: {', '.join(methods)}")
    
    dfs_main = load_experiment(DIR_MAIN, methods)

    # Calcular límites globales para normalizar el color de los Heatmaps
    all_vals = []
    for df in dfs_main.values():
        all_vals.append(df["train_acc"].values)
        all_vals.append(df["val_acc"].values)

    if all_vals:
        all_vals = np.concatenate(all_vals)
        global_vmin = all_vals.min()
        global_vmax = all_vals.max()
    else:
        global_vmin, global_vmax = 0, 1

    print("📊 Generando Reporte PDF...")
    with PdfPages(OUTPUT_PDF) as pdf:
        
        # 1. Análisis Principal
        generate_block(pdf, "MAIN EXPERIMENT", DIR_MAIN, methods, global_vmin, global_vmax)

        # 2. Comparativa Cruzada (Si está activada)
        if COMPARE_MODE:
            print("🔀 Generando sección de comparativa cruzada...")
            generate_cross_comparison(pdf, DIR_MAIN, DIR_COMPARE, methods)

    print(f"🚀 ¡PDF generado exitosamente en: {OUTPUT_PDF}!")

if __name__ == "__main__":
    main()
    