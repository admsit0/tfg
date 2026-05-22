# Regularization in CNNs — Internal Activation Dynamics

**Thesis:** *Estudio sobre técnicas de regularización en redes neuronales — Cuantificación a partir de la dinámica de activaciones internas*  
**Author:** Adam Maltoni — Grado en Ciencia e Ingeniería de Datos, UAM  
**Supervisor:** Dr. Luis Fernando Lago Fernández

---

## Overview

Comparative study of seven regularization techniques (L1, L2, Dropout, Early Stopping, Gaussian Noise, Data Augmentation, Batch Normalization) applied to a standard CNN trained on CIFAR-10 and SVHN. Analysis spans two dimensions: external performance (validation accuracy) and internal model behaviour (Shannon entropy, dispersion ratio, unique activation states).

---

## Repository structure

```
├── src/
│   ├── training/
│   │   ├── train.py                    # Main training script — all regularizers, grid search
│   │   └── train_data_augmentation.py  # Custom DataAug training variant
│   ├── analysis/
│   │   ├── collect_activations.py      # Extract activations from saved checkpoints
│   │   ├── run_experiment.py           # Main experiment pipeline (entropy, dispersion, states)
│   │   ├── temporal_evolution.py       # Epoch-wise unique_pctg tracking
│   │   ├── activation_histograms.py    # Discretized state-space analysis
│   │   ├── entropy_analysis.py         # Shannon entropy per layer
│   │   ├── dispersion_analysis.py      # Dispersion ratio vs baseline
│   │   └── generate_report.py          # Summary tables and CSV export
│   └── robustness/
│       ├── data_noise.py               # Robustness to Gaussian noise in inputs
│       ├── weight_noise.py             # Flat-minima analysis — weight perturbation
│       └── train_weight_noise.py       # Train with weight noise injection
├── notebooks/
│   ├── activation_analysis.ipynb       # Interactive activation exploration
│   ├── l1_l2_debug.ipynb              # L1/L2 behaviour deep-dive
│   └── state_space_diagnosis.ipynb    # State-space coverage diagnosis
├── scripts/
│   └── generate_figures.py            # Generates all 19 thesis figures → figures_old/
├── results/
│   ├── accuracy/
│   │   ├── cifar10/                   # data_CNN_<method>.csv  (7 files)
│   │   └── svhn/                      # data_CNN_<method>.csv  (7 files)
│   ├── internal_activations/
│   │   ├── cifar10/                   # entropy, dispersion, unique_pctg CSVs
│   │   └── svhn/
│   ├── temporal_evolution/            # Bottleneck_Data_CIFAR10_{Custom,Optimum}.csv
│   ├── robustness_data_noise/         # Data_Noise_Data_CIFAR10_{Custom,Optimum}.csv
│   └── robustness_weight_noise/       # Flat_Minima_Data_CIFAR10_{layer}_{series}.csv
├── figures/                           # Adam's actual experiment PDFs (cifar10/ + svhn/)
├── figures_old/                       # Backup: generated individual scatter plots
├── legacy/                            # Deprecated scripts kept for reference
├── docs/                              # Experiment notes and analysis summaries
├── thesis/
│   ├── drafts/                        # Full-text proposals for §5–§9
│   ├── tables.tex                     # LaTeX tables
│   ├── audit_suggestions.md
│   ├── references_suggested.md        # BibTeX entries (24 references in bibliography.bib)
│   └── writing_plan.md
├── .gitignore
├── requirements.txt
├── README.md
├── USAGE.md                           # Full guide: structure, what's included and why
├── Adam_changes.md                    # Manual changes required (paths, figures, datasets)
└── anexos.md                          # Annex structure plan (all 4 datasets)
```

---

## Execution pipeline

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train — saves checkpoints to results_cnn_*/
python src/training/train.py

# 3. Collect activations from checkpoints
python src/analysis/collect_activations.py

# 4. Run main analysis (entropy, dispersion, unique states)
python src/analysis/run_experiment.py

# 5. Temporal evolution of unique_pctg
python src/analysis/temporal_evolution.py

# 6. Robustness experiments
python src/robustness/data_noise.py
python src/robustness/weight_noise.py

# 7. Generate all thesis figures → figures_old/
python scripts/generate_figures.py
```

---

## Key findings

| Metric | Best method | Value |
|--------|-------------|-------|
| Validation accuracy (CIFAR-10) | Data Augmentation | 0.785 |
| Data noise robustness (σ=0.3) | Gaussian Noise | ~80% retention |
| Weight noise robustness fc1 (σ=0.15) | Dropout | 86.1% retention (Optimum), 79.9% (Custom) |
| Worst weight robustness conv1 | Batch Normalization | ~41% retention (sharp minima) |
| Optimal entropy zone (fc1) | All methods | H ∈ [1.1, 1.8] |
| Optimal dispersion ratio (fc1) | All methods | ratio ∈ [0.27, 0.46] |

L1 is the only technique that **hurts** generalization (−0.008 vs baseline). No single method dominates all dimensions — regularization is a multidimensional property.

---

## Datasets

- **CIFAR-10** — 60,000 RGB 32×32 images, 10 classes (primary dataset)
- **SVHN** — Street View House Numbers, 32×32 RGB (cross-domain validation)

---

## Architecture

CNN: Conv2d(3→32→64→128) + MaxPool 2×2 after each block + FC(2048→128→10)  
Optimizer: Adam · Batch size: 64 · Epochs: 60 · Activation: ReLU
