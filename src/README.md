# Source code — execution pipeline

Run all scripts from the **repository root** (`tfg ordered/`). Results are written to `results/` and `figures/` automatically.

## Pipeline order

```
Step 1 — Training
    python src/training/train.py
    python src/training/train_data_augmentation.py   # DataAug custom variant
    → Output: checkpoints/ (model .pth files, one per epoch per config)

Step 2 — Activation collection
    python src/analysis/collect_activations.py
    → Output: results/internal_activations/{cifar10,svhn}/

Step 3 — Main experiment (entropy, dispersion, unique states)
    python src/analysis/run_experiment.py
    → Output: results/accuracy/{cifar10,svhn}/

Step 4 — Temporal evolution of unique_pctg
    python src/analysis/temporal_evolution.py
    → Output: results/temporal_evolution/

Step 5 — Robustness experiments
    python src/robustness/data_noise.py
    → Output: results/robustness_data_noise/

    python src/robustness/weight_noise.py
    → Output: results/robustness_weight_noise/

    python src/robustness/train_weight_noise.py   # Optional: train with weight noise
    → Output: checkpoints/ (additional model)

Step 6 — Generate all thesis figures
    python scripts/generate_figures.py
    → Output: figures/*.pdf  (19 PDFs at 300 dpi)
```

## Helper scripts

| Script | Purpose |
|--------|---------|
| `src/analysis/activation_histograms.py` | Discretized state-space analysis |
| `src/analysis/entropy_analysis.py` | Shannon entropy per layer |
| `src/analysis/dispersion_analysis.py` | Dispersion ratio vs baseline |
| `src/analysis/generate_report.py` | Summary tables and CSV export |
| `src/analysis/temporal_evolution.py` | Epoch-wise unique_pctg tracking |

## Notes

- Fix random seed with `SEEDS = [42]` in training config for reproducibility
- All scripts require CUDA or will fall back to CPU (`torch.cuda.is_available()`)
- Checkpoints are gitignored (large binary files); results CSVs are version-controlled
- See `../requirements.txt` for dependencies
