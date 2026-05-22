@echo off

set DATASETS=SVHN

for %%D in (%DATASETS%) do (
    @REM python src/analysis/run_experiment.py
    python src/analysis/collect_activations.py
    python src/analysis/entropy_analysis.py --dataset %%D
    python src/analysis/dispersion_analysis.py --dataset %%D
    python src/analysis/temporal_evolution.py --dataset %%D
    python src/robustness/weight_noise.py --dataset %%D
    python src/robustness/data_noise.py --dataset %%D

)
