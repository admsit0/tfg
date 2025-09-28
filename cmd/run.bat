@echo off

@REM echo Running baseline experiment
@REM python src/main.py --config configs/baseline.yaml
@REM echo Saved results correctly in output dir
@REM echo .

echo Running cross-validation experiment
python src/main.py --config configs/fashion_cv_dropout_l1.yaml
echo Saved cross-validation results correctly in output dir
echo .
@REM echo .

@REM echo Running label smoothing experiment
@REM python src/main.py --config configs/lab_sm.yaml
@REM echo Saved results correctly in output dir
