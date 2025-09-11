@echo off

echo Running baseline experiment
python src/train.py --config configs/baseline.yaml
echo Saved results correctly in output dir
echo .

echo Running dropout experiment
python src/train.py --config configs/dropout.yaml
echo Saved results correctly in output dir
echo .

@REM echo Running L1 regularization experiment
@REM python src/train.py --config configs/l1_reg.yaml
@REM echo Saved results correctly in output dir
@REM echo .

@REM echo Running L2 regularization experiment
@REM python src/train.py --config configs/l2_reg.yaml
@REM echo Saved results correctly in output dir
@REM echo .

@REM echo Running label smoothing experiment
@REM python src/train.py --config configs/lab_sm.yaml
@REM echo Saved results correctly in output dir
