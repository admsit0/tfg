echo off

echo Running baseline experiment
python src/train.py --config configs/baseline.yaml
echo Saved results correctly in output dir
echo.

echo.
echo Running dropout experiment
python src/train.py --config configs/dropout.yaml
echo Saved results correctly in output dir
echo.

echo.
echo Running L1 regularization experiment
python src/train.py --config configs/l1_reg.yaml
echo Saved results correctly in output dir
echo.

echo.
echo Running L2 resultarization experiment
python src/train.py --config configs/l2_reg.yaml
echo Saved results correctly in output dir
echo.

echo.
echo Running label smoothing experiment
python src/train.py --config configs/lab_sm.yaml
echo Saved results correctly in output dir
