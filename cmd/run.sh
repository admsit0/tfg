#!/bin/bash

echo "Running baseline experiment"
python src/train.py --config configs/baseline.yaml
echo "Saved results correctly in output dir"
echo -e "\n"

echo -e "\n"
echo "Running dropout experiment"
python src/train.py --config configs/dropout.yaml
echo "Saved results correctly in output dir"
echo -e "\n"

echo -e "\n"
echo "Running L1 regularization experiment"
python src/train.py --config configs/l1_reg.yaml
echo "Saved results correctly in output dir"
echo -e "\n"

echo -e "\n"
echo "Running L2 regularization experiment"
python src/train.py --config configs/l2_reg.yaml
echo "Saved results correctly in output dir"
echo -e "\n"

echo -e "\n"
echo "Running label smoothing regularization experiment"
python src/train.py --config configs/lab_sm.yaml
echo "Saved results correctly in output dir"
