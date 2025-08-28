#!/usr/bin/env bash
set -e
python -m src.train --config experiments/configs/baseline.yaml
python -m src.train --config experiments/configs/l1_acts_lsmooth.yaml
