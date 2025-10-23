#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run the same experiment config multiple times with different random seeds.

- Reads a base YAML config (e.g., configs/fashion_cv_l1_dropoutLows_l2.yaml)
- For each seed, writes a derived config to configs/generated/<name>_seed_<seed>.yaml
- Overrides trainer.seed, cross_validation.seed, and trainer.out_dir to
  save to runs/<base_out>/seed_<seed>/...
- Invokes: python src/main.py --config <generated_yaml>

Usage examples:
  python cmd/run_seeds.py --base-config configs/fashion_cv_l1_dropoutLows_l2.yaml --seeds 0 1 2 3 4
  python cmd/run_seeds.py --base-config configs/fashion_cv_l1_dropoutLows_l2.yaml --seeds 42 43
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def build_seeded_cfg(cfg: dict, seed: int) -> dict:
    cfg2 = dict(cfg)  # shallow copy is fine for our keys below; adjust if needed
    # ensure nested dicts
    cfg2.setdefault('trainer', {})
    cfg2.setdefault('cross_validation', {})

    # set seeds
    cfg2['trainer']['seed'] = int(seed)
    if 'cross_validation' in cfg2 and isinstance(cfg2['cross_validation'], dict):
        cfg2['cross_validation']['seed'] = int(seed)

    # adjust output directory to avoid collisions
    base_out = cfg2['trainer'].get('out_dir', 'runs/experiment')
    base_out_path = Path(base_out)
    # put each seed in a subfolder of the base out_dir
    cfg2['trainer']['out_dir'] = str(base_out_path / f"seed_{seed}")
    return cfg2


def run_once(generated_cfg_path: Path) -> int:
    cmd = [sys.executable, str(REPO_ROOT / 'src' / 'main.py'), '--config', str(generated_cfg_path)]
    print(f"\n=== Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    return proc.returncode


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-config', required=True, help='Path to base YAML config')
    ap.add_argument('--seeds', type=int, nargs='+', required=True, help='List of seeds to run')
    args = ap.parse_args(argv)

    base_cfg_path = Path(args.base_config)
    if not base_cfg_path.exists():
        print(f"Base config not found: {base_cfg_path}")
        return 1

    cfg = load_yaml(base_cfg_path)
    base_name = base_cfg_path.stem
    gen_dir = REPO_ROOT / 'configs' / 'generated'
    gen_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for seed in args.seeds:
        seeded_cfg = build_seeded_cfg(cfg, seed)
        generated_cfg_path = gen_dir / f"{base_name}_seed_{seed}.yaml"
        save_yaml(seeded_cfg, generated_cfg_path)
        print(f"Generated config: {generated_cfg_path}")
        rc = run_once(generated_cfg_path)
        if rc != 0:
            failures.append((seed, rc))

    if failures:
        print(f"\nCompleted with failures for seeds: {failures}")
        return 2
    print("\nAll seed runs completed successfully.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
