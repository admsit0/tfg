"""Small visualization utilities for experiments.

This module provides a simple `run_visualization(cfg)` entry point which
looks for `metrics.csv` files under the experiment output directories and
produces PNG/HTML plots for loss and accuracy over epochs. It's intentionally
lightweight and has no heavy runtime dependencies beyond `matplotlib` and
`plotly` (both are present in `requirements.txt`).

The function accepts the global configuration dictionary so it can read
`trainer.out_dir` and optional visualization settings under `visualization`.
"""
from pathlib import Path
import csv
import json
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

try:
	import plotly.graph_objects as go
	_HAS_PLOTLY = True
except Exception:
	_HAS_PLOTLY = False


def _read_metrics_csv(path: Path) -> List[Dict[str, float]]:
	if not path.exists():
		return []
	rows = []
	with open(path, 'r', newline='') as fh:
		reader = csv.DictReader(fh)
		for r in reader:
			parsed = {}
			for k, v in r.items():
				# try to parse as float, fallback to original
				try:
					parsed[k] = float(v)
				except Exception:
					parsed[k] = v
			rows.append(parsed)
	return rows


def _plot_metrics(ax, epochs, values, label, color=None):
	ax.plot(epochs, values, label=label, color=color)
	ax.scatter(epochs, values, color=color, s=10)


def _make_matplotlib_plots(out_dir: Path, metrics: List[Dict[str, float]], name: str):
	if not metrics:
		return
	epochs = [int(m.get('epoch', i + 1)) for i, m in enumerate(metrics)]
	train_acc = [m.get('train_acc') for m in metrics if 'train_acc' in m]
	val_acc = [m.get('val_acc') for m in metrics if 'val_acc' in m]
	train_loss = [m.get('train_loss') for m in metrics if 'train_loss' in m]
	val_loss = [m.get('val_loss') for m in metrics if 'val_loss' in m]

	fig, axes = plt.subplots(1, 2, figsize=(12, 4))
	if train_acc or val_acc:
		if train_acc:
			_plot_metrics(axes[0], epochs[:len(train_acc)], train_acc, 'train_acc', color='tab:blue')
		if val_acc:
			_plot_metrics(axes[0], epochs[:len(val_acc)], val_acc, 'val_acc', color='tab:orange')
		axes[0].set_title('Accuracy')
		axes[0].set_xlabel('epoch')
		axes[0].set_ylabel('accuracy')
		axes[0].legend()

	if train_loss or val_loss:
		if train_loss:
			_plot_metrics(axes[1], epochs[:len(train_loss)], train_loss, 'train_loss', color='tab:blue')
		if val_loss:
			_plot_metrics(axes[1], epochs[:len(val_loss)], val_loss, 'val_loss', color='tab:orange')
		axes[1].set_title('Loss')
		axes[1].set_xlabel('epoch')
		axes[1].set_ylabel('loss')
		axes[1].legend()

	fig.suptitle(name)
	out_png = out_dir / f'{name}_metrics.png'
	fig.tight_layout(rect=[0, 0.03, 1, 0.95])
	fig.savefig(out_png)
	plt.close(fig)


def _make_plotly_plot(out_dir: Path, metrics: List[Dict[str, float]], name: str):
	if not _HAS_PLOTLY or not metrics:
		return
	epochs = [int(m.get('epoch', i + 1)) for i, m in enumerate(metrics)]
	fig = go.Figure()
	if any('train_acc' in m for m in metrics):
		fig.add_trace(go.Scatter(x=epochs, y=[m.get('train_acc') for m in metrics], mode='lines+markers', name='train_acc'))
	if any('val_acc' in m for m in metrics):
		fig.add_trace(go.Scatter(x=epochs, y=[m.get('val_acc') for m in metrics], mode='lines+markers', name='val_acc'))
	if any('train_loss' in m for m in metrics):
		fig.add_trace(go.Scatter(x=epochs, y=[m.get('train_loss') for m in metrics], mode='lines+markers', name='train_loss', yaxis='y2'))
	if any('val_loss' in m for m in metrics):
		fig.add_trace(go.Scatter(x=epochs, y=[m.get('val_loss') for m in metrics], mode='lines+markers', name='val_loss', yaxis='y2'))

	# add secondary axis for loss
	fig.update_layout(title=name, xaxis_title='epoch')
	# export to HTML
	out_html = out_dir / f'{name}_metrics.html'
	fig.write_html(str(out_html))


def run_visualization(cfg: Dict):
	"""Run visualization for experiments described by cfg.

	This looks for `trainer.out_dir` and inspects immediate subdirectories
	for `metrics.csv` files. For each found metrics file it emits a PNG and an
	optional HTML interactive plot.
	"""
	viz_cfg = cfg.get('visualization', {})
	out_base = cfg.get('trainer', {}).get('out_dir', 'runs')
	out_base = Path(out_base)
	if not out_base.exists():
		print(f"Visualization: no output directory found at {out_base}")
		return

	# find metric files in immediate subdirectories and their 'run' or 'fold' children
	candidates = []
	for child in sorted(out_base.iterdir()):
		if not child.is_dir():
			continue
		# look for metrics.csv directly under child or in a 'run'/'fold*' subdir
		direct = child / 'metrics.csv'
		if direct.exists():
			candidates.append((child.name, direct))
			continue
		# scan subfolders one level deep
		for sub in child.iterdir():
			if not sub.is_dir():
				continue
			f = sub / 'metrics.csv'
			if f.exists():
				candidates.append((f"{child.name}/{sub.name}", f))

	if not candidates:
		print(f"Visualization: found no metrics.csv files under {out_base}")
		return

	for name, csv_path in candidates:
		try:
			metrics = _read_metrics_csv(csv_path)
			target_dir = csv_path.parent
			safe_name = name.replace('/', '_')
			_make_matplotlib_plots(target_dir, metrics, safe_name)
			_make_plotly_plot(target_dir, metrics, safe_name)
			print(f"Visualization: saved plots for {name} in {target_dir}")
		except Exception as e:
			print(f"Visualization: failed for {name}: {e}")


if __name__ == '__main__':
	# quick manual runner for debugging
	import json
	cfg = {}
	run_visualization(cfg)
