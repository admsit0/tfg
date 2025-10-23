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
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


def _make_regularizer_plots(out_base: Path, df: 'pd.DataFrame'):
	"""Create per-regularizer and combined scatter plots from a summary DataFrame.

	df columns expected: method_dir, method (dropout|l1|l2|baseline|other), fold, train_acc, val_acc
	"""
	viz_dir = out_base / 'viz'
	viz_dir.mkdir(parents=True, exist_ok=True)

	# Normalize fold to an integer index when possible
	def fold_to_idx(f):
		if pd.isna(f) or f is None:
			return None
		try:
			s = str(f).lower()
			if s.startswith('fold'):
				return int(s.replace('fold', '').strip())
			return int(s)
		except Exception:
			return None

	df = df.copy()
	df['fold_idx'] = df['fold'].map(fold_to_idx)

	# Per-method plots
	for method_name, sub in df.groupby('method'):
		if sub.empty:
			continue
		fig, ax = plt.subplots(figsize=(8, 5))
		# Plot train and val per fold, with small jitter for visibility
		x = sub['fold_idx'].fillna(0).values
		ax.scatter(x - 0.05, sub['train_acc'], label='Train', marker='o', color='tab:blue', alpha=0.8)
		ax.scatter(x + 0.05, sub['val_acc'], label='Val', marker='s', color='tab:orange', alpha=0.8)
		# connect pairs when fold is available
		for _, r in sub.dropna(subset=['fold_idx']).iterrows():
			ax.plot([r['fold_idx'] - 0.05, r['fold_idx'] + 0.05], [r['train_acc'], r['val_acc']], color='gray', alpha=0.4)
		ax.set_title(f"Accuracy por fold - Método: {method_name}")
		ax.set_xlabel('Fold')
		ax.set_ylabel('Accuracy última época')
		ax.set_xticks(sorted([v for v in sub['fold_idx'].dropna().unique()]))
		ax.legend()
		fig.tight_layout()
		fig.savefig(viz_dir / f"per_method_{method_name}_scatter.png", dpi=150)
		plt.close(fig)

	# Combined plot: all methods together, small legend (only Train/Val)
	if not df.empty:
		fig, ax = plt.subplots(figsize=(9, 5))
		# Map method to x-position bands for readability
		methods = list(df['method'].unique())
		xpos = {m: i + 1 for i, m in enumerate(sorted(methods))}
		# jitter per fold
		rng = np.random.default_rng(0)
		xs_train = []
		ys_train = []
		xs_val = []
		ys_val = []
		colors = []
		cmap = plt.cm.get_cmap('tab10', len(xpos))
		color_map = {m: cmap(i) for i, m in enumerate(sorted(xpos.keys()))}
		for _, r in df.iterrows():
			m = r['method']
			base = xpos.get(m, 0)
			jitter = (rng.random() - 0.5) * 0.2
			xs_train.append(base - 0.1 + jitter)
			ys_train.append(r['train_acc'])
			xs_val.append(base + 0.1 + jitter)
			ys_val.append(r['val_acc'])
			colors.append(color_map.get(m, (0.5, 0.5, 0.5, 0.7)))
		# Plot with method color, but legend only for Train/Val
		ax.scatter(xs_train, ys_train, c=[colors[i] for i in range(len(xs_train))], marker='o', alpha=0.7, label='Train')
		ax.scatter(xs_val, ys_val, c=[colors[i] for i in range(len(xs_val))], marker='s', alpha=0.7, label='Val')
		ax.set_xticks(list(xpos.values()))
		ax.set_xticklabels(list(sorted(xpos.keys())))
		ax.set_xlim(0.5, len(xpos) + 0.5)
		ax.set_ylabel('Accuracy última época')
		ax.set_title('Accuracy por método (todos juntos)')
		ax.legend()
		fig.tight_layout()
		fig.savefig(viz_dir / "all_methods_scatter.png", dpi=150)
		plt.close(fig)


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

	# find metric files recursively up to 3 levels deep to support seed_*/method/fold*
	candidates = []
	def _collect_metrics(dir_path: Path, prefix: str = "", depth: int = 0, max_depth: int = 3):
		if depth > max_depth:
			return
		# metrics.csv directly here?
		m = dir_path / 'metrics.csv'
		if m.exists():
			name = prefix.rstrip('/') or dir_path.name
			candidates.append((name, m))
			return
		# else scan children
		for child in sorted(dir_path.iterdir()):
			if child.is_dir():
				_collect_metrics(child, f"{prefix}{child.name}/", depth + 1, max_depth)

	for child in sorted(out_base.iterdir()):
		if child.is_dir():
			_collect_metrics(child, f"{child.name}/", 0, 3)

	if not candidates:
		print(f"Visualization: found no metrics.csv files under {out_base}")
		return

	# Save per-run plots as before, and build a summary for new aggregate plots
	summary_rows: List[Dict] = []
	for name, csv_path in candidates:
		try:
			metrics = _read_metrics_csv(csv_path)
			target_dir = csv_path.parent
			safe_name = name.replace('/', '_')
			_make_matplotlib_plots(target_dir, metrics, safe_name)
			_make_plotly_plot(target_dir, metrics, safe_name)
			print(f"Visualization: saved plots for {name} in {target_dir}")

			# build summary row for last epoch
			if metrics:
				last = metrics[-1]
				# infer method folder (parent of fold if present)
				# cases: .../<method>/foldX/metrics.csv OR .../<method>/metrics.csv OR .../seed_*/<method>/foldX/metrics.csv
				parts = list(csv_path.parts)
				# find 'fold*' index if exists
				method = None
				fold = None
				try:
					if csv_path.parent.name.lower().startswith('fold'):
						fold = csv_path.parent.name
						method = csv_path.parent.parent.name
					else:
						# maybe no folds
						method = csv_path.parent.name
				except Exception:
					method = csv_path.parent.name

				# if seeded path like seed_XX present, step one more up
				if method and method.startswith('seed_') and csv_path.parent.name.lower().startswith('fold'):
					method = csv_path.parent.parent.parent.name
				elif method and method.startswith('seed_'):
					method = csv_path.parent.parent.name

				def _method_kind(mname: str) -> str:
					if mname is None:
						return 'unknown'
					lower = mname.lower()
					if lower.startswith('dropout'):
						return 'dropout'
					if lower.startswith('l1'):
						return 'l1'
					if lower.startswith('l2'):
						return 'l2'
					if lower.startswith('baseline'):
						return 'baseline'
					return 'other'

				summary_rows.append({
					'name': name,
					'method_dir': method or 'unknown',
					'method': _method_kind(method or ''),
					'fold': fold,
					'train_acc': last.get('train_acc'),
					'val_acc': last.get('val_acc')
				})
		except Exception as e:
			print(f"Visualization: failed for {name}: {e}")

	# Aggregate plots per regularizer method and a combined plot
	try:
		if summary_rows:
			df = pd.DataFrame(summary_rows)
			_make_regularizer_plots(out_base, df)
	except Exception as e:
		print(f"Visualization: failed to create aggregate plots: {e}")


# ===== Activation manifold visualization (PCA, t-SNE, etc.) ===== #
def make_activation_embedding_plots_for_file(csv_path: Path, methods: Optional[List[str]] = None,
											   max_points: int = 3000, standardize: bool = True,
											   out_dir: Optional[Path] = None, random_state: int = 42) -> Optional[Path]:
	"""Create 2D manifold plots (grid) for a single activations CSV.

	- methods: list like ["pca", "tsne", "isomap", "lle", "spectral", "mds", "umap"]
	- Returns the saved PNG path if created, else None.
	"""
	try:
		import pandas as pd  # local import
		import src.analysis.analysis as analysis
		df = pd.read_csv(csv_path)
		X, true_y, pred_y, correct = analysis.extract_activation_matrix(df)
		emb = analysis.compute_embeddings(X, methods=methods, max_points=max_points, standardize=standardize, random_state=random_state)
		fig = analysis.plot_embedding_grid(emb, true_y=true_y[emb.get('indices')] if true_y is not None else None,
											correct=correct[emb.get('indices')] if correct is not None else None)
		if fig is None:
			return None
		# decide output dir
		if out_dir is None:
			out_dir = csv_path.parent / 'embeddings'
		out_dir.mkdir(parents=True, exist_ok=True)
		out_png = out_dir / f"{csv_path.stem}_embeddings.png"
		fig.savefig(out_png, dpi=150, bbox_inches='tight')
		plt.close(fig)
		return out_png
	except Exception as e:
		print(f"Activation embeddings failed for {csv_path}: {e}")
		return None


if __name__ == '__main__':
	# quick manual runner for debugging
	import json
	cfg = {}
	run_visualization(cfg)
