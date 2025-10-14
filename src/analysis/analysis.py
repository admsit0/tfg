"""Analysis helpers for hidden-state experiments.

This module provides fast utilities to:
- count unique hidden states using rounding + np.unique (fast)
- compute distance statistics (nearest-neighbor, pairwise summaries)
- correlate distance / unique-state metrics with overfitting measures
- plotting helpers to improve presentation

The functions are written to be small, well-typed and easy to call from
`analysis.ipynb` (the notebook should import and call these helpers).

Assumptions / notes:
- Inputs are numpy arrays (2D: samples x features) or pandas DataFrames.
- For large numbers of samples, pairwise operations sample to remain fast.
"""
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns



def as_numpy(X: Any) -> np.ndarray:
    """Convert array-like or DataFrame to a 2D numpy array."""
    if isinstance(X, pd.DataFrame):
        return X.values
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return X


def count_unique_states_rounding(X: Any, decimals: int = 5, normalize_l2: bool = True) -> Tuple[int, np.ndarray]:
    """Count unique rows in X by rounding features to `decimals` and using np.unique.

    This is fast and deterministic. It assumes numerical stability within the
    chosen decimal tolerance. If you need a distance-based clustering, use a
    different function (or set smaller `decimals`).

    Returns (n_unique, representatives) where representatives is an array
    with one representative per unique rounded row.
    """
    Xn = as_numpy(X).astype(float)
    if normalize_l2:
        norms = np.linalg.norm(Xn, axis=1, keepdims=True)
        # avoid division by zero
        norms[norms == 0] = 1.0
        Xn = Xn / norms

    Xr = np.round(Xn, decimals=decimals)
    # np.unique rows: use view trick for speed
    try:
        # For contiguous arrays only
        Xr_view = np.ascontiguousarray(Xr).view(np.dtype((np.void, Xr.dtype.itemsize * Xr.shape[1])))
        uniq_view, idx = np.unique(Xr_view, return_index=True)
        reps = Xr[np.sort(idx)]
    except Exception:
        # Fallback to axis=0 unique (numpy>=1.13)
        reps = np.unique(Xr, axis=0)

    return reps.shape[0], reps


def sample_pairwise_distances(X: Any, max_pairs: int = 200_000, random_state: Optional[int] = 0) -> np.ndarray:
    """Return a sampled array of pairwise Euclidean distances from rows of X.

    For large N the full pairwise matrix is huge; this function samples pairs
    uniformly at random (without replacement) up to `max_pairs` entries.
    """
    Xn = as_numpy(X).astype(float)
    n = Xn.shape[0]
    if n < 2:
        return np.array([])

    # number of possible pairs
    total_pairs = n * (n - 1) // 2
    rng = np.random.default_rng(random_state)
    if total_pairs <= max_pairs:
        # compute full condensed distances using pdist-like method
        from scipy.spatial.distance import pdist

        dists = pdist(Xn, metric="euclidean")
        return dists

    # sample pair indices
    # map a linear index in [0, total_pairs) to pair (i,j) with i<j
    choices = rng.choice(total_pairs, size=max_pairs, replace=False)
    # convert indices
    # algorithm: for k in 0..total_pairs-1, pairs correspond to triangular numbers
    # we'll compute i via solving k < n*i - i*(i+1)/2 etc. vectorized approach
    i = (np.floor((1 + np.sqrt(1 + 8 * choices)) / 2)).astype(int)
    # fix i so that triangular number base is <= choice
    tri = i * (i - 1) // 2
    j = choices - tri
    # adjust j so that j > i
    j = j + i
    # compute distances
    dists = np.linalg.norm(Xn[i] - Xn[j], axis=1)
    return dists


def nearest_neighbor_distances(X: Any, k: int = 1) -> np.ndarray:
    """Compute distances to the k-th nearest neighbor for each row in X.

    Uses sklearn's NearestNeighbors if available, otherwise falls back to
    a pairwise approach (may be slower for large n).
    """
    Xn = as_numpy(X).astype(float)
    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
        nn.fit(Xn)
        dists, _ = nn.kneighbors(Xn)
        # dists[:,0] is zero (self), so take k-th neighbor at index k
        return dists[:, k]
    except Exception:
        # fallback: compute full pairwise distances (O(n^2)) but small n expected
        from scipy.spatial.distance import cdist

        D = cdist(Xn, Xn)
        np.fill_diagonal(D, np.inf)
        kth = np.partition(D, kth=k, axis=1)[:, k]
        return kth


def correlate_with_overfitting(df: pd.DataFrame, state_col: str = "num_states", train_col: str = "train_acc", val_col: str = "val_acc") -> Dict[str, Any]:
    """Compute correlations between state metrics and overfitting.

    The function computes:
    - overfit = train_col - val_col
    - Pearson and Spearman correlations between state_col and overfit
    - Returns a dict with statistics and the augmented DataFrame.
    """
    if state_col not in df.columns or train_col not in df.columns or val_col not in df.columns:
        raise ValueError("DataFrame must contain state, train and val columns")

    df2 = df.copy()
    df2 = df2.dropna(subset=[state_col, train_col, val_col])
    df2["overfit"] = df2[train_col] - df2[val_col]

    from scipy.stats import pearsonr, spearmanr

    res = {}
    try:
        r_pearson, p_pearson = pearsonr(df2[state_col], df2["overfit"]) if len(df2) > 1 else (np.nan, np.nan)
    except Exception:
        r_pearson, p_pearson = np.nan, np.nan
    try:
        r_spear, p_spear = spearmanr(df2[state_col], df2["overfit"]) if len(df2) > 1 else (np.nan, np.nan)
    except Exception:
        r_spear, p_spear = np.nan, np.nan

    res.update({
        "pearson_r": float(r_pearson) if not np.isnan(r_pearson) else None,
        "pearson_p": float(p_pearson) if not np.isnan(p_pearson) else None,
        "spearman_r": float(r_spear) if not np.isnan(r_spear) else None,
        "spearman_p": float(p_spear) if not np.isnan(p_spear) else None,
        "n": int(len(df2)),
        "df": df2,
    })
    return res


def pretty_summary(df: pd.DataFrame, state_col: str = "num_states", train_col: str = "train_acc", val_col: str = "val_acc") -> pd.DataFrame:
    """Return a tidy aggregated summary grouped by parameter (if present) or overall.

    The function attempts to be robust to input formats produced by the
    notebook's aggregation step.
    """
    if "param" in df.columns:
        agg = df.groupby("param").agg(
            num_states_mean=(state_col, "mean"),
            num_states_std=(state_col, "std"),
            train_acc_mean=(train_col, "mean"),
            train_acc_std=(train_col, "std"),
            val_acc_mean=(val_col, "mean"),
            val_acc_std=(val_col, "std"),
        )
        return agg.reset_index()
    else:
        return pd.DataFrame({
            "num_states_mean": [df[state_col].mean()],
            "num_states_std": [df[state_col].std()],
            "train_acc_mean": [df[train_col].mean()],
            "train_acc_std": [df[train_col].std()],
            "val_acc_mean": [df[val_col].mean()],
            "val_acc_std": [df[val_col].std()],
        })


def plot_states_vs_accuracy(agg_df: pd.DataFrame, eps: Optional[float] = None, ax: Optional[Any] = None) -> Any:
    """Plot num_states (mean ± std) vs train/validation accuracy with labels.

    Requires matplotlib (and seaborn if you want nicer style). Returns the
    matplotlib Axes object.
    """
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")
    if sns is not None:
        sns.set(style="whitegrid")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    x = agg_df["num_states_mean"].values
    xerr = agg_df.get("num_states_std", np.zeros_like(x)).values

    ax.errorbar(x, agg_df["train_acc_mean"].values, xerr=xerr, yerr=agg_df.get("train_acc_std", 0).values,
                fmt="o-", label="Train")
    ax.errorbar(x, agg_df["val_acc_mean"].values, xerr=xerr, yerr=agg_df.get("val_acc_std", 0).values,
                fmt="s-", label="Validation")

    for i, row in agg_df.iterrows():
        label = row.get("param", str(i))
        ax.text(row["num_states_mean"], row["val_acc_mean"], label, fontsize=8)

    ax.set_xlabel(f"Número de hidden states{(f' (eps={eps})' if eps is not None else '')}")
    ax.set_ylabel("Accuracy (mean ± std)")
    ax.set_title("Num Hidden States vs Accuracy (última época)")
    ax.legend()
    return ax


if __name__ == "__main__":
    # quick smoke demonstration when running module directly
    print("placeholder analysis module: no direct demo available. Import from notebook.")
