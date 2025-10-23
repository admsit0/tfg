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
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding, MDS
from sklearn.preprocessing import StandardScaler



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


# ========= Embedding utilities for activation vectors ========= #

def extract_activation_matrix(df: pd.DataFrame, act_prefix: str = "act_",
                              exclude_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """From an activation DataFrame, extract X (activations) and label vectors.

    Returns (X, true_y, pred_y, correct_mask)
    - act columns start with act_prefix; if none, use numeric columns except exclude.
    - true_y comes from 'true_label' if present; pred_y from 'pred_label'.
    - correct_mask is a boolean array where pred == true when both exist.
    """
    if exclude_cols is None:
        exclude_cols = ["pred_label", "true_label"]
    act_cols = [c for c in df.columns if c.startswith(act_prefix)]
    if not act_cols:
        act_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in set(exclude_cols)]
    X = df[act_cols].values.astype(float)

    true_y = df["true_label"].values if "true_label" in df.columns else None
    pred_y = df["pred_label"].values if "pred_label" in df.columns else None
    correct_mask = None
    if true_y is not None and pred_y is not None:
        try:
            correct_mask = (pred_y == true_y).astype(bool)
        except Exception:
            correct_mask = None
    return X, true_y, pred_y, correct_mask


def _maybe_standardize(X: np.ndarray, standardize: bool) -> Tuple[np.ndarray, Optional[StandardScaler]]:
    if not standardize:
        return X, None
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler


def compute_pca(X: np.ndarray, n_components: int = 2, standardize: bool = True, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Return (coords, explained_variance_ratio)."""
    Xs, _ = _maybe_standardize(X, standardize)
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(Xs)
    return coords, pca.explained_variance_ratio_


def compute_tsne(X: np.ndarray, n_components: int = 2, perplexity: float = 30.0, n_iter: int = 1000,
                 standardize: bool = True, random_state: int = 42) -> np.ndarray:
    Xs, _ = _maybe_standardize(X, standardize)
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter,
                learning_rate='auto', init='pca', random_state=random_state, n_jobs=None)
    return tsne.fit_transform(Xs)


def compute_isomap(X: np.ndarray, n_components: int = 2, n_neighbors: int = 15, standardize: bool = True) -> np.ndarray:
    Xs, _ = _maybe_standardize(X, standardize)
    iso = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    return iso.fit_transform(Xs)


def compute_lle(X: np.ndarray, n_components: int = 2, n_neighbors: int = 15, method: str = 'standard', standardize: bool = True) -> np.ndarray:
    Xs, _ = _maybe_standardize(X, standardize)
    lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method=method)
    return lle.fit_transform(Xs)


def compute_spectral(X: np.ndarray, n_components: int = 2, n_neighbors: int = 15, standardize: bool = True, random_state: int = 42) -> np.ndarray:
    Xs, _ = _maybe_standardize(X, standardize)
    emb = SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state)
    return emb.fit_transform(Xs)


def compute_mds(X: np.ndarray, n_components: int = 2, n_init: int = 4, max_iter: int = 300, standardize: bool = True, random_state: int = 42) -> np.ndarray:
    Xs, _ = _maybe_standardize(X, standardize)
    mds = MDS(n_components=n_components, n_init=n_init, max_iter=max_iter, random_state=random_state, normalized_stress='auto')
    return mds.fit_transform(Xs)


def compute_umap_optional(X: np.ndarray, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1,
                          standardize: bool = True, random_state: int = 42) -> Optional[np.ndarray]:
    """Compute UMAP embedding if umap-learn is installed, else return None."""
    try:
        import umap  # type: ignore
    except Exception:
        return None
    Xs, _ = _maybe_standardize(X, standardize)
    um = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    return um.fit_transform(Xs)


def compute_embeddings(X: np.ndarray, methods: Optional[List[str]] = None, standardize: bool = True,
                       random_state: int = 42, perplexity: float = 30.0, n_neighbors: int = 15,
                       max_points: Optional[int] = 3000) -> Dict[str, Any]:
    """Compute multiple 2D embeddings for activations.

    Returns a dict mapping method name -> dict(coords=..., meta=...)
    The PCA entry also includes explained_variance_ratio.
    If max_points is set and X has more rows, a uniform sample is taken for speed.
    """
    Xn = as_numpy(X).astype(float)
    idx = np.arange(Xn.shape[0])
    if max_points is not None and Xn.shape[0] > max_points:
        rng = np.random.default_rng(random_state)
        sel = np.sort(rng.choice(Xn.shape[0], size=max_points, replace=False))
        Xn = Xn[sel]
        idx = idx[sel]

    methods = methods or ["pca", "tsne", "isomap", "lle", "spectral"]
    out: Dict[str, Any] = {"indices": idx}
    for m in methods:
        try:
            if m == "pca":
                coords, var = compute_pca(Xn, standardize=standardize, random_state=random_state)
                out[m] = {"coords": coords, "explained_variance_ratio": var}
            elif m == "tsne":
                coords = compute_tsne(Xn, standardize=standardize, random_state=random_state, perplexity=perplexity)
                out[m] = {"coords": coords}
            elif m == "isomap":
                coords = compute_isomap(Xn, standardize=standardize, n_neighbors=n_neighbors)
                out[m] = {"coords": coords}
            elif m == "lle":
                coords = compute_lle(Xn, standardize=standardize, n_neighbors=n_neighbors)
                out[m] = {"coords": coords}
            elif m == "spectral":
                coords = compute_spectral(Xn, standardize=standardize, n_neighbors=n_neighbors, random_state=random_state)
                out[m] = {"coords": coords}
            elif m == "mds":
                coords = compute_mds(Xn, standardize=standardize, random_state=random_state)
                out[m] = {"coords": coords}
            elif m == "umap":
                coords = compute_umap_optional(Xn, standardize=standardize, random_state=random_state, n_neighbors=n_neighbors)
                if coords is not None:
                    out[m] = {"coords": coords}
        except Exception as e:
            # keep going, record failure
            out[m] = {"coords": None, "error": str(e)}
    return out


def plot_embedding(coords: np.ndarray, labels: Optional[np.ndarray] = None, correct: Optional[np.ndarray] = None,
                   title: str = "", ax: Optional[Any] = None, cmap: str = "tab10") -> Any:
    """Plot a 2D embedding with color by label and marker by correctness."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    if coords is None or coords.shape[1] != 2:
        ax.text(0.5, 0.5, "embedding unavailable", ha='center', va='center')
        ax.set_axis_off()
        return ax
    if labels is None:
        ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.8)
    else:
        if correct is None:
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=8, alpha=0.8, cmap=cmap)
            plt.colorbar(sc, ax=ax, shrink=0.8)
        else:
            # different marker for incorrect predictions
            correct = correct.astype(bool)
            sc1 = ax.scatter(coords[correct, 0], coords[correct, 1], c=labels[correct], s=9, alpha=0.85, cmap=cmap, marker='o', label='correct')
            sc2 = ax.scatter(coords[~correct, 0], coords[~correct, 1], c=labels[~correct], s=18, alpha=0.9, cmap=cmap, marker='x', label='incorrect')
            ax.legend(loc='best', fontsize=8)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_embedding_grid(embeddings: Dict[str, Any], true_y: Optional[np.ndarray] = None,
                        correct: Optional[np.ndarray] = None, cols: int = 3, figsize: Tuple[int, int] = (15, 10)) -> Any:
    """Plot a grid of embeddings returned by compute_embeddings.

    Only entries with coords!=None are shown.
    """
    # filter available
    items = [(m, v) for m, v in embeddings.items() if m != 'indices' and v.get('coords') is not None]
    if not items:
        return None
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for i, (m, v) in enumerate(items):
        ax = axes[i]
        title = m.upper()
        if m == 'pca' and 'explained_variance_ratio' in v and v['explained_variance_ratio'] is not None:
            var = v['explained_variance_ratio']
            title += f" (var: {var[0]:.2f}, {var[1]:.2f})"
        plot_embedding(v['coords'], labels=true_y, correct=correct, title=title, ax=ax)
    # hide extra axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # quick smoke demonstration when running module directly
    print("placeholder analysis module: no direct demo available. Import from notebook.")
