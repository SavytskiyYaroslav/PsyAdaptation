from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .base import BaseReliabilityEstimator, FitInfo


Scale = Literal["nominal", "ordinal", "interval"]
Agg = Literal["mean", "median"]


@dataclass(frozen=True)
class KrippendorffDetails:
    alpha_per_item: np.ndarray
    n_pairs_per_item: np.ndarray
    de_per_item: np.ndarray
    do_per_item: np.ndarray


def _pairwise_sum_dist(values: np.ndarray, dist_matrix: np.ndarray, idx_map: dict[float, int]) -> float:
    """
    Sum of distances over all unordered pairs in `values`.
    values: 1D array of observed categories (no NaNs)
    dist_matrix: precomputed distance^2 between categories (k x k)
    idx_map: map category value -> index in dist_matrix
    """
    m = values.size
    if m < 2:
        return 0.0

    # Map to indices
    idx = np.array([idx_map[v] for v in values], dtype=int)

    # Sum over i<j
    s = 0.0
    for i in range(m - 1):
        s += float(np.sum(dist_matrix[idx[i], idx[i + 1 :]]))
    return s


def _build_distance_matrix(categories: np.ndarray, scale: Scale, pooled_values: np.ndarray) -> np.ndarray:
    """
    Build a k x k matrix of squared distances between categories for Krippendorff alpha.

    - nominal: 0 if same else 1
    - interval: (x - y)^2
    - ordinal: Krippendorff ordinal distance based on cumulative frequencies
      between categories in pooled data:
        d(i,j) = (sum_{k=i}^{j-1} p_k)^2  for i<j
      where p_k are category proportions.
    """
    cats = categories.astype(float)
    k = cats.size

    if k == 0:
        return np.zeros((0, 0), dtype=float)

    if scale == "nominal":
        D = np.ones((k, k), dtype=float)
        np.fill_diagonal(D, 0.0)
        return D

    if scale == "interval":
        diff = cats.reshape(-1, 1) - cats.reshape(1, -1)
        return diff * diff

    if scale == "ordinal":
        # Sort categories in ordinal order (numeric order under your ideal assumptions)
        # Frequencies in pooled data
        pooled = pooled_values.astype(float)
        # Count in same order as categories
        freq = np.array([(pooled == c).sum() for c in cats], dtype=float)
        N = float(freq.sum())
        if N <= 0:
            return np.zeros((k, k), dtype=float)

        p = freq / N  # proportions per category
        # cumulative proportions between categories:
        # for i<j: sum p[i:j]
        cum = np.cumsum(p)  # cum[t] = sum_{0..t} p
        D = np.zeros((k, k), dtype=float)
        for i in range(k):
            for j in range(i + 1, k):
                between = (cum[j - 1] - (cum[i - 1] if i > 0 else 0.0))
                d = between * between
                D[i, j] = d
                D[j, i] = d
        return D

    raise ValueError(f"Unknown scale='{scale}'.")


def _krippendorff_alpha_1item(values_by_object: list[np.ndarray], scale: Scale) -> tuple[float, float, float, int]:
    """
    Compute Krippendorff's alpha for one item.

    values_by_object: list where each element is 1D array of ratings for that object (NaNs removed)
    Returns: (alpha, Do, De, n_pairs_total)
    """
    # Pool all ratings
    pooled = np.concatenate([v for v in values_by_object if v.size > 0], axis=0) if values_by_object else np.array([])
    pooled = pooled.astype(float)

    if pooled.size < 2:
        return float("nan"), float("nan"), float("nan"), 0

    categories = np.unique(pooled)
    categories = np.sort(categories)
    idx_map = {float(c): i for i, c in enumerate(categories)}

    dist = _build_distance_matrix(categories, scale, pooled)

    # Observed disagreement Do: average distance across within-object pairs
    do_sum = 0.0
    n_pairs = 0
    for v in values_by_object:
        m = v.size
        if m < 2:
            continue
        do_sum += _pairwise_sum_dist(v, dist, idx_map)
        n_pairs += m * (m - 1) // 2

    if n_pairs == 0:
        return float("nan"), float("nan"), float("nan"), 0

    Do = do_sum / n_pairs

    # Expected disagreement De: average distance across all pooled pairs
    N = pooled.size
    n_pairs_all = N * (N - 1) // 2
    if n_pairs_all == 0:
        return float("nan"), Do, float("nan"), n_pairs

    de_sum = _pairwise_sum_dist(pooled, dist, idx_map)
    De = de_sum / n_pairs_all

    if not np.isfinite(De) or De == 0.0:
        return float("nan"), Do, De, n_pairs

    alpha = 1.0 - (Do / De)
    return float(alpha), float(Do), float(De), int(n_pairs)


class KrippendorffAlpha(BaseReliabilityEstimator):
    """
    Krippendorff's Alpha (α): inter-rater reliability for any number of raters,
    supports missing values.

    Input:
    - X: np.ndarray shape (n_raters, n_objects, n_items)

    Parameters:
    - scale: 'nominal' | 'ordinal' | 'interval'
    - aggregate: 'mean' | 'median' across items (ignoring NaNs)
    """

    def __init__(self, *, scale: Scale = "nominal", aggregate: Agg = "mean") -> None:
        super().__init__()
        if scale not in {"nominal", "ordinal", "interval"}:
            raise ValueError("scale must be 'nominal', 'ordinal', or 'interval'.")
        if aggregate not in {"mean", "median"}:
            raise ValueError("aggregate must be 'mean' or 'median'.")
        self.scale = scale
        self.aggregate = aggregate

    def fit(self, X: np.ndarray) -> "KrippendorffAlpha":
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError("X must be 3D: (n_raters, n_objects, n_items).")

        n_raters, n_objects, n_items = X.shape
        has_missing = bool(np.isnan(X).any())

        alpha_items = np.full((n_items,), np.nan, dtype=float)
        do_items = np.full((n_items,), np.nan, dtype=float)
        de_items = np.full((n_items,), np.nan, dtype=float)
        pairs_items = np.zeros((n_items,), dtype=int)

        for j in range(n_items):
            # Build per-object ratings for this item
            # X[:, i, j] are ratings across raters for object i
            values_by_object: list[np.ndarray] = []
            for i in range(n_objects):
                vals = X[:, i, j]
                vals = vals[~np.isnan(vals)]
                values_by_object.append(vals.astype(float))

            a, Do, De, n_pairs = _krippendorff_alpha_1item(values_by_object, self.scale)
            alpha_items[j] = a
            do_items[j] = Do
            de_items[j] = De
            pairs_items[j] = n_pairs

        valid = np.isfinite(alpha_items)
        if not np.any(valid):
            self.warnings_.append(
                "Alpha is undefined for all items (check missingness/constant ratings)."
            )
            score = float("nan")
        else:
            score = float(np.nanmean(alpha_items) if self.aggregate == "mean" else np.nanmedian(alpha_items))

        if has_missing:
            self.warnings_.append(
                "Missing values detected. Krippendorff's alpha handles missingness via available pairs."
            )

        low_pairs = int(np.sum(pairs_items < 5))
        if low_pairs > 0:
            self.warnings_.append(
                f"{low_pairs} item(s) have fewer than 5 within-object rating pairs; alpha may be unstable."
            )

        self.fit_info_ = FitInfo(
            n_objects=n_objects,
            n_items=n_items,
            n_raters=n_raters,
            has_missing=has_missing,
        )
        self.score_ = score
        self.details_ = {
            "scale": self.scale,
            "aggregate": self.aggregate,
            "alpha_per_item": alpha_items.tolist(),
            "do_per_item": do_items.tolist(),
            "de_per_item": de_items.tolist(),
            "n_pairs_per_item": pairs_items.tolist(),
            "n_raters": n_raters,
            "n_objects": n_objects,
            "n_items": n_items,
        }
        return self

    def fit_csv(self, paths: list[str]) -> "KrippendorffAlpha":
        from psyadapt.io.csv import load_csv_raters

        X = load_csv_raters(paths)
        return self.fit(X)

    def interpret(self) -> str:
        a = self.score()
        if not np.isfinite(a):
            return "Alpha is undefined (check constant ratings, too few pairs, or extreme missingness)."
        if a < 0:
            return "Poor reliability (worse than chance)."
        if a < 0.67:
            return "Low reliability (often insufficient for strong conclusions)."
        if a < 0.80:
            return "Moderate reliability."
        return "High reliability."
