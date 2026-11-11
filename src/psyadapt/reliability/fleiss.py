from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .base import BaseReliabilityEstimator, FitInfo


@dataclass(frozen=True)
class FleissDetails:
    kappa_per_item: np.ndarray
    agreement_per_item: np.ndarray  # Pbar per item (observed agreement)
    expected_agreement_per_item: np.ndarray  # Pe per item
    n_objects: int
    n_items: int
    n_raters: int


def _fleiss_kappa_counts(M: np.ndarray) -> tuple[float, float, float]:
    """
    Fleiss' kappa for a single item using count matrix M.

    M shape: (N, k)
      N = number of objects (respondents)
      k = number of categories
      Each row sums to n (number of raters).

    Returns: (kappa, Pbar, Pe)
    """
    if M.ndim != 2:
        raise ValueError("Count matrix M must be 2D (N, k).")

    N, k = M.shape
    if N == 0 or k == 0:
        return float("nan"), float("nan"), float("nan")

    n = int(np.sum(M[0]))
    if n < 2:
        # Fleiss requires at least 2 raters
        return float("nan"), float("nan"), float("nan")

    # Validate row sums
    row_sums = np.sum(M, axis=1)
    if not np.all(row_sums == n):
        raise ValueError("Each object must be rated by exactly the same number of raters.")

    # P_i = (sum_j n_ij^2 - n) / (n*(n-1))
    Pi = (np.sum(M * M, axis=1) - n) / (n * (n - 1))
    Pbar = float(np.mean(Pi))

    # p_j = sum_i n_ij / (N*n)
    pj = np.sum(M, axis=0) / (N * n)
    Pe = float(np.sum(pj * pj))

    denom = 1.0 - Pe
    if denom == 0.0:
        return float("nan"), Pbar, Pe

    kappa = (Pbar - Pe) / denom
    return float(kappa), Pbar, Pe


def _to_counts_per_item(X_item: np.ndarray) -> np.ndarray:
    """
    Convert ratings for one item to Fleiss count matrix.

    X_item shape: (n_raters, n_objects)
    Returns M shape: (n_objects, n_categories)
    """
    if X_item.ndim != 2:
        raise ValueError("X_item must be 2D (n_raters, n_objects).")

    n_raters, n_objects = X_item.shape

    if np.isnan(X_item).any():
        raise ValueError("FleissKappa (ideal mode) does not support missing values (NaN).")

    # Categories across all raters/objects
    cats = np.unique(X_item)
    # Ensure deterministic order
    cats = np.sort(cats)

    k = int(cats.size)
    M = np.zeros((n_objects, k), dtype=int)

    # For each object, count how many raters selected each category
    # We transpose to iterate objects (n_objects, n_raters)
    Xo = X_item.T
    for i in range(n_objects):
        # counts via equality per category
        row = Xo[i]
        M[i] = np.array([(row == c).sum() for c in cats], dtype=int)

    # Validate each row sums to n_raters
    if not np.all(np.sum(M, axis=1) == n_raters):
        raise RuntimeError("Internal error: count rows do not sum to n_raters.")

    return M


class FleissKappa(BaseReliabilityEstimator):
    """
    Fleiss' Kappa for >2 raters (nominal categories).

    Input:
    - X: np.ndarray shape (n_raters, n_objects, n_items)

    Assumptions (ideal mode):
    - no missing (NaN)
    - each object rated by all raters
    - ratings treated as nominal categories

    Aggregation across items:
    - 'mean' (default) or 'median'
    """

    def __init__(self, *, aggregate: Literal["mean", "median"] = "mean") -> None:
        super().__init__()
        if aggregate not in {"mean", "median"}:
            raise ValueError("aggregate must be 'mean' or 'median'.")
        self.aggregate = aggregate

    def fit(self, X: np.ndarray) -> "FleissKappa":
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError("X must be 3D: (n_raters, n_objects, n_items).")

        n_raters, n_objects, n_items = X.shape
        if n_raters < 3:
            raise ValueError("FleissKappa requires at least 3 raters.")
        if np.isnan(X).any():
            raise ValueError(
                "FleissKappa (ideal mode) does not support missing values. "
                "Use KrippendorffAlpha later for missingness."
            )

        kappas = np.full((n_items,), np.nan, dtype=float)
        pbar = np.full((n_items,), np.nan, dtype=float)
        pe = np.full((n_items,), np.nan, dtype=float)

        # Per item: convert to counts, compute kappa
        for j in range(n_items):
            X_item = X[:, :, j]  # (n_raters, n_objects)
            M = _to_counts_per_item(X_item)
            kappa_j, pbar_j, pe_j = _fleiss_kappa_counts(M)
            kappas[j] = kappa_j
            pbar[j] = pbar_j
            pe[j] = pe_j

        valid = np.isfinite(kappas)
        if not np.any(valid):
            self.warnings_.append(
                "Kappa is undefined for all items (possibly constant ratings)."
            )
            score = float("nan")
        else:
            score = float(np.nanmean(kappas) if self.aggregate == "mean" else np.nanmedian(kappas))

        self.fit_info_ = FitInfo(
            n_objects=n_objects,
            n_items=n_items,
            n_raters=n_raters,
            has_missing=False,
        )
        self.score_ = score
        self.details_ = {
            "aggregate": self.aggregate,
            "kappa_per_item": kappas.tolist(),
            "agreement_per_item": pbar.tolist(),
            "expected_agreement_per_item": pe.tolist(),
            "n_raters": n_raters,
            "n_objects": n_objects,
            "n_items": n_items,
        }
        return self

    def fit_csv(self, paths: list[str]) -> "FleissKappa":
        from psyadapt.io.csv import load_csv_raters

        X = load_csv_raters(paths)
        return self.fit(X)

    def interpret(self) -> str:
        k = self.score()
        if not np.isfinite(k):
            return "Fleiss' kappa is undefined (check whether ratings are constant)."
        if k < 0:
            return "Poor agreement (worse than chance)."
        if k < 0.20:
            return "Slight agreement."
        if k < 0.40:
            return "Fair agreement."
        if k < 0.60:
            return "Moderate agreement."
        if k < 0.80:
            return "Substantial agreement."
        return "Almost perfect agreement."
