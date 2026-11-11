from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .base import BaseReliabilityEstimator, FitInfo


@dataclass(frozen=True)
class CohenDetails:
    kappa_per_item: np.ndarray
    agreement_per_item: np.ndarray
    n_pairs_per_item: np.ndarray


def _cohen_kappa_1d(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    """
    Cohen's kappa for 1 item (1D arrays of equal length).
    Returns: (kappa, observed_agreement, n_used)
    Missing values (NaN) are ignored pairwise.
    """
    if a.shape != b.shape:
        raise ValueError("Rater arrays must have the same shape.")

    mask = ~np.isnan(a) & ~np.isnan(b)
    n = int(np.sum(mask))
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return float("nan"), float("nan"), 1

    aa = a[mask]
    bb = b[mask]

    # Observed agreement
    p0 = float(np.mean(aa == bb))

    # Expected agreement: sum_c p(a=c) * p(b=c)
    cats = np.union1d(np.unique(aa), np.unique(bb))
    if cats.size == 0:
        return float("nan"), p0, n

    pa = np.array([np.mean(aa == c) for c in cats], dtype=float)
    pb = np.array([np.mean(bb == c) for c in cats], dtype=float)
    pe = float(np.sum(pa * pb))

    denom = 1.0 - pe
    if denom == 0.0:
        # If pe == 1, kappa undefined (all mass on one category in both raters)
        return float("nan"), p0, n

    kappa = (p0 - pe) / denom
    return float(kappa), p0, n


class CohenKappa(BaseReliabilityEstimator):
    """
    Cohen's Kappa for 2 raters (nominal categories).

    Input:
    - X: np.ndarray shape (2, n_objects, n_items)
      Raters x Objects x Items

    Aggregation:
    - 'mean' (default): mean of kappas across items (ignoring NaNs)
    - 'median': median across items (ignoring NaNs)
    """

    def __init__(self, *, aggregate: Literal["mean", "median"] = "mean") -> None:
        super().__init__()
        if aggregate not in {"mean", "median"}:
            raise ValueError("aggregate must be 'mean' or 'median'.")
        self.aggregate = aggregate

    def fit(self, X: np.ndarray) -> "CohenKappa":
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError("X must be 3D: (2, n_objects, n_items).")
        if X.shape[0] != 2:
            raise ValueError("CohenKappa requires exactly 2 raters (X.shape[0] == 2).")

        _, n_objects, n_items = X.shape
        has_missing = bool(np.isnan(X).any())

        kappas = np.full((n_items,), np.nan, dtype=float)
        p0s = np.full((n_items,), np.nan, dtype=float)
        ns = np.zeros((n_items,), dtype=int)

        r1 = X[0, :, :]
        r2 = X[1, :, :]

        for j in range(n_items):
            k, p0, n = _cohen_kappa_1d(r1[:, j], r2[:, j])
            kappas[j] = k
            p0s[j] = p0
            ns[j] = n

        # Aggregate
        valid = np.isfinite(kappas)
        if not np.any(valid):
            self.warnings_.append(
                "Kappa is undefined for all items (possibly constant ratings or too much missingness)."
            )
            score = float("nan")
        else:
            if self.aggregate == "mean":
                score = float(np.nanmean(kappas))
            else:
                score = float(np.nanmedian(kappas))

        # Warnings
        if has_missing:
            self.warnings_.append(
                "Missing values detected. Pairwise deletion was applied per item."
            )

        # If many items have very low n, warn
        low_n = int(np.sum(ns < 5))
        if low_n > 0:
            self.warnings_.append(
                f"{low_n} item(s) have fewer than 5 paired observations; kappa may be unstable."
            )

        self.fit_info_ = FitInfo(
            n_objects=n_objects,
            n_items=n_items,
            n_raters=2,
            has_missing=has_missing,
        )
        self.score_ = score
        self.details_ = {
            "aggregate": self.aggregate,
            "kappa_per_item": kappas.tolist(),
            "agreement_per_item": p0s.tolist(),
            "n_pairs_per_item": ns.tolist(),
        }
        return self

    def fit_csv(self, path_rater1: str, path_rater2: str) -> "CohenKappa":
        from psyadapt.io.csv import load_csv_raters

        X = load_csv_raters([path_rater1, path_rater2])
        return self.fit(X)

    def interpret(self) -> str:
        k = self.score()
        if not np.isfinite(k):
            return "Kappa is undefined (check missingness and whether ratings are constant)."
        # Common heuristic scale (Landis & Koch style)
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
