from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .base import BaseReliabilityEstimator, FitInfo

Model = Literal["oneway", "twoway"]
Type = Literal["agreement", "consistency"]
Unit = Literal["single", "average"]
Agg = Literal["mean", "median"]


@dataclass(frozen=True)
class ICCItemStats:
    icc: float
    n_targets: int
    n_raters: int
    msr: float  # mean square rows/targets
    msc: float  # mean square columns/raters (twoway only)
    mse: float  # residual mean square (twoway) or within (oneway)


def _anova_twoway(Y: np.ndarray) -> tuple[float, float, float]:
    """
    Two-way ANOVA mean squares for matrix Y of shape (n_targets, n_raters).
    Returns (MSR, MSC, MSE).
    """
    n, k = Y.shape
    grand = float(np.mean(Y))

    mean_rows = np.mean(Y, axis=1)
    mean_cols = np.mean(Y, axis=0)

    ssr = k * float(np.sum((mean_rows - grand) ** 2))          # rows/targets
    ssc = n * float(np.sum((mean_cols - grand) ** 2))          # cols/raters
    sse = float(np.sum((Y - mean_rows[:, None] - mean_cols[None, :] + grand) ** 2))

    msr = ssr / (n - 1) if n > 1 else float("nan")
    msc = ssc / (k - 1) if k > 1 else float("nan")
    mse = sse / ((n - 1) * (k - 1)) if (n > 1 and k > 1) else float("nan")

    return float(msr), float(msc), float(mse)


def _anova_oneway(Y: np.ndarray) -> tuple[float, float]:
    """
    One-way ANOVA mean squares for matrix Y of shape (n_targets, n_raters).
    Returns (MSR, MSW) where MSW is within-target mean square.
    """
    n, k = Y.shape
    grand = float(np.mean(Y))
    mean_rows = np.mean(Y, axis=1)

    ssr = k * float(np.sum((mean_rows - grand) ** 2))
    ssw = float(np.sum((Y - mean_rows[:, None]) ** 2))

    msr = ssr / (n - 1) if n > 1 else float("nan")
    msw = ssw / (n * (k - 1)) if (n > 0 and k > 1) else float("nan")

    return float(msr), float(msw)


def _icc_from_ms(
    *,
    model: Model,
    icc_type: Type,
    unit: Unit,
    msr: float,
    msc: float,
    mse: float,
    n: int,
    k: int,
) -> float:
    """
    Compute ICC from mean squares.
    """
    if not (np.isfinite(msr) and np.isfinite(mse)) or n < 2 or k < 2:
        return float("nan")

    if model == "oneway":
        # ICC(1,1) / ICC(1,k)
        msw = mse
        denom = msr + (k - 1) * msw
        if denom == 0:
            return float("nan")
        icc_1_1 = (msr - msw) / denom
        if unit == "single":
            return float(icc_1_1)
        # average measure
        denom_k = msr
        if denom_k == 0:
            return float("nan")
        icc_1_k = (msr - msw) / msr
        return float(icc_1_k)

    # twoway
    if not np.isfinite(msc):
        return float("nan")

    if icc_type == "agreement":
        # ICC(2,1) / ICC(2,k): two-way random, absolute agreement
        if unit == "single":
            denom = msr + (k - 1) * mse + (k * (msc - mse) / n)
            if denom == 0:
                return float("nan")
            return float((msr - mse) / denom)
        # average
        denom = msr + ((msc - mse) / n)
        if denom == 0:
            return float("nan")
        return float((msr - mse) / denom)

    # consistency
    # ICC(3,1) / ICC(3,k): two-way mixed/fixed, consistency
    if unit == "single":
        denom = msr + (k - 1) * mse
        if denom == 0:
            return float("nan")
        return float((msr - mse) / denom)

    # average
    if msr == 0:
        return float("nan")
    return float((msr - mse) / msr)


class ICC(BaseReliabilityEstimator):
    """
    Intraclass Correlation Coefficient (ICC) for numeric ratings.

    Input:
    - X: np.ndarray shape (n_raters, n_objects, n_items)

    Parameters:
    - model: 'oneway' or 'twoway'
    - icc_type:
        - 'agreement'   -> ICC(2,1)/(2,k) when model='twoway'
        - 'consistency' -> ICC(3,1)/(3,k) when model='twoway'
        For model='oneway' it computes ICC(1,1)/(1,k) regardless of icc_type.
    - unit: 'single' or 'average' (k)
    - aggregate: mean/median over items (ignoring NaNs)

    Missing:
    - supported via listwise deletion per item (drop objects with any NaN across raters for that item)
    """

    def __init__(
        self,
        *,
        model: Model = "twoway",
        icc_type: Type = "agreement",
        unit: Unit = "single",
        aggregate: Agg = "mean",
    ) -> None:
        super().__init__()
        if model not in {"oneway", "twoway"}:
            raise ValueError("model must be 'oneway' or 'twoway'.")
        if icc_type not in {"agreement", "consistency"}:
            raise ValueError("icc_type must be 'agreement' or 'consistency'.")
        if unit not in {"single", "average"}:
            raise ValueError("unit must be 'single' or 'average'.")
        if aggregate not in {"mean", "median"}:
            raise ValueError("aggregate must be 'mean' or 'median'.")

        self.model = model
        self.icc_type = icc_type
        self.unit = unit
        self.aggregate = aggregate

    def fit(self, X: np.ndarray) -> "ICC":
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError("X must be 3D: (n_raters, n_objects, n_items).")

        n_raters, n_objects, n_items = X.shape
        has_missing = bool(np.isnan(X).any())

        icc_items = np.full((n_items,), np.nan, dtype=float)
        n_used = np.zeros((n_items,), dtype=int)
        msr_items = np.full((n_items,), np.nan, dtype=float)
        msc_items = np.full((n_items,), np.nan, dtype=float)
        mse_items = np.full((n_items,), np.nan, dtype=float)

        # Work per item: build targets x raters matrix Y
        for j in range(n_items):
            # Shape: (n_objects, n_raters)
            Y = X[:, :, j].T

            # listwise deletion per object: keep rows with all raters present
            mask = ~np.isnan(Y).any(axis=1)
            Yv = Y[mask]
            n = Yv.shape[0]
            k = Yv.shape[1]

            n_used[j] = n
            if n < 2 or k < 2:
                continue

            if self.model == "oneway":
                msr, msw = _anova_oneway(Yv)
                msr_items[j] = msr
                mse_items[j] = msw
                msc_items[j] = float("nan")
                icc_items[j] = _icc_from_ms(
                    model="oneway",
                    icc_type=self.icc_type,
                    unit=self.unit,
                    msr=msr,
                    msc=float("nan"),
                    mse=msw,
                    n=n,
                    k=k,
                )
            else:
                msr, msc, mse = _anova_twoway(Yv)
                msr_items[j] = msr
                msc_items[j] = msc
                mse_items[j] = mse
                icc_items[j] = _icc_from_ms(
                    model="twoway",
                    icc_type=self.icc_type,
                    unit=self.unit,
                    msr=msr,
                    msc=msc,
                    mse=mse,
                    n=n,
                    k=k,
                )

        valid = np.isfinite(icc_items)
        if not np.any(valid):
            self.warnings_.append(
                "ICC is undefined for all items (check missingness, constant scores, or too few targets)."
            )
            score = float("nan")
        else:
            score = float(np.nanmean(icc_items) if self.aggregate == "mean" else np.nanmedian(icc_items))

        if has_missing:
            self.warnings_.append(
                "Missing values detected. Per-item listwise deletion across raters was applied."
            )

        low_n = int(np.sum(n_used < 5))
        if low_n > 0:
            self.warnings_.append(
                f"{low_n} item(s) have fewer than 5 usable targets after NaN filtering; ICC may be unstable."
            )

        self.fit_info_ = FitInfo(
            n_objects=n_objects,
            n_items=n_items,
            n_raters=n_raters,
            has_missing=has_missing,
        )
        self.score_ = score
        self.details_ = {
            "model": self.model,
            "icc_type": self.icc_type,
            "unit": self.unit,
            "aggregate": self.aggregate,
            "icc_per_item": icc_items.tolist(),
            "n_targets_used_per_item": n_used.tolist(),
            "msr_per_item": msr_items.tolist(),
            "msc_per_item": msc_items.tolist(),
            "mse_per_item": mse_items.tolist(),
            "n_raters": n_raters,
            "n_objects": n_objects,
            "n_items": n_items,
        }
        return self

    def fit_csv(self, paths: list[str]) -> "ICC":
        from psyadapt.io.csv import load_csv_raters

        X = load_csv_raters(paths)
        return self.fit(X)

    def interpret(self) -> str:
        v = self.score()
        if not np.isfinite(v):
            return "ICC is undefined (check missingness/constant ratings/too few targets)."
        # Very common heuristic bands (not universal!)
        if v < 0.50:
            return "Poor reliability."
        if v < 0.75:
            return "Moderate reliability."
        if v < 0.90:
            return "Good reliability."
        return "Excellent reliability."
