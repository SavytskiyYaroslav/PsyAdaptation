from __future__ import annotations

from typing import Iterable, Literal, Optional

import numpy as np

from psyadapt.io.csv import load_csv_matrix, load_csv_raters

from .base import BaseReliabilityEstimator, FitInfo
from .cohen import CohenKappa
from .cronbach import CronbachAlpha
from .fleiss import FleissKappa
from .icc import ICC
from .krippendorff import KrippendorffAlpha

RatingScale = Literal["nominal", "ordinal", "interval"]
Agg = Literal["mean", "median"]
Force = Literal["cronbach", "cohen", "fleiss", "krippendorff", "icc"]


class AutoReliability(BaseReliabilityEstimator):
    """
    Auto selection aligned with your stats.py protocol (deterministic, no scale inference):

    - 1 file  -> CronbachAlpha
    - 2 files -> CohenKappa
    - 3-10    -> FleissKappa
    - >10     -> KrippendorffAlpha

    IMPORTANT:
    - ICC is NEVER selected automatically.
      Use force="icc" to compute ICC.
    """

    def __init__(
        self,
        *,
        rating_scale: RatingScale = "nominal",
        aggregate: Agg = "mean",
        force: Optional[Force] = None,
        # ICC parameters (used only when force="icc")
        icc_model: Literal["oneway", "twoway"] = "twoway",
        icc_type: Literal["agreement", "consistency"] = "agreement",
        icc_unit: Literal["single", "average"] = "single",
    ) -> None:
        super().__init__()
        self.rating_scale = rating_scale
        self.aggregate = aggregate
        self.force = force

        self.icc_model = icc_model
        self.icc_type = icc_type
        self.icc_unit = icc_unit

        self.estimator_: Optional[BaseReliabilityEstimator] = None

    # ---------------------------
    # Public API
    # ---------------------------

    def fit_csv(self, paths: Iterable[str]) -> "AutoReliability":
        paths = list(paths)
        if len(paths) == 0:
            raise ValueError("No CSV paths provided.")

        if self.force is not None:
            return self._fit_forced_csv(paths)

        n_files = len(paths)

        # 1 file: Cronbach
        if n_files == 1:
            est = CronbachAlpha().fit(load_csv_matrix(paths[0]))
            return self._adopt(est, selected_rule="n_files==1 -> CronbachAlpha")

        # 2+ files: inter-rater tensor
        X = load_csv_raters(paths)

        # 2 files: Cohen
        if n_files == 2:
            est = CohenKappa(aggregate=self.aggregate).fit(X)
            return self._adopt(est, selected_rule="n_files==2 -> CohenKappa")

        # 3-10 files: Fleiss
        if 3 <= n_files <= 10:
            est = FleissKappa(aggregate=self.aggregate).fit(X)
            return self._adopt(est, selected_rule="3<=n_files<=10 -> FleissKappa")

        # >10 files: Krippendorff (scale set by rating_scale)
        est = KrippendorffAlpha(scale=self.rating_scale, aggregate=self.aggregate).fit(X)
        return self._adopt(est, selected_rule="n_files>10 -> KrippendorffAlpha")

    def fit(self, X: np.ndarray) -> "AutoReliability":
        """
        Fit from arrays directly.

        - 2D -> CronbachAlpha
        - 3D -> Cohen (2), Fleiss (3-10), Kripp (>10)
        ICC only via force="icc".
        """
        X = np.asarray(X, dtype=float)

        if self.force is not None:
            return self._fit_forced_array(X)

        if X.ndim == 2:
            est = CronbachAlpha().fit(X)
            return self._adopt(est, selected_rule="X.ndim==2 -> CronbachAlpha")

        if X.ndim != 3:
            raise ValueError("X must be 2D (Cronbach) or 3D (inter-rater).")

        n_raters = X.shape[0]

        if n_raters == 2:
            est = CohenKappa(aggregate=self.aggregate).fit(X)
            return self._adopt(est, selected_rule="n_raters==2 -> CohenKappa")

        if 3 <= n_raters <= 10:
            est = FleissKappa(aggregate=self.aggregate).fit(X)
            return self._adopt(est, selected_rule="3<=n_raters<=10 -> FleissKappa")

        est = KrippendorffAlpha(scale=self.rating_scale, aggregate=self.aggregate).fit(X)
        return self._adopt(est, selected_rule="n_raters>10 -> KrippendorffAlpha")

    # ---------------------------
    # Forced modes
    # ---------------------------

    def _fit_forced_csv(self, paths: list[str]) -> "AutoReliability":
        key = self.force
        n_files = len(paths)

        if key == "cronbach":
            if n_files != 1:
                raise ValueError("force='cronbach' requires exactly 1 CSV.")
            est = CronbachAlpha().fit(load_csv_matrix(paths[0]))
            return self._adopt(est, selected_rule="forced -> CronbachAlpha", forced=True)

        # inter-rater tensor
        if n_files < 2:
            raise ValueError(f"force='{key}' requires at least 2 CSV files.")

        X = load_csv_raters(paths)

        if key == "cohen":
            if n_files != 2:
                raise ValueError("force='cohen' requires exactly 2 CSV files.")
            est = CohenKappa(aggregate=self.aggregate).fit(X)
            return self._adopt(est, selected_rule="forced -> CohenKappa", forced=True)

        if key == "fleiss":
            if n_files < 3:
                raise ValueError("force='fleiss' requires at least 3 CSV files.")
            est = FleissKappa(aggregate=self.aggregate).fit(X)
            return self._adopt(est, selected_rule="forced -> FleissKappa", forced=True)

        if key == "krippendorff":
            est = KrippendorffAlpha(scale=self.rating_scale, aggregate=self.aggregate).fit(X)
            return self._adopt(est, selected_rule="forced -> KrippendorffAlpha", forced=True)

        if key == "icc":
            est = ICC(
                model=self.icc_model,
                icc_type=self.icc_type,
                unit=self.icc_unit,
                aggregate=self.aggregate,
            ).fit(X)
            return self._adopt(est, selected_rule="forced -> ICC", forced=True)

        raise ValueError(f"Unknown force='{key}'.")

    def _fit_forced_array(self, X: np.ndarray) -> "AutoReliability":
        key = self.force

        if key == "cronbach":
            if X.ndim != 2:
                raise ValueError("force='cronbach' requires 2D array.")
            est = CronbachAlpha().fit(X)
            return self._adopt(est, selected_rule="forced -> CronbachAlpha", forced=True)

        if X.ndim != 3:
            raise ValueError(f"force='{key}' requires 3D array (n_raters, n_objects, n_items).")

        if key == "cohen":
            if X.shape[0] != 2:
                raise ValueError("force='cohen' requires exactly 2 raters.")
            est = CohenKappa(aggregate=self.aggregate).fit(X)
            return self._adopt(est, selected_rule="forced -> CohenKappa", forced=True)

        if key == "fleiss":
            if X.shape[0] < 3:
                raise ValueError("force='fleiss' requires at least 3 raters.")
            est = FleissKappa(aggregate=self.aggregate).fit(X)
            return self._adopt(est, selected_rule="forced -> FleissKappa", forced=True)

        if key == "krippendorff":
            est = KrippendorffAlpha(scale=self.rating_scale, aggregate=self.aggregate).fit(X)
            return self._adopt(est, selected_rule="forced -> KrippendorffAlpha", forced=True)

        if key == "icc":
            est = ICC(
                model=self.icc_model,
                icc_type=self.icc_type,
                unit=self.icc_unit,
                aggregate=self.aggregate,
            ).fit(X)
            return self._adopt(est, selected_rule="forced -> ICC", forced=True)

        raise ValueError(f"Unknown force='{key}'.")

    # ---------------------------
    # Adopt child estimator outputs
    # ---------------------------

    def _adopt(self, est: BaseReliabilityEstimator, *, selected_rule: str, forced: bool = False) -> "AutoReliability":
        self.estimator_ = est
        self.score_ = est.score()
        self.warnings_ = list(est.warnings_)
        self.fit_info_ = est.fit_info_ if est.fit_info_ is not None else FitInfo(0, 0, 0, False)

        self.details_ = {
            "selected_estimator": est.__class__.__name__,
            "selected_rule": selected_rule,
            "forced": forced,
            "rating_scale": self.rating_scale,
            "aggregate": self.aggregate,
            "selected_report": est.report(),
        }
        return self
