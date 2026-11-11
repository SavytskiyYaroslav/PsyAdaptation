from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class FitInfo:
    """Lightweight metadata about the last fit() call."""
    n_objects: int
    n_items: int
    n_raters: int = 1
    has_missing: bool = False


class BaseReliabilityEstimator:
    """
    sklearn-style base estimator for reliability metrics.

    Design rules:
    - __init__ stores parameters only (no heavy computation)
    - fit() computes and stores results in *_ attributes
    - fit() returns self
    - score() returns the main metric as float
    """

    def __init__(self) -> None:
        self.fit_info_: Optional[FitInfo] = None
        self.score_: Optional[float] = None
        self.details_: Dict[str, Any] = {}
        self.diagnostics_: Dict[str, Any] = {}
        self.warnings_: list[str] = []

    def fit(self, X: np.ndarray) -> "BaseReliabilityEstimator":
        raise NotImplementedError

    def score(self) -> float:
        if self.score_ is None:
            raise RuntimeError("Estimator is not fitted yet. Call fit() first.")
        return float(self.score_)

    def report(self) -> Dict[str, Any]:
        """
        Structured report (safe for JSON serialization).
        """
        return {
            "estimator": self.__class__.__name__,
            "score": self.score_,
            "fit_info": None if self.fit_info_ is None else self.fit_info_.__dict__,
            "details": self.details_,
            "diagnostics": self.diagnostics_,
            "warnings": self.warnings_,
        }
