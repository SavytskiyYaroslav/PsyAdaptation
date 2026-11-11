import numpy as np

from psyadapt.reliability.cronbach import CronbachAlpha


def test_cronbach_basic_no_missing():
    # Simple dataset with some consistency across items
    X = np.array(
        [
            [1, 2, 1, 2],
            [2, 3, 2, 3],
            [1, 2, 1, 2],
            [2, 3, 2, 3],
            [1, 2, 1, 2],
        ],
        dtype=float,
    )
    est = CronbachAlpha().fit(X)
    a = est.score()
    assert np.isfinite(a)
    assert -1.0 <= a <= 1.0


def test_cronbach_with_missing():
    X = np.array(
        [
            [1, 2, np.nan, 2],
            [2, 3, 2, 3],
            [1, np.nan, 1, 2],
            [2, 3, 2, np.nan],
            [1, 2, 1, 2],
        ],
        dtype=float,
    )
    est = CronbachAlpha().fit(X)
    a = est.score()
    # Could be finite or nan depending on covariance availability; but should not crash
    assert est.fit_info_ is not None
    assert est.fit_info_.has_missing is True
    assert "Missing values detected" in " ".join(est.warnings_)
