import numpy as np

from psyadapt.reliability import AutoReliability


def test_auto_fit_2d_selects_cronbach():
    X = np.array(
        [
            [1, 2, 1],
            [2, 3, 2],
            [1, 2, 1],
            [2, 3, 2],
        ],
        dtype=float,
    )
    est = AutoReliability().fit(X)
    assert est.details_["selected_estimator"] == "CronbachAlpha"


def test_auto_fit_3d_selects_cohen_when_two_raters():
    r1 = np.array([[1, 0], [1, 0], [0, 1], [1, 0]], dtype=float)
    r2 = r1.copy()
    X = np.stack([r1, r2], axis=0)

    est = AutoReliability().fit(X)
    assert est.details_["selected_estimator"] == "CohenKappa"
    assert abs(est.score() - 1.0) < 1e-12
