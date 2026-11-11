import numpy as np

from psyadapt.reliability import AutoReliability


def test_auto_interval_selects_icc():
    # 3 raters, 6 targets, 2 items (numeric interval-like)
    base = np.array(
        [
            [10, 1],
            [11, 2],
            [12, 3],
            [13, 4],
            [14, 5],
            [15, 6],
        ],
        dtype=float,
    )
    X = np.stack([base, base, base], axis=0)

    est = AutoReliability(force="icc", rating_scale="interval").fit(X)
    assert est.details_["selected_estimator"] == "ICC"
    assert abs(est.score() - 1.0) < 1e-12


def test_auto_nominal_selects_fleiss_for_3_raters():
    # 3 raters, 5 targets, 2 items (nominal categories)
    base = np.array(
        [
            [1, 0],
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
        ],
        dtype=float,
    )
    X = np.stack([base, base, base], axis=0)

    est = AutoReliability(rating_scale="nominal").fit(X)
    assert est.details_["selected_estimator"] == "FleissKappa"
    assert abs(est.score() - 1.0) < 1e-12
