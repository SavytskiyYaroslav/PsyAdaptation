import numpy as np

from psyadapt.reliability.icc import ICC


def test_icc_perfect_agreement_is_one():
    # 4 raters, 6 targets, 2 items
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
    X = np.stack([base, base, base, base], axis=0)

    est = ICC(model="twoway", icc_type="agreement", unit="single").fit(X)
    v = est.score()
    assert np.isfinite(v)
    assert abs(v - 1.0) < 1e-12


def test_icc_with_missing_does_not_crash():
    # 3 raters, 5 targets, 1 item
    X = np.array(
        [
            [[1], [2], [3], [4], [np.nan]],
            [[1], [2], [3], [np.nan], [5]],
            [[1], [2], [np.nan], [4], [5]],
        ],
        dtype=float,
    )
    est = ICC(model="twoway", icc_type="agreement", unit="single").fit(X)
    assert est.fit_info_ is not None
