import numpy as np

from psyadapt.reliability.krippendorff import KrippendorffAlpha


def test_krippendorff_perfect_agreement_nominal():
    # 4 raters, 5 objects, 2 items
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
    X = np.stack([base, base, base, base], axis=0)  # (4, 5, 2)
    est = KrippendorffAlpha(scale="nominal").fit(X)
    a = est.score()
    assert np.isfinite(a)
    assert abs(a - 1.0) < 1e-12


def test_krippendorff_with_missing_does_not_crash():
    # 3 raters, 4 objects, 1 item
    X = np.array(
        [
            [[1], [1], [0], [np.nan]],
            [[1], [np.nan], [0], [1]],
            [[1], [1], [np.nan], [1]],
        ],
        dtype=float,
    )
    est = KrippendorffAlpha(scale="nominal").fit(X)
    assert est.fit_info_ is not None
    assert est.fit_info_.has_missing is True
    # score might be nan in pathological cases, but should not error
