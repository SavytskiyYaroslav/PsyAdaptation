import numpy as np

from psyadapt.reliability.cohen import CohenKappa


def test_cohen_kappa_perfect_agreement():
    # 2 raters, 5 respondents, 3 items
    r1 = np.array(
        [
            [1, 0, 2],
            [1, 1, 2],
            [0, 0, 2],
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    r2 = r1.copy()
    X = np.stack([r1, r2], axis=0)

    est = CohenKappa().fit(X)
    k = est.score()
    # Perfect agreement -> kappa should be 1 for each item, aggregate 1
    assert np.isfinite(k)
    assert abs(k - 1.0) < 1e-12


def test_cohen_kappa_with_missing_does_not_crash():
    r1 = np.array([[1, np.nan], [1, 0], [0, 1], [1, 0]], dtype=float)
    r2 = np.array([[1, 1], [1, 0], [0, np.nan], [1, 0]], dtype=float)
    X = np.stack([r1, r2], axis=0)

    est = CohenKappa().fit(X)
    assert est.fit_info_ is not None
    assert est.fit_info_.n_raters == 2
