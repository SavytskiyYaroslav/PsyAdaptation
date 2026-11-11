import numpy as np

from psyadapt.reliability.fleiss import FleissKappa
from psyadapt.reliability import AutoReliability


def test_fleiss_perfect_agreement():
    # 3 raters, 5 respondents, 2 items
    r1 = np.array([[1, 0], [1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
    r2 = r1.copy()
    r3 = r1.copy()
    X = np.stack([r1, r2, r3], axis=0)

    est = FleissKappa().fit(X)
    k = est.score()
    assert np.isfinite(k)
    assert abs(k - 1.0) < 1e-12


def test_auto_fit_csv_rule_selects_fleiss(tmp_path):
    # Create three identical CSVs in temp folder
    data = "a,b\n1,0\n1,0\n0,1\n1,0\n"
    p1 = tmp_path / "e1.csv"
    p2 = tmp_path / "e2.csv"
    p3 = tmp_path / "e3.csv"
    p1.write_text(data, encoding="utf-8")
    p2.write_text(data, encoding="utf-8")
    p3.write_text(data, encoding="utf-8")

    est = AutoReliability().fit_csv([str(p1), str(p2), str(p3)])
    assert est.details_["selected_estimator"] == "FleissKappa"
    assert abs(est.score() - 1.0) < 1e-12
