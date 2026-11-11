from psyadapt.compare import compare_reliability_csv


def test_compare_two_files_returns_primary_and_icc(tmp_path):
    data = "a,b\n1,0\n1,0\n0,1\n1,0\n"
    p1 = tmp_path / "e1.csv"
    p2 = tmp_path / "e2.csv"
    p1.write_text(data, encoding="utf-8")
    p2.write_text(data, encoding="utf-8")

    rep = compare_reliability_csv([str(p1), str(p2)], rating_scale="nominal")

    assert "meta" in rep
    assert rep["inputs"]["n_files"] == 2
    assert "summary" in rep
    assert "results" in rep
    assert "primary" in rep["results"]
    assert "icc" in rep["results"]
    assert rep["results"]["primary"]["estimator"] in {"CohenKappa", "FleissKappa", "KrippendorffAlpha"}
    assert rep["results"]["icc"]["estimator"] == "ICC"

    # per-item schema
    assert isinstance(rep["per_item"], list)
    assert len(rep["per_item"]) == 2  # a,b
    assert "primary" in rep["per_item"][0]
    assert "icc" in rep["per_item"][0]
