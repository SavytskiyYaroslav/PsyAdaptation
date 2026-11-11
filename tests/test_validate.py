from psyadapt.validate import validate_single_test_csv, validate_raters_csvs


def test_validate_single_ok(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    rep = validate_single_test_csv(str(p), allow_missing=False)
    assert rep["ok"] is True


def test_validate_single_non_numeric_column(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("a,b\n1,x\n3,y\n", encoding="utf-8")
    rep = validate_single_test_csv(str(p), allow_missing=True)
    assert rep["ok"] is False
    assert any(i["code"] == "csv.non_numeric_column" for i in rep["issues"])


def test_validate_raters_shape_mismatch(tmp_path):
    p1 = tmp_path / "e1.csv"
    p2 = tmp_path / "e2.csv"
    p1.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    p2.write_text("a,b\n1,2\n", encoding="utf-8")  # fewer rows

    rep = validate_raters_csvs([str(p1), str(p2)])
    assert rep["ok"] is False
    assert any(i["code"] == "raters.shape_mismatch" for i in rep["issues"])
