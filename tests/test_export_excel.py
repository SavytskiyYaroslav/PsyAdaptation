from pathlib import Path

from psyadapt.export_excel import compare_reliability_to_excel


def test_compare_reliability_to_excel_creates_file(tmp_path):
    data = "a,b\n1,0\n1,0\n0,1\n1,0\n"
    p1 = tmp_path / "e1.csv"
    p2 = tmp_path / "e2.csv"
    p1.write_text(data, encoding="utf-8")
    p2.write_text(data, encoding="utf-8")

    out_xlsx = tmp_path / "report.xlsx"
    rep = compare_reliability_to_excel([str(p1), str(p2)], out_path=str(out_xlsx))

    assert out_xlsx.exists()
    assert rep["inputs"]["n_files"] == 2
