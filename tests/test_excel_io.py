from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from psyadapt.io.excel import (
    excel_to_csv,
    excel_to_matrix,
    list_excel_sheets,
    read_excel_table,
    workbook_to_csvs,
)


def _make_xlsx(path: Path, sheet_name: str, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name=sheet_name, index=False)


def test_list_excel_sheets(tmp_path: Path):
    xlsx = tmp_path / "book.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        pd.DataFrame({"a": [1]}).to_excel(w, sheet_name="RAW", index=False)
        pd.DataFrame({"b": [2]}).to_excel(w, sheet_name="SCORES", index=False)

    sheets = list_excel_sheets(str(xlsx))
    assert "RAW" in sheets
    assert "SCORES" in sheets


def test_read_excel_table_normalizes_and_numeric(tmp_path: Path):
    # Includes whitespace and an "Unnamed" column
    df = pd.DataFrame(
        {
            "  Q1  ": [1, 2, 3],
            "Q2": ["4", "5", "6"],  # strings but numeric-like
            "Unnamed: 2": [None, None, None],  # should be dropped as empty col
        }
    )
    xlsx = tmp_path / "t.xlsx"
    _make_xlsx(xlsx, "RAW", df)

    out = read_excel_table(str(xlsx), sheet_name="RAW", header_row=0)

    # dropped fully empty column
    assert out.shape == (3, 2)

    # columns trimmed
    assert list(out.columns) == ["Q1", "Q2"]

    # conversion isn't done in read_excel_table (by design), but strings trimmed
    assert int(out.iloc[0, 1]) == 4


def test_excel_to_csv_writes_csv_and_numeric(tmp_path: Path):
    df = pd.DataFrame(
        {
            "Q1": [1, 2, 3],
            "Q2": [4, 5, 6],
        }
    )
    xlsx = tmp_path / "t.xlsx"
    _make_xlsx(xlsx, "RAW", df)

    out_csv = tmp_path / "out.csv"
    res = excel_to_csv(
        str(xlsx),
        sheet_name="RAW",
        out_csv=str(out_csv),
        header_row=0,
        allow_missing=True,
    )

    assert out_csv.exists()
    assert res.n_rows == 3
    assert res.n_cols == 2
    assert res.columns == ["Q1", "Q2"]

    df2 = pd.read_csv(out_csv)
    assert list(df2.columns) == ["Q1", "Q2"]
    assert df2.iloc[2, 1] == 6


def test_excel_to_matrix_returns_X_and_names(tmp_path: Path):
    df = pd.DataFrame(
        {
            "A": [10, 11],
            "B": [1, 2],
        }
    )
    xlsx = tmp_path / "m.xlsx"
    _make_xlsx(xlsx, "RAW", df)

    X, names = excel_to_matrix(str(xlsx), sheet_name="RAW")
    assert names == ["A", "B"]
    assert isinstance(X, np.ndarray)
    assert X.shape == (2, 2)
    assert float(X[0, 0]) == 10.0


def test_workbook_to_csvs_multiple_sheets(tmp_path: Path):
    xlsx = tmp_path / "book.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        pd.DataFrame({"Q1": [1, 2], "Q2": [3, 4]}).to_excel(w, sheet_name="M1", index=False)
        pd.DataFrame({"X": [5, 6], "Y": [7, 8]}).to_excel(w, sheet_name="M2", index=False)

    out_dir = tmp_path / "csvs"
    results = workbook_to_csvs(
        str(xlsx),
        out_dir=str(out_dir),
        sheets=["M1", "M2"],
        filename_prefix="method_",
    )

    assert len(results) == 2
    p1 = out_dir / "method_M1.csv"
    p2 = out_dir / "method_M2.csv"
    assert p1.exists()
    assert p2.exists()

    df1 = pd.read_csv(p1)
    df2 = pd.read_csv(p2)
    assert list(df1.columns) == ["Q1", "Q2"]
    assert list(df2.columns) == ["X", "Y"]
