
from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook

from psyadapt.io.excel import workbook_vertical_to_matrix


def _make_vertical_workbook(path: Path, *, start_row: int = 10) -> None:
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "R1"
    ws1["A1"] = "noise"

    questions = ["Q1", "Q2", "Q3"]
    scores1 = [1, 2, 3]
    for i, (q, s) in enumerate(zip(questions, scores1), start=start_row):
        ws1[f"A{i}"] = q
        ws1[f"B{i}"] = s

    ws2 = wb.create_sheet("R2")
    scores2 = [1, 5, 3]
    for i, (q, s) in enumerate(zip(questions, scores2), start=start_row):
        ws2[f"A{i}"] = q
        ws2[f"B{i}"] = s
    ws2[f"A{start_row+10}"] = "Q2"
    ws2[f"B{start_row+10}"] = 999  # should be ignored (first wins)

    ws3 = wb.create_sheet("R3")
    ws3[f"A{start_row}"] = "Q1"
    ws3[f"B{start_row}"] = 7
    ws3[f"A{start_row+1}"] = "Q2"
    ws3[f"B{start_row+1}"] = 8

    wb.save(path)


def test_workbook_vertical_to_matrix_first_dupe_and_missing(tmp_path: Path):
    xlsx = tmp_path / "v.xlsx"
    _make_vertical_workbook(xlsx, start_row=10)

    df, questions = workbook_vertical_to_matrix(
        str(xlsx),
        start_row=10,
        question_col="A",
        score_col="B",
        on_duplicate_question="first",
    )

    assert questions == ["Q1", "Q2", "Q3"]
    assert list(df.columns) == ["Q1", "Q2", "Q3"]
    assert list(df.index) == ["R1", "R2", "R3"]

    assert df.loc["R1", "Q1"] == 1
    assert df.loc["R1", "Q2"] == 2
    assert df.loc["R1", "Q3"] == 3

    assert df.loc["R2", "Q2"] == 5

    assert pd.isna(df.loc["R3", "Q3"])
