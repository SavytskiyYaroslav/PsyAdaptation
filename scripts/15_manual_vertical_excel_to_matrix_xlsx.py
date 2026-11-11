from __future__ import annotations

from pathlib import Path

import pandas as pd

from psyadapt.io.excel import workbook_vertical_to_matrix


def main() -> None:
    # Вкажи свій XLSX
    XLSX = Path("data/raw/vertical_workbook.xlsx")
    OUT_DIR = Path("data/converted")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Параметри під твій формат
    start_row = 10
    question_col = "A"
    score_col = "B"

    header_sheet = None   # або, напр. "RAW"
    sheets = None         # або список: ["S1", "S2", ...]

    df, questions = workbook_vertical_to_matrix(
        str(XLSX),
        start_row=start_row,
        question_col=question_col,
        score_col=score_col,
        header_sheet=header_sheet,
        sheets=sheets,
        on_duplicate_question="first",
        strip_questions=True,
    )

    print("Loaded:", XLSX)
    print("Matrix shape:", df.shape)
    print("First 5 columns:", list(df.columns[:5]))

    # (Опційно) короткий звіт про пропуски
    missing_by_col = df.isna().sum().sort_values(ascending=False)
    missing_by_row = df.isna().sum(axis=1).sort_values(ascending=False)

    out_xlsx = OUT_DIR / f"{XLSX.stem}_MATRIX.xlsx"

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        # Головна матриця
        df.to_excel(writer, sheet_name="MATRIX", index=True)  # index = sheet_name (респондент)

        # Опційний лист з пропусками (щоб швидко зрозуміти, де є пустоти)
        miss_df = pd.DataFrame({
            "missing_in_column": missing_by_col,
        })
        miss_df.to_excel(writer, sheet_name="MISSING_BY_COLUMN", index=True)

        miss_row_df = pd.DataFrame({
            "missing_in_row": missing_by_row,
        })
        miss_row_df.to_excel(writer, sheet_name="MISSING_BY_ROW", index=True)

    print("Saved:", out_xlsx.as_posix())


if __name__ == "__main__":
    main()
