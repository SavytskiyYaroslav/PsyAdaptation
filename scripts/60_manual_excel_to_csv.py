from __future__ import annotations

from pathlib import Path

from psyadapt.io.excel import list_excel_sheets, excel_to_csv, workbook_to_csvs
from psyadapt import compare_reliability_csv, compare_reliability_to_excel


# ====== НАЛАШТУВАННЯ ======
XLSX = "data/source/test_xlsx.xlsx"
SHEET = "0"
OUT_DIR = Path("data/exports_excel")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUT = OUT_DIR / "test_from_excel.csv"
XLSX_REPORT_OUT = OUT_DIR / "report_from_excel.xlsx"
# ==========================


def main() -> None:
    xlsx_path = Path(XLSX)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"XLSX not found: {xlsx_path}")

    # 1) Листи книги
    sheets = list_excel_sheets(str(xlsx_path))
    print("\n=== SHEETS ===")
    for s in sheets:
        print("-", s)

    # 2) Excel -> CSV (1 sheet)
    print("\n=== EXPORT ONE SHEET -> CSV ===")
    res = excel_to_csv(
        str(xlsx_path),
        sheet_name=SHEET,
        out_csv=str(CSV_OUT),
        header_row=0,        # перший рядок — назви питань
        allow_missing=True,  # пропуски допускаємо
    )
    print("Saved CSV:", res.out_csv)
    print("Rows:", res.n_rows, "Cols:", res.n_cols)
    print("Columns (first 10):", res.columns[:10])

    # 3) Запускаємо аналіз на CSV
    print("\n=== RELIABILITY (Cronbach) ===")
    rep = compare_reliability_csv([str(CSV_OUT)], rating_scale="nominal")
    print("STATUS:", rep.get("status", "ok"))
    print("SUMMARY:", rep.get("summary"))

    # 4) Excel report
    print("\n=== EXPORT REPORT -> XLSX ===")
    compare_reliability_to_excel(
        [str(CSV_OUT)],
        out_path=str(XLSX_REPORT_OUT),
        rating_scale="nominal",
    )
    print("Saved report:", XLSX_REPORT_OUT.as_posix())

    # EXPORT MULTIPLE SHEETS
    # print("\n=== EXPORT MULTIPLE SHEETS -> CSVs ===")
    # multi = workbook_to_csvs(
    #     str(xlsx_path),
    #     out_dir=str(OUT_DIR / "sheets"),
    #     sheets=["RAW", "METHOD2"],   # list of sheets
    #     filename_prefix="sheet_",
    # )
    # for r in multi:
    #     print(r.sheet, "->", r.out_csv)


if __name__ == "__main__":
    main()
