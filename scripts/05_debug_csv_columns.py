from psyadapt.io.csv import load_csv_dataframe

CSV = "data/leary/vertical_workbook_MATRIX.csv"  # або твій актуальний csv

df = load_csv_dataframe(CSV)

print("COLUMNS:", df.columns.tolist())
print("N_COLS:", len(df.columns))
