from pathlib import Path
from psyadapt import compare_reliability_csv, compare_reliability_to_excel


CSV = "data/leary/vertical_workbook_MATRIX.csv"
OUT = Path("data/leary/reports/leary_cronbach.xlsx")
OUT.parent.mkdir(parents=True, exist_ok=True)

rep = compare_reliability_csv([CSV], rating_scale="nominal")
print("STATUS:", rep.get("status", "ok"))
print("SUMMARY:", rep.get("summary"))
rep = compare_reliability_csv([CSV], rating_scale="nominal")
print(rep["per_item"][:3])
diag = rep["results"]["primary"]["details"].get("diagnostics")
print("SIM:", diag.get("simulation"))

compare_reliability_to_excel([CSV], out_path=str(OUT), rating_scale="nominal")
print("Excel saved:", OUT.as_posix())
