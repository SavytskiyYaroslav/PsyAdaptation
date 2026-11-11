from pathlib import Path
from psyadapt import compare_reliability_csv, compare_reliability_to_excel

paths = [
    "data/leary/leary_t1.csv",
    "data/leary/leary_t2.csv",
]

OUT = Path("data/leary/reports/leary_cohen_icc.xlsx")
OUT.parent.mkdir(parents=True, exist_ok=True)

rep = compare_reliability_csv(paths, rating_scale="nominal")
print("STATUS:", rep.get("status", "ok"))
print("SUMMARY:", rep.get("summary"))

compare_reliability_to_excel(paths, out_path=str(OUT), rating_scale="nominal")
print("Excel saved:", OUT.as_posix())
