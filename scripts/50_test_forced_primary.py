from pathlib import Path
from psyadapt import compare_reliability_csv, compare_reliability_to_excel

OUT_DIR = Path("data/leary/reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

paths3 = [
    "data/leary/leary_t1.csv",
    "data/leary/leary_t2.csv",
    "data/leary/leary_t3.csv",
]

# 5.1) auto на 3 файлах (має бути Fleiss)
rep_auto = compare_reliability_csv(paths3, rating_scale="nominal", primary="auto")
print("[AUTO] primary:", rep_auto["summary"]["primary_estimator"])

# 5.2) forced krippendorff на 3 файлах
rep_k = compare_reliability_csv(paths3, rating_scale="nominal", primary="krippendorff")
print("[FORCED KRIPP] primary:", rep_k["summary"]["primary_estimator"])

compare_reliability_to_excel(
    paths3,
    out_path=str(OUT_DIR / "leary_forced_krippendorff.xlsx"),
    rating_scale="nominal",
    primary="krippendorff",
)
print("Excel saved:", (OUT_DIR / "leary_forced_krippendorff.xlsx").as_posix())

# 5.3) forced cohen на 2 файлах
paths2 = ["data/leary/leary_t1.csv", "data/leary/leary_t2.csv"]
rep_c = compare_reliability_csv(paths2, rating_scale="nominal", primary="cohen")
print("[FORCED COHEN] primary:", rep_c["summary"]["primary_estimator"])

# 5.4) негативний кейс: 1 файл + krippendorff (має помилку або status=error залежно від реалізації)
try:
    rep_bad = compare_reliability_csv(["data/leary/leary_interpersonal.csv"], primary="krippendorff")
    print("[BAD] got status:", rep_bad.get("status", "ok"), "summary:", rep_bad.get("summary"))
except Exception as e:
    print("[BAD] raised as expected:", repr(e))
