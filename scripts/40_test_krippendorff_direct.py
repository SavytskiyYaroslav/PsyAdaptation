from pathlib import Path
import numpy as np
import pandas as pd

from psyadapt.reliability import KrippendorffAlpha
from psyadapt import compare_reliability_to_excel


paths = [
    "data/leary/leary_t1.csv",
    "data/leary/leary_t2.csv",
    "data/leary/leary_t3.csv",
]

# ---------- DIRECT (smoke) ----------
dfs = [pd.read_csv(p) for p in paths]

cols0 = list(dfs[0].columns)
shape0 = dfs[0].shape
for i, df in enumerate(dfs[1:], start=2):
    if list(df.columns) != cols0:
        raise ValueError(f"Columns mismatch in file{i}")
    if df.shape != shape0:
        raise ValueError(f"Shape mismatch in file{i}: {df.shape} vs {shape0}")

X = np.stack([df.to_numpy(dtype=float) for df in dfs], axis=0)  # (raters, objects, items)

est = KrippendorffAlpha().fit(X)

print("KrippendorffAlpha score:", est.score_)
print("Details keys:", list(est.details_.keys()))
print("n_raters:", est.details_.get("n_raters"))
print("n_objects:", est.details_.get("n_objects"))
print("n_items:", est.details_.get("n_items"))

# ---------- EXCEL (via compare, forced primary) ----------
out = Path("data/leary/reports/leary_kripp_compare.xlsx")
out.parent.mkdir(parents=True, exist_ok=True)

compare_reliability_to_excel(
    paths,
    out_path=str(out),
    rating_scale="nominal",
    primary="krippendorff",
)

print("Excel saved:", out.as_posix())
