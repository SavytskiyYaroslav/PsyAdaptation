from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def load_matrix(path: Path) -> pd.DataFrame:
    """Load matrix-like data: rows=respondents, cols=items/questions."""
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    suf = path.suffix.lower()
    if suf in {".xlsx", ".xlsm", ".xls"}:
        # If your matrix is stored on the first sheet - this is enough.
        # If needed, add: sheet_name="SHEET"
        df = pd.read_excel(path, engine="openpyxl")
    elif suf == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported input format: {suf} (use .xlsx or .csv)")

    # keep only numeric-like columns (your pipeline expects numeric)
    # but do NOT drop columns silently if that’s risky for you — feel free to remove this filter
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df.shape[0] < 2 or df.shape[1] < 1:
        raise ValueError(f"Matrix seems too small: shape={df.shape}")

    return df


def make_variant(
    df: pd.DataFrame,
    p: float,
    seed: int,
    delta_choices: tuple[float, ...] = (-1.0, 1.0),
) -> pd.DataFrame:
    """
    Create pseudo-rater/retst variant by changing p-fraction of cells by +/- delta.
    NaNs are preserved.
    Values are clipped per-column to original column min/max.
    """
    rng = np.random.default_rng(seed)

    base = df.to_numpy(dtype=float)

    # mask only finite cells
    finite = np.isfinite(base)
    rand = rng.random(base.shape)
    mask = (rand < p) & finite

    delta = rng.choice(delta_choices, size=base.shape).astype(float)
    out = base.copy()
    out[mask] = out[mask] + delta[mask]

    out_df = pd.DataFrame(out, columns=df.columns)

    # clip per column to original min/max (ignoring NaN)
    for col in df.columns:
        lo = float(np.nanmin(df[col].to_numpy(dtype=float)))
        hi = float(np.nanmax(df[col].to_numpy(dtype=float)))
        out_df[col] = out_df[col].clip(lower=lo, upper=hi)

    return out_df


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate pseudo-rater CSVs (t1/t2/t3) from a matrix (.xlsx/.csv)."
    )
    ap.add_argument(
        "--src",
        default="data/leary/vertical_workbook_MATRIX.xlsx",
        help="Path to matrix file (.xlsx or .csv). Default: data/leary/vertical_workbook_MATRIX.xlsx",
    )
    ap.add_argument(
        "--out-dir",
        default="data/leary",
        help="Output directory. Default: data/leary",
    )
    ap.add_argument("--prefix", default="leary", help="Output prefix. Default: leary")
    ap.add_argument("--p2", type=float, default=0.10, help="Noise rate for t2. Default 0.10")
    ap.add_argument("--p3", type=float, default=0.10, help="Noise rate for t3. Default 0.10")
    ap.add_argument("--seed2", type=int, default=42, help="Seed for t2. Default 42")
    ap.add_argument("--seed3", type=int, default=123, help="Seed for t3. Default 123")
    ap.add_argument(
        "--delta",
        default="-1,1",
        help="Comma-separated delta choices, e.g. '-1,1' or '-2,-1,1,2'. Default '-1,1'",
    )

    args = ap.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    delta_choices = tuple(float(x.strip()) for x in str(args.delta).split(",") if x.strip())
    if not delta_choices:
        raise ValueError("Empty --delta choices. Example: --delta -1,1")

    df = load_matrix(src)

    t1 = out_dir / f"{args.prefix}_t1.csv"
    t2 = out_dir / f"{args.prefix}_t2.csv"
    t3 = out_dir / f"{args.prefix}_t3.csv"

    # T1 = baseline (no noise)
    df.to_csv(t1, index=False, encoding="utf-8")

    # T2 / T3 = pseudo-raters
    make_variant(df, p=float(args.p2), seed=int(args.seed2), delta_choices=delta_choices).to_csv(
        t2, index=False, encoding="utf-8"
    )
    make_variant(df, p=float(args.p3), seed=int(args.seed3), delta_choices=delta_choices).to_csv(
        t3, index=False, encoding="utf-8"
    )

    print("Created:")
    print(" ", t1.as_posix())
    print(" ", t2.as_posix(), f"(p={args.p2}, seed={args.seed2}, delta={delta_choices})")
    print(" ", t3.as_posix(), f"(p={args.p3}, seed={args.seed3}, delta={delta_choices})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
