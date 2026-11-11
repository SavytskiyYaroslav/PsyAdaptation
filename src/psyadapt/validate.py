from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import os
import pandas as pd
import numpy as np
import re


Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    file: Optional[str] = None
    column: Optional[str] = None


def _issue(sev: Severity, code: str, msg: str, file: Optional[str] = None, col: Optional[str] = None) -> ValidationIssue:
    return ValidationIssue(severity=sev, code=code, message=msg, file=file, column=col)


_ID_COL_RE = re.compile(
    r"^(id|respondent(_id)?|participant(_id)?|subject(_id)?|index)$",
    re.IGNORECASE,
)


def _drop_service_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier / index / empty 'Unnamed:' columns.

    Validation and computation should operate on item columns only.
    """
    if df is None or not hasattr(df, "columns"):
        return df

    cols = [str(c).strip() for c in df.columns]
    df = df.copy()
    df.columns = cols

    drop = []
    for c in cols:
        if c.startswith("Unnamed:"):
            drop.append(c)
            continue
        if _ID_COL_RE.match(c):
            drop.append(c)

    if drop:
        df = df.drop(columns=drop, errors="ignore")

    # Drop fully empty columns
    if df.shape[1] > 0:
        df = df.dropna(axis=1, how="all")
    return df


def _read_csv(path: str) -> Tuple[Optional[pd.DataFrame], List[ValidationIssue]]:
    issues: List[ValidationIssue] = []

    if not isinstance(path, str) or not path.strip():
        issues.append(_issue("error", "path.invalid", "CSV path is empty or not a string.", file=str(path)))
        return None, issues

    if not os.path.exists(path):
        issues.append(_issue("error", "file.not_found", "File does not exist.", file=path))
        return None, issues

    if not os.path.isfile(path):
        issues.append(_issue("error", "file.not_a_file", "Path is not a file.", file=path))
        return None, issues

    try:
        df = pd.read_csv(path)
    except Exception as e:
        issues.append(_issue("error", "csv.read_failed", f"Failed to read CSV: {e}", file=path))
        return None, issues

    # Normalize headers and drop obvious service columns (id/index/unnamed).
    df = _drop_service_columns(df)

    return df, issues


def validate_single_test_csv(
    path: str,
    *,
    allow_missing: bool = False,
    require_min_rows: int = 2,
    require_min_cols: int = 2,
) -> Dict[str, Any]:
    """
    Validate '1 CSV = 1 test' format:
      - rows = respondents
      - columns = items
      - currently expects numeric-only columns (your ideal mode)
    """
    df, issues = _read_csv(path)

    if df is None:
        return {"ok": False, "issues": [asdict(i) for i in issues]}

    # Empty checks
    if df.shape[0] < require_min_rows:
        issues.append(_issue("error", "csv.too_few_rows", f"Need at least {require_min_rows} rows (respondents).", file=path))
    if df.shape[1] < require_min_cols:
        issues.append(_issue("error", "csv.too_few_cols", f"Need at least {require_min_cols} columns (items).", file=path))

    # Column name sanity
    if df.columns.duplicated().any():
        issues.append(_issue("error", "csv.duplicate_columns", "CSV contains duplicate column names.", file=path))

    # Numeric-only (ideal mode)
    numeric_df = df.apply(pd.to_numeric, errors="coerce")

    # Detect non-numeric columns (everything becomes NaN)
    for col in df.columns:
        orig = df[col]
        conv = numeric_df[col]
        # if there exists at least one non-empty original value, but all converted are NaN -> non-numeric column
        has_any = orig.notna().any()
        all_nan_after = conv.isna().all()
        if has_any and all_nan_after:
            issues.append(
                _issue(
                    "error",
                    "csv.non_numeric_column",
                    "Column contains non-numeric values (ideal mode expects numeric item columns only).",
                    file=path,
                    col=str(col),
                )
            )

    # Missing values
    if numeric_df.isna().any().any():
        if allow_missing:
            issues.append(_issue("warning", "csv.missing_values", "Missing values (NaN) detected.", file=path))
        else:
            issues.append(_issue("error", "csv.missing_values", "Missing values (NaN) detected but allow_missing=False.", file=path))

    ok = not any(i.severity == "error" for i in issues)
    return {"ok": ok, "issues": [asdict(i) for i in issues], "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])}}


def validate_raters_csvs(
    paths: Sequence[str],
    *,
    allow_missing: bool = True,
    require_same_columns: bool = True,
    require_same_shape: bool = True,
    require_min_raters: int = 2,
    require_min_rows: int = 2,
    require_min_cols: int = 1,
) -> Dict[str, Any]:
    """
    Validate multiple expert CSVs:
      - all readable
      - numeric-only in ideal mode
      - (optional) same columns and same shape
      - missing values: allowed by default (Krippendorff/ICC can handle; kappa methods may not)
    """
    issues: List[ValidationIssue] = []

    if len(paths) < require_min_raters:
        issues.append(_issue("error", "raters.too_few_files", f"Need at least {require_min_raters} CSV file(s)."))
        return {"ok": False, "issues": [asdict(i) for i in issues]}

    dfs: List[pd.DataFrame] = []
    shapes: List[Tuple[int, int]] = []
    columns_list: List[List[str]] = []

    for p in paths:
        df, iss = _read_csv(p)
        issues.extend(iss)
        if df is None:
            continue

        if df.shape[0] < require_min_rows:
            issues.append(_issue("error", "csv.too_few_rows", f"Need at least {require_min_rows} rows (respondents).", file=p))
        if df.shape[1] < require_min_cols:
            issues.append(_issue("error", "csv.too_few_cols", f"Need at least {require_min_cols} columns (items).", file=p))

        if df.columns.duplicated().any():
            issues.append(_issue("error", "csv.duplicate_columns", "CSV contains duplicate column names.", file=p))

        # Numeric-only check (ideal mode)
        numeric_df = df.apply(pd.to_numeric, errors="coerce")
        for col in df.columns:
            orig = df[col]
            conv = numeric_df[col]
            has_any = orig.notna().any()
            all_nan_after = conv.isna().all()
            if has_any and all_nan_after:
                issues.append(_issue("error", "csv.non_numeric_column", "Non-numeric column detected.", file=p, col=str(col)))

        if numeric_df.isna().any().any():
            if allow_missing:
                issues.append(_issue("warning", "csv.missing_values", "Missing values (NaN) detected.", file=p))
            else:
                issues.append(_issue("error", "csv.missing_values", "Missing values detected but allow_missing=False.", file=p))

        dfs.append(df)
        shapes.append((int(df.shape[0]), int(df.shape[1])))
        columns_list.append([str(c) for c in df.columns])

    if require_same_shape and shapes:
        if len(set(shapes)) > 1:
            issues.append(_issue("error", "raters.shape_mismatch", f"CSV shapes differ across raters: {shapes}"))

    if require_same_columns and columns_list:
        # exact match (including order)
        first = columns_list[0]
        for idx, cols in enumerate(columns_list[1:], start=1):
            if cols != first:
                issues.append(
                    _issue(
                        "error",
                        "raters.columns_mismatch",
                        "Column names/order differ across rater files. (Ideal mode requires identical columns in same order.)",
                        file=str(paths[idx]),
                    )
                )
                break

    ok = not any(i.severity == "error" for i in issues)
    return {
        "ok": ok,
        "issues": [asdict(i) for i in issues],
        "n_files": len(paths),
        "shapes": shapes,
        "columns": columns_list[0] if columns_list else None,
    }
