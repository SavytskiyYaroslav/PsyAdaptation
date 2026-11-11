from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import re

import numpy as np
import pandas as pd


PathLike = Union[str, Path]


# Columns that represent respondent identifiers are *not* test items.
# If they leak into reliability computations they will distort results.
_ID_COL_RE = re.compile(
    r"^(id|respondent(_id)?|participant(_id)?|subject(_id)?|index)$",
    re.IGNORECASE,
)


def _drop_service_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop obvious service columns (id/index/unnamed) and fully-empty columns."""
    if df is None or df.shape[1] == 0:
        return df

    cols = [str(c) for c in df.columns]

    drop: list[str] = []
    for c in cols:
        c_stripped = c.strip()
        # Pandas sometimes writes index column as "Unnamed: 0" when exported.
        if c_stripped.lower().startswith("unnamed"):
            drop.append(c)
            continue
        if _ID_COL_RE.match(c_stripped):
            drop.append(c)

    if drop:
        df = df.drop(columns=drop, errors="ignore")

    # Drop fully empty columns
    if df.shape[1] > 0:
        df = df.dropna(axis=1, how="all")
    return df


def _read_csv(path: PathLike) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize header names (strip whitespace) while keeping the original order.
    if hasattr(df, "columns"):
        df.columns = [str(c).strip() for c in df.columns]

    # Drop service columns (id/index/unnamed) so they don't pollute statistics.
    df = _drop_service_columns(df)

    # We assume only one test per file and all remaining columns are items.
    # Convert to numeric; non-numeric becomes NaN (validation can catch it).
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def load_csv_matrix(path: PathLike) -> np.ndarray:
    """
    Load single CSV as a 2D matrix: (n_objects, n_items).
    Rows are objects/respondents; columns are items.
    """
    df = _read_csv(path)
    return df.to_numpy(dtype=float)


def load_csv_raters(paths: Iterable[PathLike]) -> np.ndarray:
    """
    Load multiple CSVs (one per rater) as a 3D tensor:
    (n_raters, n_objects, n_items).
    Matching is positional (by row order), since id_col is absent.
    """
    paths = list(paths)
    if len(paths) == 0:
        raise ValueError("No CSV paths provided.")

    frames = [_read_csv(p) for p in paths]
    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        raise ValueError(
            f"All CSV files must have identical shape (rows, cols). Got: {sorted(shapes)}"
        )

    arrays = [f.to_numpy(dtype=float) for f in frames]
    return np.stack(arrays, axis=0)


def load_csv_dataframe(path: PathLike) -> pd.DataFrame:
    """
    Load single CSV as a pandas DataFrame with numeric coercion.
    Preserves column names for item diagnostics.
    """
    return _read_csv(path)


def load_csv_raters_with_names(paths: Iterable[PathLike]) -> tuple[np.ndarray, list[str]]:
    """
    Load multiple CSVs (one per rater) as:
      X3 shape: (n_raters, n_objects, n_items)
      item_names: list of column names from the first CSV

    Assumptions:
    - All CSVs have identical number of rows (respondents) and columns (items)
    - Columns must match exactly in the same order
    """
    paths = list(paths)
    if len(paths) == 0:
        raise ValueError("No CSV paths provided.")

    frames = [_read_csv(p) for p in paths]  # <- твій існуючий reader (pandas df)
    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        raise ValueError(f"All CSV files must have identical shape (rows, cols). Got: {sorted(shapes)}")

    cols0 = list(frames[0].columns)
    for i, f in enumerate(frames[1:], start=1):
        if list(f.columns) != cols0:
            raise ValueError("All CSV files must have identical columns in the same order.")

    arrays = [f.to_numpy(dtype=float) for f in frames]
    X3 = np.stack(arrays, axis=0)
    item_names = [str(c) for c in cols0]
    return X3, item_names
