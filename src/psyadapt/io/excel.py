from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import re


@dataclass(frozen=True)
class ExcelToCsvResult:
    source_xlsx: str
    sheet: str
    out_csv: str
    n_rows: int
    n_cols: int
    columns: list[str]



_SHEET_ID_RE = re.compile(r"^\d{3}$")

def _sheet_id_key(name: str) -> int:
    """
    Convert sheet name '001' -> 1
    Raises ValueError if format is invalid.
    """
    if not _SHEET_ID_RE.match(name):
        raise ValueError(
            f"Invalid sheet name '{name}'. Expected format: 001, 002, ..."
        )
    return int(name)


def list_excel_sheets(xlsx_path: str) -> list[str]:
    """Return sheet names in workbook."""
    xlsx_path = str(xlsx_path)
    xl = pd.ExcelFile(xlsx_path)
    return list(xl.sheet_names)


def _normalize_columns(cols: Sequence[str]) -> list[str]:
    """Trim, ensure unique, replace empties."""
    cleaned: list[str] = []
    seen: dict[str, int] = {}

    for i, c in enumerate(cols):
        name = "" if c is None else str(c)
        name = name.strip()
        if name == "" or name.lower().startswith("unnamed"):
            name = f"Q{i+1}"

        if name in seen:
            seen[name] += 1
            name = f"{name}__{seen[name]}"
        else:
            seen[name] = 1

        cleaned.append(name)

    return cleaned


def read_excel_table(
    xlsx_path: str,
    *,
    sheet_name: str | int | None = None,
    header_row: int = 0,
    usecols: Optional[str] = None,
    skiprows: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """
    Read a single sheet as a response table.
    - header_row: 0 means first row is header
    - usecols: Excel-like string e.g. "B:I"
    """
    df = pd.read_excel(
        xlsx_path,
        sheet_name=sheet_name,
        header=header_row,
        usecols=usecols,
        skiprows=skiprows,
        engine="openpyxl",
    )

    # drop fully empty rows/cols
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    # normalize column names
    df.columns = _normalize_columns(df.columns)

    # strip strings in cells
    df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

    return df


def excel_to_csv(
    xlsx_path: str,
    *,
    out_csv: str,
    sheet_name: str | int | None = None,
    header_row: int = 0,
    usecols: Optional[str] = None,
    skiprows: Optional[Iterable[int]] = None,
    allow_missing: bool = True,
) -> ExcelToCsvResult:
    """
    Convert one Excel sheet to one CSV (responses table).
    """
    df = read_excel_table(
        xlsx_path,
        sheet_name=sheet_name,
        header_row=header_row,
        usecols=usecols,
        skiprows=skiprows,
    )

    # Convert to numeric where possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if not allow_missing and df.isna().any().any():
        raise ValueError("Missing/non-numeric values detected, but allow_missing=False")

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    return ExcelToCsvResult(
        source_xlsx=str(xlsx_path),
        sheet=str(sheet_name) if sheet_name is not None else "0",
        out_csv=str(out_path),
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        columns=[str(c) for c in df.columns],
    )


def workbook_to_csvs(
    xlsx_path: str,
    *,
    out_dir: str,
    sheets: Optional[Sequence[str]] = None,
    header_row: int = 0,
    allow_missing: bool = True,
    filename_prefix: str = "",
) -> list[ExcelToCsvResult]:
    """
    Convert multiple sheets from one workbook into multiple CSVs.
    Each sheet becomes one CSV file.
    """
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    all_sheets = list_excel_sheets(xlsx_path)
    target_sheets = list(sheets) if sheets is not None else all_sheets

    results: list[ExcelToCsvResult] = []
    for sh in target_sheets:
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in sh).strip("_")
        out_csv = out_dir_p / f"{filename_prefix}{safe_name}.csv"
        res = excel_to_csv(
            xlsx_path,
            out_csv=str(out_csv),
            sheet_name=sh,
            header_row=header_row,
            allow_missing=allow_missing,
        )
        results.append(res)

    return results


def excel_to_matrix(
    xlsx_path: str,
    *,
    sheet_name: str | int | None = None,
    header_row: int = 0,
    usecols: Optional[str] = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Convenience: read Excel responses and return numpy matrix + item names.
    """
    df = read_excel_table(
        xlsx_path,
        sheet_name=sheet_name,
        header_row=header_row,
        usecols=usecols,
    )
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    X = df.to_numpy(dtype=float)
    item_names = [str(c) for c in df.columns]
    return X, item_names


# ----------------------------
# Vertical workbook -> matrix
# ----------------------------

def _col_letter_to_index(col: str) -> int:
    """Excel column letter (A, B, AA) -> 0-based index."""
    col = str(col).strip().upper()
    if not re.fullmatch(r"[A-Z]+", col):
        raise ValueError(f"Invalid Excel column letter: {col!r}")
    n = 0
    for ch in col:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


def workbook_vertical_to_matrix(
    xlsx_path: str,
    *,
    start_row: int = 10,
    question_col: str = "A",
    score_col: str = "B",
    header_sheet: str | None = None,
    sheets: Sequence[str] | None = None,
    on_duplicate_question: str = "first",
    strip_questions: bool = True,
    enforce_sheet_ids: bool = True,
    sort_sheets: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Convert a workbook where each sheet has vertical rows (Question, Score)
    into a matrix DataFrame: rows = sheets (respondents), columns = questions.

    Assumptions (current psyadapt workflow):
    - Each sheet corresponds to one respondent.
    - Questions and scores are in two columns.
    - Scores are numeric (or numeric-like).
    - Canonical questions list is taken from the first sheet (or header_sheet).
    - If a question repeats, keep the FIRST occurrence (default).

    Parameters:
      start_row: 1-based Excel row index where data begins (e.g., 10).
      question_col / score_col: Excel column letters (e.g., 'A', 'B').
      header_sheet: sheet used to extract the canonical questions list.
      sheets: optional subset of sheets to include (order preserved).
      on_duplicate_question: 'first' | 'last' | 'error'
    """
    if start_row < 1:
        raise ValueError("start_row must be >= 1 (Excel is 1-based).")

    xlsx_path = str(xlsx_path)
    xl = pd.ExcelFile(xlsx_path, engine="openpyxl")

    all_sheets = list(xl.sheet_names)
    target_sheets = list(sheets) if sheets is not None else all_sheets
    if len(target_sheets) == 0:
        raise ValueError("No sheets selected.")
        # Enforce and sort sheets like 001, 002, ...
        enforce_sheet_ids: bool = True
        sort_sheets: bool = True

        if enforce_sheet_ids:
            bad = [sh for sh in target_sheets if not _SHEET_ID_RE.match(str(sh))]
            if bad:
                raise ValueError(
                    f"Invalid sheet names {bad}. Expected format: 001, 002, ..."
                )

        if sort_sheets:
            # numeric sort: "001" -> 1
            target_sheets = sorted(target_sheets, key=lambda s: _sheet_id_key(str(s)))

    hdr = header_sheet if header_sheet is not None else target_sheets[0]
    if hdr not in all_sheets:
        raise ValueError(f"header_sheet={hdr!r} not found. Available: {all_sheets}")

    q_idx = _col_letter_to_index(question_col)
    s_idx = _col_letter_to_index(score_col)

    def _read_vertical(sheet: str) -> pd.DataFrame:
        df = pd.read_excel(
            xlsx_path,
            sheet_name=sheet,
            header=None,
            skiprows=start_row - 1,
            usecols=[q_idx, s_idx],
            engine="openpyxl",
        )
        df = df.rename(columns={df.columns[0]: "question", df.columns[1]: "score"})
        df = df.dropna(subset=["question"])
        if strip_questions:
            df["question"] = df["question"].map(lambda x: str(x).strip() if x is not None else x)
        df = df[df["question"].astype(str).str.len() > 0]
        return df

    # 1) Canonical questions from header sheet
    df_hdr = _read_vertical(hdr)
    questions_raw = [str(x) for x in df_hdr["question"].tolist()]

    # Handle duplicates in header
    if on_duplicate_question not in {"first", "last", "error"}:
        raise ValueError("on_duplicate_question must be one of: 'first', 'last', 'error'")

    seen: set[str] = set()
    questions: list[str] = []
    if on_duplicate_question == "first":
        for q in questions_raw:
            if q not in seen:
                seen.add(q)
                questions.append(q)
    elif on_duplicate_question == "last":
        tmp: list[str] = []
        for q in reversed(questions_raw):
            if q not in seen:
                seen.add(q)
                tmp.append(q)
        questions = list(reversed(tmp))
    else:
        dupes = [q for q in questions_raw if questions_raw.count(q) > 1]
        if dupes:
            raise ValueError(f"Duplicate questions detected in header sheet: {sorted(set(dupes))}")

    if len(questions) == 0:
        raise ValueError("No questions found in header sheet (after start_row).")

    # 2) Build matrix rows for each sheet
    rows: list[list[float]] = []
    index: list[str] = []

    for sh in target_sheets:
        df_sh = _read_vertical(sh)
        m: dict[str, float] = {}
        for q, v in zip(df_sh["question"].tolist(), df_sh["score"].tolist()):
            key = str(q)
            if key in m:
                if on_duplicate_question == "first":
                    continue
                elif on_duplicate_question == "last":
                    pass
                else:
                    raise ValueError(f"Duplicate question {key!r} in sheet {sh!r}.")
            m[key] = v

        rows.append([m.get(q, np.nan) for q in questions])
        index.append(sh)

    out = pd.DataFrame(rows, columns=questions, index=index)

    # Ensure numeric dtype where possible
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out, questions


def workbook_vertical_to_csv(
    xlsx_path: str,
    *,
    out_csv: str,
    start_row: int = 10,
    question_col: str = "A",
    score_col: str = "B",
    header_sheet: str | None = None,
    sheets: Sequence[str] | None = None,
    on_duplicate_question: str = "first",
    strip_questions: bool = True,
    enforce_sheet_ids: bool = True,
    sort_sheets: bool = True,
) -> ExcelToCsvResult:
    """Convenience: convert vertical workbook into ONE matrix CSV."""
    df, _ = workbook_vertical_to_matrix(
        xlsx_path,
        start_row=start_row,
        question_col=question_col,
        score_col=score_col,
        header_sheet=header_sheet,
        sheets=sheets,
        on_duplicate_question=on_duplicate_question,
        strip_questions=strip_questions,
        enforce_sheet_ids=enforce_sheet_ids,
        sort_sheets=sort_sheets,
    )
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    return ExcelToCsvResult(
        source_xlsx=str(xlsx_path),
        sheet="WORKBOOK_VERTICAL",
        out_csv=str(out_path),
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        columns=[str(c) for c in df.columns],
    )
