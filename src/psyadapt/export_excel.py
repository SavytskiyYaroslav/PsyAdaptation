from __future__ import annotations

import json
import numpy as np

from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import CellIsRule


from psyadapt.compare import compare_reliability_csv

RatingScale = Literal["nominal", "ordinal", "interval"]
Agg = Literal["mean", "median"]


def _set_col_width(ws, col_idx: int, width: float) -> None:
    ws.column_dimensions[get_column_letter(col_idx)].width = width


def _header_style(cell) -> None:
    cell.font = Font(bold=True, color="FFFFFF")
    cell.fill = PatternFill("solid", fgColor="2F5597")  # dark blue
    cell.alignment = Alignment(vertical="center", wrap_text=True)


def _summary_style(cell) -> None:
    cell.font = Font(bold=True)
    cell.fill = PatternFill("solid", fgColor="E7E6E6")  # light gray
    cell.alignment = Alignment(vertical="center", wrap_text=True)


def _kv_style(cell_key, cell_val) -> None:
    cell_key.font = Font(bold=True)
    cell_key.alignment = Alignment(vertical="center", wrap_text=True)
    cell_val.alignment = Alignment(vertical="center", wrap_text=True)


def _cell_key(ws, r, c, text):
    cell = ws.cell(row=r, column=c, value=text)
    cell.font = Font(bold=True)
    cell.alignment = Alignment(vertical="center", wrap_text=True)
    return cell

def _cell_val(ws, r, c, value):
    cell = ws.cell(row=r, column=c, value=value)
    cell.alignment = Alignment(vertical="center", wrap_text=True)
    return cell


def _extract_cronbach_diagnostics(report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Try to locate Cronbach diagnostics in the compare report.
    Supports a few shapes to stay backward-compatible.
    """
    primary = (report.get("results") or {}).get("primary") or {}

    diag = primary.get("diagnostics")
    if isinstance(diag, dict):
        return diag

    details = primary.get("details")
    if isinstance(details, dict):
        diag = details.get("diagnostics")
        if isinstance(diag, dict):
            return diag

    payload = primary.get("payload")
    if isinstance(payload, dict):
        diag = payload.get("diagnostics")
        if isinstance(diag, dict):
            return diag

    return None


def _add_cronbach_diagnostics_sheet(wb: Workbook, report: Dict[str, Any]) -> None:
    """Create a readable Cronbach diagnostics sheet (for psychologists)."""
    diag = _extract_cronbach_diagnostics(report)
    if not isinstance(diag, dict):
        return

    items = diag.get("items", []) or []
    if not isinstance(items, list) or len(items) == 0:
        return

    overall_alpha = diag.get("overall_alpha", None)

    ws = wb.create_sheet("Cronbach diagnostics")

    # Columns (psychologist-friendly order)
    headers = [
        "item_name",
        "CITC",
        "CITC after reverse",
        "alpha if deleted",
        "Δ alpha if deleted",
        "suggest_reverse",
        "remove_candidate",
        "flag",
        "reasons",
    ]

    # Set widths
    widths = [26, 10, 16, 14, 14, 14, 16, 12, 52]
    for i, w in enumerate(widths, start=1):
        _set_col_width(ws, i, w)

    # Title
    ws["A1"] = "Cronbach α — item diagnostics"
    _summary_style(ws["A1"])
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
    ws.row_dimensions[1].height = 22

    # Info row
    ws["A2"] = "Overall alpha"
    ws["B2"] = overall_alpha if overall_alpha is not None else ""
    _kv_style(ws["A2"], ws["B2"])
    ws["D2"] = "CITC threshold"
    ws["E2"] = diag.get("citc_threshold", "")
    _kv_style(ws["D2"], ws["E2"])
    ws["G2"] = "remove_delta"
    ws["H2"] = diag.get("remove_delta", "")
    _kv_style(ws["G2"], ws["H2"])

    # Header row at 4
    header_row = 4
    for c, h in enumerate(headers, start=1):
        cell = ws.cell(header_row, c, h)
        _header_style(cell)

    # Data
    start_row = header_row + 1
    def _num(x):
        # Cast numpy scalars to plain python numbers for Excel
        if x is None:
            return None
        try:
            import numpy as _np
            if isinstance(x, (_np.generic,)):
                x = x.item()
        except Exception:
            pass
        return x

    for r, it in enumerate(items, start=start_row):
        ws.cell(r, 1, it.get('item_name'))
        ws.cell(r, 2, _num(it.get('corrected_item_total_corr')))
        ws.cell(r, 3, _num(it.get('citc_reversed')))
        ws.cell(r, 4, _num(it.get('alpha_if_deleted')))
        ws.cell(r, 5, _num(it.get('delta_alpha_if_deleted')))
        # booleans -> real Excel booleans (TRUE/FALSE), not strings
        ws.cell(r, 6, bool(it.get('suggest_reverse')) if it.get('suggest_reverse') is not None else None)
        ws.cell(r, 7, bool(it.get('remove_candidate')) if it.get('remove_candidate') is not None else None)
        ws.cell(r, 8, it.get('flag'))
        reasons_val = it.get('reasons')
        if isinstance(reasons_val, (list, tuple, set)):
            reasons_val = '; '.join(str(x) for x in reasons_val)
        elif isinstance(reasons_val, dict):
            reasons_val = json.dumps(reasons_val, ensure_ascii=False)
        ws.cell(r, 9, reasons_val)

    last_row = start_row + len(items) - 1

    # Number formats
    for r in range(start_row, last_row + 1):
        for c in (2, 3, 4, 5):
            cell = ws.cell(r, c)
            cell.number_format = "0.000"
        for c in (6, 7):
            ws.cell(r, c).number_format = "GENERAL"

    # Freeze and filter
    ws.freeze_panes = ws.cell(start_row, 1)  # keeps title+info+headers
    ws.auto_filter.ref = f"A{header_row}:I{last_row}"

    # Conditional formatting
    # CITC: <0 red, 0..0.30 yellow
    red = red = PatternFill(fill_type="solid", start_color="F8CBAD", end_color="F8CBAD")
    yellow = yellow = PatternFill(fill_type="solid", start_color="FFF2CC", end_color="FFF2CC")
    green = green = PatternFill(fill_type="solid", start_color="C6EFCE", end_color="C6EFCE")

    # Helper ranges
    def _col_range(col: int) -> str:
        L = get_column_letter(col)
        return f"{L}{start_row}:{L}{max(last_row, start_row)}"


    # CITC (col 2)
    ws.conditional_formatting.add(
        _col_range(2),
        CellIsRule(operator="lessThan", formula=["0"], fill=red),
    )
    ws.conditional_formatting.add(
        _col_range(2),
        CellIsRule(
            operator="between",
            formula=["0", str(diag.get("citc_threshold", 0.3))],
            fill=yellow,
        ),
    )

    # CITC after reverse (col 3)
    ws.conditional_formatting.add(
        _col_range(3),
        CellIsRule(operator="lessThan", formula=["0"], fill=red),
    )
    ws.conditional_formatting.add(
        _col_range(3),
        CellIsRule(
            operator="between",
            formula=["0", str(diag.get("citc_threshold", 0.3))],
            fill=yellow,
        ),
    )

    # Alpha if deleted (col 4) vs overall alpha (if numeric)
    if isinstance(overall_alpha, (int, float)):
        ws.conditional_formatting.add(
            _col_range(4),
            CellIsRule(operator="greaterThan", formula=[str(float(overall_alpha))], fill=green),
        )
        ws.conditional_formatting.add(
            _col_range(4),
            CellIsRule(operator="lessThan", formula=[str(float(overall_alpha))], fill=red),
        )

    # Δ alpha if deleted (col 5): green / yellow / red bands
    ws.conditional_formatting.add(
        _col_range(5),
        CellIsRule(operator="greaterThan", formula=["0.005"], fill=green),
    )
    ws.conditional_formatting.add(
        _col_range(5),
        CellIsRule(operator="between", formula=["-0.005", "0.005"], fill=yellow),
    )
    ws.conditional_formatting.add(
        _col_range(5),
        CellIsRule(operator="lessThan", formula=["-0.005"], fill=red),
    )

    # suggest_reverse / remove_candidate
    ws.conditional_formatting.add(
        _col_range(6),
        CellIsRule(operator="equal", formula=["TRUE"], fill=red),
    )
    ws.conditional_formatting.add(
        _col_range(7),
        CellIsRule(operator="equal", formula=["TRUE"], fill=yellow),
    )

    # Flag column (col 8): highlight problematic rows
    ws.conditional_formatting.add(
        _col_range(8),
        CellIsRule(operator="equal", formula=['"problematic"'], fill=red),
    )
    ws.conditional_formatting.add(
        _col_range(8),
        CellIsRule(operator="equal", formula=['"ok"'], fill=green),
    )

    # Improve readability: wrap reasons
    for r in range(start_row, last_row + 1):
        ws.cell(r, 9).alignment = Alignment(vertical="top", wrap_text=True)
def _add_cronbach_cleaning_sheet(wb: Workbook, report: Dict[str, Any]) -> None:
    """Cleaning simulator sheet (scenarios + actionable recommendations)."""
    diag = _extract_cronbach_diagnostics(report)
    if not isinstance(diag, dict):
        return
    sim = diag.get("simulation", {}) or {}
    if not isinstance(sim, dict) or not sim:
        return

    # Recommendations inputs (computed by diagnostics)

    ws = wb.create_sheet("Cronbach cleaning")

    ws["A1"] = "Cronbach α — cleaning simulator (what-if)"
    _summary_style(ws["A1"])
    ws.merge_cells("A1:D1")
    ws.row_dimensions[1].height = 22

    # Scenario table
    ws["A3"] = "Scenario"
    ws["B3"] = "Alpha"
    ws["C3"] = "Δ vs original"
    ws["D3"] = "Notes"
    for c in range(1, 5):
        _header_style(ws.cell(3, c))

    alpha_orig = sim.get("alpha_original")

    # Extract item-level recommendations (names) from diagnostics
    items = diag.get("items") or []
    if not isinstance(items, list):
        items = []

    def _iname(it: Any) -> str:
        nm = it.get("item_name") or it.get("name") or it.get("item")
        return str(nm) if nm is not None else ""

    rev_items = [
        _iname(it)
        for it in items
        if isinstance(it, dict) and bool(it.get("suggest_reverse")) and _iname(it)
    ]

    rem_items = [
        _iname(it)
        for it in items
        if isinstance(it, dict) and (it.get("flag") == "problematic") and _iname(it)
    ]

    # remove_delta rule used by the cleaning simulator (fallback to 0.01)
    remove_delta = diag.get("remove_delta_rule")
    if remove_delta is None:
        remove_delta = diag.get("remove_delta")
    if remove_delta is None:
        remove_delta = 0.01
    scenarios = [
        ("Original", sim.get("alpha_original"), None, ""),
        ("After reverse only", sim.get("alpha_after_reverse_only"), None, "Reverse suggested items only: " + (", ".join(map(str, rev_items)) if rev_items else "—")),
        ("After remove only", sim.get("alpha_after_remove_only"), None, "Remove suggested items only: " + (", ".join(map(str, rem_items)) if rem_items else "—")),
        ("After reverse + remove", sim.get("alpha_after_reverse_and_remove"), None, "Reverse first, then remove. Reverse: " + (", ".join(map(str, rev_items)) if rev_items else "—") + "; Remove: " + (", ".join(map(str, rem_items)) if rem_items else "—")),
    ]

    # fill deltas
    filled = []
    for name, a, _, note in scenarios:
        d = (a - alpha_orig) if (a is not None and alpha_orig is not None) else None
        filled.append((name, a, d, note))

    def _delta_static_fill(value: Any):
        if not isinstance(value, (int, float)):
            return None
        if value > 0.005:
            return PatternFill("solid", fgColor="C6EFCE")  # green
        if value < -0.005:
            return PatternFill("solid", fgColor="F8CBAD")  # red
        return PatternFill("solid", fgColor="FFF2CC")  # yellow

    for i, (name, a, d, note) in enumerate(filled, start=4):
        ws.cell(i, 1, name)
        ws.cell(i, 2, a)
        ws.cell(i, 3, d)
        ws.cell(i, 4, note)
        ws.cell(i, 2).number_format = "0.000"
        ws.cell(i, 3).number_format = "+0.000;-0.000;0.000"

        delta_cell = ws.cell(i, 3)
        fill = _delta_static_fill(d)
        if fill is not None:
            delta_cell.fill = fill

    # Highlight best scenario (max alpha)
    alphas = [(r, ws.cell(r, 2).value) for r in range(4, 8)]
    best_row = None
    best_val = None
    for r, v in alphas:
        if isinstance(v, (int, float)) and (best_val is None or v > best_val):
            best_val = v
            best_row = r
    if best_row is not None:
        for c in range(1, 5):
            ws.cell(best_row, c).font = Font(bold=True)


    # --- Stage 3: human-readable explanations + UX highlighting ---
    # Conditional formatting for Δ vs original (green/yellow/red)
    green_fill = PatternFill("solid", fgColor="C6EFCE")
    yellow_fill = PatternFill("solid", fgColor="FFF2CC")
    red_fill = PatternFill("solid", fgColor="F8CBAD")

    # delta range is C4:C7
    delta_range = "C4:C7"
    ws.conditional_formatting.add(delta_range, CellIsRule(operator="greaterThan", formula=["0.005"], fill=green_fill))
    ws.conditional_formatting.add(delta_range, CellIsRule(operator="between", formula=["-0.005", "0.005"], fill=yellow_fill))
    ws.conditional_formatting.add(delta_range, CellIsRule(operator="lessThan", formula=["-0.005"], fill=red_fill))

    # Build compact narrative notes per scenario
    def _fmt_delta(dv):
        return "" if dv is None else f"{dv:+.3f}"

    # Append a short explanation to Notes (col D)
    for r in range(4, 8):
        name = ws.cell(r, 1).value or ""
        d = ws.cell(r, 3).value
        if name == "After reverse only":
            ws.cell(r, 4).value = f"Reversing items [{', '.join(map(str, rev_items)) if rev_items else '—'}] changes α by {_fmt_delta(d)}."
        elif name == "After remove only":
            ws.cell(r, 4).value = f"Removing items [{', '.join(map(str, rem_items)) if rem_items else '—'}] changes α by {_fmt_delta(d)} (rule: remove_delta={remove_delta})."
        elif name == "After reverse + remove":
            ws.cell(r, 4).value = f"Reverse then remove. Δα {_fmt_delta(d)}. Reverse [{', '.join(map(str, rev_items)) if rev_items else '—'}], remove [{', '.join(map(str, rem_items)) if rem_items else '—'}]."

    # Explanations section (per-item deltas if available)
    items = diag.get("items") or []
    if isinstance(items, list) and items:
        rr = 15
        ws[f"A{rr}"] = "Explanations"
        _summary_style(ws[f"A{rr}"])
        ws.merge_cells(f"A{rr}:D{rr}")
        rr += 1

        # Removing item X increases alpha by +...
        for it in items:
            nm = it.get("item_name") or it.get("name") or it.get("item") or None
            d_del = it.get("delta_alpha_if_deleted")
            if nm is None or d_del is None:
                continue
            if not rem_items or str(nm) in set(map(str, rem_items)):
                ws.cell(rr, 1, f"Removing '{nm}' changes α by {float(d_del):+0.3f}.")
                ws.merge_cells(f"A{rr}:D{rr}")
                rr += 1

    # Recommendations
    ws["A10"] = "Recommendations"
    _summary_style(ws["A10"])
    ws.merge_cells("A10:D10")

    ws["A11"] = "Reverse items"
    ws["B11"] = ", ".join(map(str, rev_items)) if rev_items else "—"
    _kv_style(ws["A11"], ws["B11"])
    ws["A12"] = "Remove items"
    ws["B12"] = ", ".join(map(str, rem_items)) if rem_items else "—"
    _kv_style(ws["A12"], ws["B12"])
    ws["A13"] = "remove_delta rule"
    ws["B13"] = remove_delta if remove_delta is not None else ""
    _kv_style(ws["A13"], ws["B13"])
    ws["B13"].number_format = "0.000"

    # Keep widths
    _set_col_width(ws, 1, 26)
    _set_col_width(ws, 2, 22)
    _set_col_width(ws, 3, 14)
    _set_col_width(ws, 4, 34)

    ws.freeze_panes = "A4"


def _add_ai_sheet(wb: Workbook, report: Dict[str, Any]) -> None:
    """Add a simple sheet with AI interpretation text (if present in report)."""
    ai = report.get("ai") if isinstance(report, dict) else None
    if not isinstance(ai, dict):
        return

    text = ai.get("text")
    err = ai.get("error")
    has_text = isinstance(text, str) and bool(text.strip())
    has_err = isinstance(err, str) and bool(err.strip())
    if not (has_text or has_err):
        return

    ws = wb.create_sheet("AI interpretation")
    _set_col_width(ws, 1, 120)

    ws["A1"] = "AI interpretation (plain-language)"
    _header_style(ws["A1"])
    ws.row_dimensions[1].height = 18

    meta = []
    provider = ai.get("provider")
    model = ai.get("model")
    if provider:
        meta.append(f"provider={provider}")
    if model:
        meta.append(f"model={model}")
    if meta:
        ws["A2"] = "; ".join(meta)
        ws["A2"].font = Font(italic=True)

    if has_text:
        ws["A4"] = text.strip()
    else:
        # Debug-friendly: show why AI is missing (e.g., 401/404/429).
        status = ai.get("status") or "failed"
        ws["A4"] = f"AI status: {status}\n\n{err}".strip()
    ws["A4"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[4].height = 240

def _add_cronbach_alpha_if_deleted_sheet(wb: Workbook, report: Dict[str, Any]) -> None:
    diag = _extract_cronbach_diagnostics(report)
    if not isinstance(diag, dict):
        return
    items = diag.get("items", []) or []
    if not isinstance(items, list) or not items:
        return

    ws = wb.create_sheet("Alpha if deleted")

    headers = ["item_name", "alpha_if_deleted", "Δ alpha", "remove_candidate"]
    widths = [26, 14, 12, 16]
    for i, w in enumerate(widths, start=1):
        _set_col_width(ws, i, w)

    ws["A1"] = "Cronbach α — alpha if deleted (sorted by Δ)"
    _summary_style(ws["A1"])
    ws.merge_cells("A1:D1")
    ws.row_dimensions[1].height = 22

    # header row
    for c, h in enumerate(headers, start=1):
        _header_style(ws.cell(3, c, h))

    # sort by delta desc
    rows = []
    for it in items:
        rows.append((
            it.get("item_name"),
            it.get("alpha_if_deleted"),
            it.get("delta_alpha_if_deleted"),
            it.get("remove_candidate"),
        ))
    rows.sort(key=lambda t: (t[2] if isinstance(t[2], (int, float)) else -1e18), reverse=True)

    for r, (name, a, d, rem) in enumerate(rows, start=4):
        ws.cell(r, 1, name)
        ws.cell(r, 2, a)
        ws.cell(r, 3, d)
        ws.cell(r, 4, rem)
        ws.cell(r, 2).number_format = "0.000"
        ws.cell(r, 3).number_format = "+0.000;-0.000;0.000"

    last_row = 3 + len(rows)

    ws.auto_filter.ref = f"A3:D{last_row}"
    ws.freeze_panes = "A4"

    red = PatternFill("solid", fgColor="F8CBAD")
    green = PatternFill("solid", fgColor="C6EFCE")
    yellow = PatternFill("solid", fgColor="FFF2CC")

    # Δ > 0 green
    ws.conditional_formatting.add(f"C4:C{max(last_row,4)}", CellIsRule(operator="greaterThan", formula=["0"], fill=green))
    # remove_candidate TRUE yellow
    ws.conditional_formatting.add(f"D4:D{max(last_row,4)}", CellIsRule(operator="equal", formula=["TRUE"], fill=yellow))


def export_report_to_excel(report: Dict[str, Any], out_path: str | Path) -> str:
    """
    Export already computed compare_reliability_csv JSON report to XLSX.
    Returns str path to saved file.
    """
    out_path = str(out_path)
    wb = Workbook()

    # ----------------- Summary -----------------
    ws = wb.active
    ws.title = "Summary"

    _set_col_width(ws, 1, 28)
    _set_col_width(ws, 2, 60)
    ws.row_dimensions[1].height = 22

    ws["A1"] = "PsyAdapt Reliability Report"
    ws["A1"].font = Font(bold=True, size=14)
    ws.merge_cells("A1:B1")

    meta = report.get("meta", {})
    inputs = report.get("inputs", {})
    summary = report.get("summary", {})
    status = report.get("status", "ok")

    rows = [
        ("Status", status),
        ("Generated (UTC)", meta.get("generated_at_utc")),
        ("Schema version", meta.get("schema_version")),
        ("Files", ", ".join(inputs.get("files", []))),
        ("n_files", inputs.get("n_files")),
        ("rating_scale", inputs.get("rating_scale")),
        ("aggregate", inputs.get("aggregate")),
        ("Primary estimator", summary.get("primary_estimator")),
        ("Primary score", summary.get("primary_score")),
        ("ICC estimator", summary.get("icc_estimator")),
        ("ICC score", summary.get("icc_score")),
        ("n_raters", summary.get("n_raters")),
        ("n_objects", summary.get("n_objects")),
        ("n_items", summary.get("n_items")),
        ("ICC model", (inputs.get("icc") or {}).get("model")),
        ("ICC type", (inputs.get("icc") or {}).get("icc_type")),
        ("ICC unit", (inputs.get("icc") or {}).get("unit")),
    ]

    start_row = 3
    for i, (k, v) in enumerate(rows, start=start_row):
        ws.cell(i, 1, k)
        ws.cell(i, 2, "" if v is None else v)
        _kv_style(ws.cell(i, 1), ws.cell(i, 2))

    # ----------------- Per-item -----------------
    ws_items = wb.create_sheet("Per-item")
    headers = [
        "item_name",
        "item_index",
        "primary.estimator",
        "primary.kappa",
        "primary.alpha",
        "primary.n_pairs",
        "icc.icc",
        "icc.n_targets_used",
    ]
    for c, h in enumerate(headers, start=1):
        cell = ws_items.cell(1, c, h)
        _header_style(cell)

    _set_col_width(ws_items, 1, 28)
    _set_col_width(ws_items, 2, 12)
    _set_col_width(ws_items, 3, 20)
    _set_col_width(ws_items, 4, 14)
    _set_col_width(ws_items, 5, 14)
    _set_col_width(ws_items, 6, 14)
    _set_col_width(ws_items, 7, 12)
    _set_col_width(ws_items, 8, 16)

    per_item = report.get("per_item", []) or []
    for r, row in enumerate(per_item, start=2):
        primary = row.get("primary", {}) or {}
        icc = row.get("icc", {}) or {}

        name = row.get("item_name") or row.get("name")
        if not name:
            idx = row.get("item_index")
            name = f"Q{int(idx)+1}" if idx is not None else ""
        ws_items.cell(r, 1, name)
        ws_items.cell(r, 2, row.get("item_index"))
        ws_items.cell(r, 3, primary.get("estimator"))
        ws_items.cell(r, 4, primary.get("kappa"))
        ws_items.cell(r, 5, primary.get("alpha"))
        ws_items.cell(r, 6, primary.get("n_pairs"))
        ws_items.cell(r, 7, icc.get("icc"))
        ws_items.cell(r, 8, icc.get("n_targets_used"))

    # --------- Auto-filter ----------
    # Apply filter to the full used range (header + all rows)
    last_row = 1 + len(per_item)
    last_col = len(headers)
    ws_items.auto_filter.ref = f"A1:{get_column_letter(last_col)}{max(last_row, 1)}"

    # --------- Conditional formatting ----------
    # Thresholds (can be tuned later)
    # Common practical interpretation:
    # < 0.50 low, 0.50–0.75 moderate, >=0.75 good
    low_fill = PatternFill("solid", fgColor="F8CBAD")   # light red
    mid_fill = PatternFill("solid", fgColor="FFF2CC")   # light yellow

    # Helper to apply formatting to a whole column (from row 2 to last row)
    def _cf_range(col_letter: str) -> str:
        return f"{col_letter}2:{col_letter}{max(last_row, 2)}"

    # Column indices:
    # 4 = primary.kappa, 5 = primary.alpha, 7 = icc.icc
    col_kappa = get_column_letter(4)
    col_alpha = get_column_letter(5)
    col_icc = get_column_letter(7)

    # Apply for kappa
    ws_items.conditional_formatting.add(
        _cf_range(col_kappa),
        CellIsRule(operator="lessThan", formula=["0.5"], fill=low_fill),
    )
    ws_items.conditional_formatting.add(
        _cf_range(col_kappa),
        CellIsRule(operator="between", formula=["0.5", "0.75"], fill=mid_fill),
    )

    # Apply for alpha (Krippendorff) when present
    ws_items.conditional_formatting.add(
        _cf_range(col_alpha),
        CellIsRule(operator="lessThan", formula=["0.5"], fill=low_fill),
    )
    ws_items.conditional_formatting.add(
        _cf_range(col_alpha),
        CellIsRule(operator="between", formula=["0.5", "0.75"], fill=mid_fill),
    )

    # Apply for ICC
    ws_items.conditional_formatting.add(
        _cf_range(col_icc),
        CellIsRule(operator="lessThan", formula=["0.5"], fill=low_fill),
    )
    ws_items.conditional_formatting.add(
        _cf_range(col_icc),
        CellIsRule(operator="between", formula=["0.5", "0.75"], fill=mid_fill),
    )
    # --------- Summary rows (min/mean/max) ----------
    # Data rows are from 2..last_row
    data_first = 2
    data_last = max(last_row, 2)

    # Column letters
    col_kappa = get_column_letter(4)  # primary.kappa
    col_alpha = get_column_letter(5)  # primary.alpha
    col_icc = get_column_letter(7)    # icc.icc

    # Place summary block starting two rows after the last data row
    s0 = data_last + 2  # header row for summary block
    ws_items.cell(s0, 1, "Summary (over items)")
    ws_items.merge_cells(start_row=s0, start_column=1, end_row=s0, end_column=last_col)
    _summary_style(ws_items.cell(s0, 1))
    ws_items.row_dimensions[s0].height = 18

    # Labels and formulas
    rows = [
        ("MIN", "MIN"),
        ("MEAN", "AVERAGE"),
        ("MAX", "MAX"),
    ]

    # Write the 3 summary lines
    for idx, (label, func) in enumerate(rows, start=1):
        r = s0 + idx
        ws_items.cell(r, 1, label)
        _summary_style(ws_items.cell(r, 1))

        # Put formulas in the numeric columns
        ws_items.cell(r, 4, f"={func}({col_kappa}{data_first}:{col_kappa}{data_last})")
        ws_items.cell(r, 5, f"={func}({col_alpha}{data_first}:{col_alpha}{data_last})")
        ws_items.cell(r, 7, f"={func}({col_icc}{data_first}:{col_icc}{data_last})")

        _summary_style(ws_items.cell(r, 4))
        _summary_style(ws_items.cell(r, 5))
        _summary_style(ws_items.cell(r, 7))

        # Optional: keep other columns empty but styled for readability
        for c in (2, 3, 6, 8):
            _summary_style(ws_items.cell(r, c))

    ws_items.freeze_panes = "A2"
    ws_items.sheet_view.showGridLines = True

    # ----------------- Warnings -----------------
    ws_warn = wb.create_sheet("Warnings")
    _set_col_width(ws_warn, 1, 120)

    ws_warn.cell(1, 1, "Warnings / Notes")
    _header_style(ws_warn.cell(1, 1))

    warnings = report.get("warnings", []) or []
    # warnings may be list[str] OR list[dict] depending on how you store validation warnings
    for i, w in enumerate(warnings, start=2):
        if isinstance(w, dict):
            msg = f'{w.get("code")}: {w.get("message")}'
        else:
            msg = str(w)
        ws_warn.cell(i, 1, msg)

    # ----------------- AI interpretation (optional) -----------------
    _add_ai_sheet(wb, report)

    # ----------------- Raw JSON -----------------
    ws_raw = wb.create_sheet("Raw JSON")
    _set_col_width(ws_raw, 1, 120)
    ws_raw.cell(1, 1, "report_json")
    _header_style(ws_raw.cell(1, 1))

    raw = json.dumps(report, ensure_ascii=False, indent=2)
    ws_raw.cell(2, 1, raw)
    ws_raw.cell(2, 1).alignment = Alignment(wrap_text=True, vertical="top")

    # Save
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    _add_cronbach_diagnostics_sheet(wb, report)
    _add_cronbach_cleaning_sheet(wb, report)
    _add_cronbach_alpha_if_deleted_sheet(wb, report)
    wb.save(out_path)
    return out_path


def compare_reliability_to_excel(
    paths: Iterable[str],
    *,
    out_path: str | Path,
    rating_scale: RatingScale = "nominal",
    aggregate: Agg = "mean",
    icc_model: Literal["oneway", "twoway"] = "twoway",
    icc_type: Literal["agreement", "consistency"] = "agreement",
    icc_unit: Literal["single", "average"] = "single",
    primary: Literal["auto", "cronbach", "cohen", "fleiss", "krippendorff"] = "auto",
    ai: bool = False,
    ai_provider: str = "openai",
    ai_model: str = "o4-mini",
    ai_api_key: str | None = None,
    ai_timeout_s: float = 30.0,
    ai_debug: bool = False,
) -> Dict[str, Any]:
    """
    Convenience wrapper:
      CSV(s) -> compare_reliability_csv JSON -> export to Excel
    Returns report JSON (so you can also return it via API).
    """
    report = compare_reliability_csv(
        paths,
        rating_scale=rating_scale,
        aggregate=aggregate,
        icc_model=icc_model,
        icc_type=icc_type,
        icc_unit=icc_unit,
        primary=primary,
        ai=ai,
        ai_provider=ai_provider,
        ai_model=ai_model,
        ai_api_key=ai_api_key,
        ai_timeout_s=ai_timeout_s,
        ai_debug=ai_debug,
    )
    export_report_to_excel(report, out_path=out_path)
    return report