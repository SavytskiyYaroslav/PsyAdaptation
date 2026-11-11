"""
psyadapt - Psychometric reliability metrics with sklearn-style API.
"""

from __future__ import annotations

from __future__ import annotations

from .api import (
    compare_reliability_csv,
    compare_reliability_to_excel,
    export_report_to_excel,
    workbook_vertical_to_matrix,
    workbook_vertical_to_csv,
    validate_single_test_csv,
    validate_raters_csvs,
)

__all__ = [
    "compare_reliability_csv",
    "compare_reliability_to_excel",
    "export_report_to_excel",
    "workbook_vertical_to_matrix",
    "workbook_vertical_to_csv",
    "validate_single_test_csv",
    "validate_raters_csvs",
]

# optional: version
__version__ = "0.1.0"

