"""
psyadapt Public API.

Only names exported from this module (and re-exported in psyadapt.__init__)
are considered stable public interfaces.

Everything else is internal and may change without notice.
"""

from __future__ import annotations

from psyadapt.compare import compare_reliability_csv
from psyadapt.export_excel import compare_reliability_to_excel, export_report_to_excel

# IO (Excel vertical format → matrix/csv)
from psyadapt.io.excel import workbook_vertical_to_matrix, workbook_vertical_to_csv

# Validation helpers (optional, but useful)
from psyadapt.validate import validate_single_test_csv, validate_raters_csvs

__all__ = [
    "compare_reliability_csv",
    "compare_reliability_to_excel",
    "export_report_to_excel",
    "workbook_vertical_to_matrix",
    "workbook_vertical_to_csv",
    "validate_single_test_csv",
    "validate_raters_csvs",
]
