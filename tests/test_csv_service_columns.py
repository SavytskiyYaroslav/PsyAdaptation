from __future__ import annotations

from pathlib import Path

import pandas as pd

from psyadapt.io.csv import load_csv_dataframe, load_csv_matrix
from psyadapt.validate import validate_single_test_csv


def test_csv_id_column_is_dropped(tmp_path: Path):
    df = pd.DataFrame(
        {
            "id": ["001", "002", "003"],
            "Q1": [1, 2, 3],
            "Q2": [4, 5, 6],
        }
    )
    p = tmp_path / "t.csv"
    df.to_csv(p, index=False, encoding="utf-8")

    # Validation should ignore id column, so it must not fail as non-numeric
    v = validate_single_test_csv(str(p), allow_missing=False)
    assert v["ok"] is True

    df_loaded = load_csv_dataframe(str(p))
    assert list(df_loaded.columns) == ["Q1", "Q2"]

    X = load_csv_matrix(str(p))
    assert X.shape == (3, 2)
