"""Data loading utilities using Polars for performance."""
from __future__ import annotations

import os
import polars as pl
from typing import Tuple

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

def read_csv_polars(path: str, n_rows: int | None = None) -> pl.DataFrame:
    return pl.read_csv(path, n_rows=n_rows)

def load_train(path: str = None, n_rows: int | None = None) -> pl.DataFrame:
    """Load train.csv as a Polars DataFrame.

    Args:
        path: optional absolute path override. If None, looks in project root for train.csv.
    """
    if path is None:
        path = os.path.join(DATA_DIR, "data", "raw", "train.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"train.csv not found at {path}")
    df = read_csv_polars(path, n_rows=n_rows)
    return df

def load_test(path: str = None, n_rows: int | None = None) -> pl.DataFrame:
    if path is None:
        path = os.path.join(DATA_DIR, "data", "raw", "test.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"test.csv not found at {path}")
    return read_csv_polars(path, n_rows=n_rows)

def dataset_overview(df: pl.DataFrame) -> dict:
    """Return basic stats: columns, dtypes, null counts, rows."""
    cols = df.columns
    dtypes = {c: str(df[c].dtype) for c in cols}
    nulls = {c: int(df[c].null_count()) for c in cols}
    return {"rows": df.height, "columns": cols, "dtypes": dtypes, "nulls": nulls}
