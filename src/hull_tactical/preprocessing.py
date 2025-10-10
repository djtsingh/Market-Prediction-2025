"""Preprocessing helpers: imputation, scaling, time features."""
from __future__ import annotations

import polars as pl
import numpy as np
from typing import Sequence

def simple_impute(df: pl.DataFrame, cols: Sequence[str], strategy: str = "median") -> pl.DataFrame:
    """Simple imputation using Polars expressions."""
    out = df
    for c in cols:
        if strategy == "median":
            val = float(out.select(pl.col(c).median()).to_numpy()[0])
        elif strategy == "mean":
            val = float(out.select(pl.col(c).mean()).to_numpy()[0])
        else:
            raise ValueError("unknown strategy")
        out = out.with_columns(pl.col(c).fill_null(val))
    return out

def add_time_features(df: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
    """Extract simple time features if date column present."""
    if date_col not in df.columns:
        return df
    out = df.with_columns(
        pl.col(date_col).str.strptime(pl.Date, "%Y-%m-%d").alias("_parsed_date")
    )
    out = out.with_columns(
        pl.col("_parsed_date").dt.month().alias("month"),
        pl.col("_parsed_date").dt.day().alias("day"),
        pl.col("_parsed_date").dt.weekday().alias("weekday")
    )
    return out

def scale_zscore(df: pl.DataFrame, cols: Sequence[str]) -> pl.DataFrame:
    out = df
    for c in cols:
        col_mean = float(out.select(pl.col(c).mean()).to_numpy()[0])
        col_std = float(out.select(pl.col(c).std()).to_numpy()[0])
        if col_std == 0 or np.isnan(col_std):
            out = out.with_columns(pl.lit(0).alias(c + "_z"))
        else:
            out = out.with_columns(((pl.col(c) - col_mean) / col_std).alias(c + "_z"))
    return out

def quick_test():
    import os
    from .data import load_train
    p = os.path.join(os.path.dirname(__file__), "..", "train.csv")
    df = load_train(p, n_rows=1000)
    print("Loaded rows:", df.height)
    print(dataset_overview(df))
