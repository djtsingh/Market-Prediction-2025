"""Feature engineering helpers: lags, rolling windows, group prefix aggregations."""
from __future__ import annotations

from typing import Iterable, List, Sequence
import polars as pl
import numpy as np


def _numeric_cols(df: pl.DataFrame, exclude: Sequence[str] = ("row_id", "time_id", "date_id", "date", "market_forward_excess_returns")) -> List[str]:
    numeric_kinds = ('Int', 'Float', 'Decimal')
    out = []
    for c, t in zip(df.columns, df.dtypes):
        if c in exclude:
            continue
        tstr = str(t)
        if any(k in tstr for k in numeric_kinds):
            out.append(c)
    return out


def make_lag_features(df: pl.DataFrame, cols: Iterable[str], lags: Sequence[int]) -> pl.DataFrame:
    out = df
    for c in cols:
        for l in lags:
            out = out.with_columns(pl.col(c).shift(l).alias(f"{c}_lag{l}"))
    return out


def make_rolling_features(df: pl.DataFrame, cols: Iterable[str], windows: Sequence[int], aggs=("mean", "std")) -> pl.DataFrame:
    out = df
    for c in cols:
        for w in windows:
            for agg in aggs:
                if agg == 'mean':
                    out = out.with_columns(pl.col(c).rolling_mean(window_size=w).alias(f"{c}_r{w}_mean"))
                elif agg == 'std':
                    out = out.with_columns(pl.col(c).rolling_std(window_size=w).alias(f"{c}_r{w}_std"))
                elif agg == 'sum':
                    out = out.with_columns(pl.col(c).rolling_sum(window_size=w).alias(f"{c}_r{w}_sum"))
                else:
                    raise ValueError(f"Unknown agg {agg}")
    return out


def prefix_group_aggregations(df: pl.DataFrame, prefixes: Iterable[str], windows: Sequence[int]) -> pl.DataFrame:
    """For each prefix, compute group-level mean/std across features with that prefix over rolling windows.
    Adds columns like `G_{prefix}_r{w}_mean`.
    """
    out = df
    cols = df.columns
    from functools import reduce
    import operator
    for p in prefixes:
        group_cols = [c for c in cols if c.startswith(p)]
        # keep only numeric group columns
        numeric_kinds = ('Int', 'Float', 'Decimal')
        group_cols = [c for c in group_cols if any(k in str(out.schema.get(c)) for k in numeric_kinds)]
        if not group_cols:
            continue
        # compute arithmetic mean across the group's columns per row
        expr_sum = reduce(operator.add, [pl.col(c) for c in group_cols])
        expr = expr_sum / len(group_cols)
        out = out.with_columns(expr.alias(f'G_{p}_mean'))
        for w in windows:
            out = out.with_columns(pl.col(f'G_{p}_mean').rolling_mean(window_size=w).alias(f'G_{p}_r{w}_mean'))
            out = out.with_columns(pl.col(f'G_{p}_mean').rolling_std(window_size=w).alias(f'G_{p}_r{w}_std'))
    return out


def build_features(parquet_path: str, out_path: str, sample_csv: str | None = None):
    df = pl.read_parquet(parquet_path)
    num_cols = _numeric_cols(df)
    # safe defaults
    lags = [1, 2, 3]
    windows = [5, 21]

    df2 = make_lag_features(df, num_cols, lags)
    df2 = make_rolling_features(df2, num_cols, windows, aggs=('mean','std'))
    prefixes = ['M','E','P','V','S','MOM','D','I']
    df2 = prefix_group_aggregations(df2, prefixes, windows)

    # write parquet
    df2.write_parquet(out_path)
    if sample_csv:
        # save a small sampled CSV for quick inspection
        df2.head(500).to_pandas().to_csv(sample_csv, index=False)
    return out_path
