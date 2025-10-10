"""Validation utilities: walk-forward time-series splitting."""
from __future__ import annotations

from typing import Generator, Iterable, Tuple
import numpy as np
import pandas as pd
import polars as pl


def _to_pandas(df):
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    if isinstance(df, pd.DataFrame):
        return df
    raise ValueError('df must be a Polars or Pandas DataFrame')


def walk_forward_split(
    df,
    time_col: str = 'date_id',
    initial_window: int = 1000,
    horizon: int = 250,
    step: int = 250,
    expanding: bool = True,
    n_splits: int | None = None,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Yield train/test index arrays for walk-forward backtesting.

    Contract:
    - Inputs: a DataFrame (Polars or Pandas) and parameters describing the walk-forward.
    - Yields: pairs (train_idx, test_idx) where each is a numpy array of integer row indices into the
      original DataFrame `df`.

    Parameters:
    - initial_window: number of unique time steps to use for the first training window.
    - horizon: number of unique time steps in each test fold (forecast horizon).
    - step: move the window forward by this many unique time steps per fold.
    - expanding: if True, training window grows each fold; else it slides (fixed length = initial_window).
    - n_splits: optional maximum number of splits to generate.

    Notes:
    - Splits are constructed by unique values of `time_col` (e.g., `date_id`). This ensures no leakage
      across time groups. Rows that share the same `time_col` are kept together.
    - The function is robust to Polars or Pandas inputs.
    """
    pdf = _to_pandas(df)
    if time_col not in pdf.columns:
        raise ValueError(f"time_col '{time_col}' not found in dataframe")

    # unique sorted time steps
    times = pd.Index(pdf[time_col].unique()).sort_values()
    T = len(times)
    if initial_window <= 0 or horizon <= 0:
        raise ValueError('initial_window and horizon must be > 0')

    # map time value -> integer row indices in original df
    # We keep positions as numpy integer indices
    groups = pdf.groupby(time_col).indices  # dict: time_val -> np.array(indices)

    splits_yielded = 0
    start = initial_window
    while start + horizon <= T:
        if expanding:
            train_times = times[:start]
        else:
            train_times = times[start - initial_window:start]

        test_times = times[start:start + horizon]

        # gather row indices
        train_idx = np.concatenate([groups[t] for t in train_times]) if len(train_times) > 0 else np.array([], dtype=int)
        test_idx = np.concatenate([groups[t] for t in test_times]) if len(test_times) > 0 else np.array([], dtype=int)

        yield (np.sort(train_idx), np.sort(test_idx))

        splits_yielded += 1
        if n_splits is not None and splits_yielded >= n_splits:
            break
        start += step


def example_usage(df, time_col='date_id'):
    """Return a short summary of first 3 splits for demonstration/testing."""
    gen = walk_forward_split(df, time_col=time_col, initial_window=1000, horizon=250, step=250)
    out = []
    for i, (tr, te) in enumerate(gen):
        out.append({'fold': i + 1, 'train_rows': len(tr), 'test_rows': len(te)})
        if i >= 2:
            break
    return out
