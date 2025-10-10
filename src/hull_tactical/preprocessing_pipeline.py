"""Preprocessing wrapper: fit scalers on training folds and transform feature matrices.

Provides Standard (z-score) and Robust scalers and helpers to prepare train/test arrays.
"""
from __future__ import annotations

import os
from typing import Tuple, Iterable, Union
import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib


def fit_scaler(X_train: pd.DataFrame, scaler_type: str = 'standard') -> object:
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError('unknown scaler_type')
    scaler.fit(X_train)
    return scaler


def transform_with_scaler(X: pd.DataFrame, scaler) -> pd.DataFrame:
    Xs = scaler.transform(X)
    return pd.DataFrame(Xs, index=X.index, columns=X.columns)


def prepare_fold_data(df_features: Union[pl.DataFrame, pd.DataFrame], train_idx: Iterable[int], test_idx: Iterable[int],
                      target_col: str = 'market_forward_excess_returns', scaler_type: str = 'standard') -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, object]:
    """Given Polars DataFrame and row indices, return (X_train, y_train, X_test, y_test, scaler)

    - X are pandas DataFrames, scaler is fitted on X_train.
    """
    # Accept either Polars or pandas DataFrame
    if hasattr(df_features, 'to_pandas'):
        pdf = df_features.to_pandas()
    elif isinstance(df_features, pd.DataFrame):
        pdf = df_features
    else:
        # fallback: try to construct pandas DataFrame
        pdf = pd.DataFrame(df_features)
    train_idx = list(train_idx)
    test_idx = list(test_idx)
    df_tr = pdf.iloc[train_idx].copy()
    df_te = pdf.iloc[test_idx].copy()

    if target_col not in pdf.columns:
        raise ValueError('target_col not found')

    y_tr = df_tr[target_col].astype(float)
    y_te = df_te[target_col].astype(float)

    # Drop identifiers and target, plus forward-looking columns to avoid leakage
    drop_cols = ['row_id', 'time_id', 'date_id', 'date', target_col]
    # also drop any column that starts with 'forward_'
    fwd_cols_tr = [c for c in df_tr.columns if c.startswith('forward_')]
    fwd_cols_te = [c for c in df_te.columns if c.startswith('forward_')]
    # combine
    drop_cols_tr = [c for c in drop_cols if c in df_tr.columns] + fwd_cols_tr
    drop_cols_te = [c for c in drop_cols if c in df_te.columns] + fwd_cols_te
    X_tr = df_tr.drop(columns=drop_cols_tr)
    X_te = df_te.drop(columns=drop_cols_te)

    # Guard: ensure no leakage columns remain
    forbidden = [c for c in X_tr.columns if c.startswith('forward_') or c == target_col]
    if forbidden:
        raise RuntimeError(f'Leaky columns present after drop in training features: {forbidden}')

    # Fit scaler
    scaler = fit_scaler(X_tr, scaler_type=scaler_type)
    X_tr_s = transform_with_scaler(X_tr, scaler)
    X_te_s = transform_with_scaler(X_te, scaler)

    return X_tr_s, y_tr, X_te_s, y_te, scaler


def save_scaler(scaler, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: str):
    return joblib.load(path)
