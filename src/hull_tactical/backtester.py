"""Backtesting helpers: mapping predictions to allocations, computing realized returns,
turnover, and penalized Sharpe.

All functions operate on numpy arrays and return numeric metrics suitable for fold-level
aggregation.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from math import sqrt


def map_preds_to_alloc(preds: np.ndarray, method: str = 'tanh', clip=(0.0, 2.0), scale: float = 1.0) -> np.ndarray:
    """Map raw predictions to allocations in [clip[0], clip[1]].

    Methods:
    - 'tanh': allocation = mid + width * tanh(scale * z) where z=(pred-mean)/std
    - 'quantile': map preds to empirical quantiles and linearly to [low, high]
    """
    preds = np.asarray(preds).astype(float)
    if method == 'tanh':
        mean = np.nanmean(preds)
        std = np.nanstd(preds)
        if std == 0 or np.isnan(std):
            z = np.zeros_like(preds)
        else:
            z = (preds - mean) / std
        mid = (clip[0] + clip[1]) / 2.0
        width = (clip[1] - clip[0]) / 2.0
        alloc = mid + width * np.tanh(scale * z)
    elif method == 'quantile':
        q = pd.Series(preds).rank(pct=True).values
        alloc = clip[0] + q * (clip[1] - clip[0])
    else:
        raise ValueError('unknown method')
    alloc = np.clip(alloc, clip[0], clip[1])
    return alloc


def volatility_targeting(alloc: np.ndarray, market_returns: np.ndarray, target_vol: float) -> np.ndarray:
    """Scale allocations so that realized annualized volatility of strategy matches target_vol.

    If realized_vol is zero or NaN, returns the original allocations.
    """
    alloc = np.asarray(alloc, dtype=float)
    market_returns = np.asarray(market_returns, dtype=float)
    strategy_returns = alloc * market_returns
    realized_vol = np.nanstd(strategy_returns) * np.sqrt(252)
    if realized_vol == 0 or np.isnan(realized_vol):
        return alloc
    scale = target_vol / realized_vol
    alloc_scaled = alloc * scale
    return alloc_scaled


def compute_returns_and_metrics(alloc: np.ndarray, market_returns: np.ndarray, rf: Optional[np.ndarray] = None, turnover_penalty: float = 0.0) -> dict:
    """Compute realized excess returns, annualized Sharpe, turnover, and penalized Sharpe.

    Parameters
    - alloc: numpy array of allocations per time step
    - market_returns: numpy array of realized excess returns per time step
    - rf: optional risk-free rate series (same length)
    - turnover_penalty: lambda penalty multiplied by mean absolute turnover
    """
    alloc = np.asarray(alloc, dtype=float)
    market_returns = np.asarray(market_returns, dtype=float)
    if rf is None:
        excess = market_returns
    else:
        excess = market_returns - np.asarray(rf, dtype=float)

    strat_returns = alloc * excess
    mean = np.nanmean(strat_returns)
    std = np.nanstd(strat_returns)
    sharpe = 0.0
    if std > 0:
        sharpe = (mean / std) * np.sqrt(252)

    # turnover: mean absolute change in allocation
    if len(alloc) <= 1:
        turnover = 0.0
    else:
        turnover = np.nanmean(np.abs(np.diff(alloc)))

    penalized = sharpe - turnover_penalty * turnover

    return {
        'mean_return': float(mean),
        'std_return': float(std),
        'realized_annual_vol': float(std * np.sqrt(252)),
        'sharpe': float(sharpe),
        'turnover': float(turnover),
        'penalized_sharpe': float(penalized),
    }


def _ema(arr: np.ndarray, alpha: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    out = np.empty_like(arr)
    if len(arr) == 0:
        return out
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def dynamic_volatility_targeting(
    alloc: np.ndarray,
    predicted_vol_daily: np.ndarray,
    target_annual_vol: float,
    scale_clip: tuple[float, float] = (0.25, 4.0),
    smoothing_alpha: Optional[float] = 0.2,
    eps: float = 1e-6,
    max_daily_scale_change: Optional[float] = None,
) -> np.ndarray:
    """Scale allocations day-by-day using predicted next-day volatility.

    - predicted_vol_daily: per-step forecast of daily volatility (std of market returns), non-negative.
    - target_annual_vol: annualized target volatility (e.g., 0.12).
    - smoothing_alpha: optional EMA smoothing factor applied to predicted volatility to reduce turnover.
    - scale_clip: bounds on the per-day scale factor to avoid extreme sizing.
    """
    alloc = np.asarray(alloc, dtype=float)
    pred = np.asarray(predicted_vol_daily, dtype=float)
    # optional smoothing
    if smoothing_alpha is not None:
        pred = _ema(pred, smoothing_alpha)
    pred = np.maximum(pred, eps)
    target_daily = target_annual_vol / sqrt(252.0)
    scales = target_daily / pred
    scales = np.clip(scales, scale_clip[0], scale_clip[1])
    # Optional cap on day-over-day scale change (ratio-based): +/- max_daily_scale_change
    if max_daily_scale_change is not None and len(scales) > 1:
        m = float(max_daily_scale_change)
        for t in range(1, len(scales)):
            prev = scales[t - 1]
            lo = prev * (1.0 - m)
            hi = prev * (1.0 + m)
            if scales[t] < lo:
                scales[t] = lo
            elif scales[t] > hi:
                scales[t] = hi
    alloc_scaled = alloc * scales
    return alloc_scaled


def cap_allocation_daily_change(alloc: np.ndarray, max_abs_change: float) -> np.ndarray:
    """Cap day-over-day allocation change to +/- max_abs_change (absolute units in allocation).

    Example: if max_abs_change=0.1 and alloc[t-1]=1.2, then alloc[t] will be clamped to [1.1, 1.3].
    """
    a = np.asarray(alloc, dtype=float).copy()
    if len(a) <= 1:
        return a
    for t in range(1, len(a)):
        lo = a[t - 1] - max_abs_change
        hi = a[t - 1] + max_abs_change
        if a[t] < lo:
            a[t] = lo
        elif a[t] > hi:
            a[t] = hi
    return a
