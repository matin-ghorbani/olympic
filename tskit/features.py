from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TSConfig


def add_calendar_features(df: pd.DataFrame, config: TSConfig) -> pd.DataFrame:
    """Cyclical calendar encodings from the date column."""
    out = df.copy()
    dt = pd.to_datetime(out[config.date_col])
    # day-of-week
    dow = dt.dt.dayofweek.astype(float)
    out["cal_dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["cal_dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    # month
    month = dt.dt.month.astype(float)
    out["cal_month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["cal_month_cos"] = np.cos(2 * np.pi * month / 12.0)
    out["cal_day"] = dt.dt.day.astype(float)
    out["cal_weekofyear"] = dt.dt.isocalendar().week.astype(float)
    return out


def add_lag_rolling_features(
    df: pd.DataFrame,
    config: TSConfig,
    *,
    y_col: str | None = None,
) -> pd.DataFrame:
    """Leakage-safe lag and rolling features on the level target series.

    Rolling stats are shifted by 1 so the window never includes the current row.
    """
    y_col = y_col or config.target_col
    out = df.copy()
    y = pd.to_numeric(out[y_col], errors="coerce")

    for lag in config.lags:
        out[f"lag_{lag}"] = y.shift(lag)

    for window in config.rolling_windows:
        rolled = y.shift(1).rolling(window=window, min_periods=max(1, window // 2))
        out[f"roll_mean_{window}"] = rolled.mean()
        out[f"roll_std_{window}"] = rolled.std()

    return out


def add_features(df: pd.DataFrame, config: TSConfig, *, y_col: str | None = None) -> pd.DataFrame:
    """Full feature pipeline: calendar + lag/rolling on level y."""
    out = add_calendar_features(df, config)
    out = add_lag_rolling_features(out, config, y_col=y_col)
    return out


def make_supervised_recursive(
    df: pd.DataFrame,
    config: TSConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build X, y_model, y_level for recursive one-step training.

    If ``target_diff``, model target is ``y_t - y_{t-1}``; ``y_level`` keeps levels
    for reconstruction.
    """
    feat = add_features(df, config)
    y_level = pd.to_numeric(feat[config.target_col], errors="coerce")
    if config.target_diff:
        y_model = y_level.diff()
    else:
        y_model = y_level.copy()

    feature_cols = _model_feature_columns(feat, config)
    X = feat[feature_cols]
    mask = y_model.notna() & X.notna().all(axis=1)
    return X.loc[mask], y_model.loc[mask], y_level.loc[mask]


def make_supervised_direct(
    df: pd.DataFrame,
    config: TSConfig,
    horizon_step: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build X/y for direct forecasting of y_{t+h} from features at time t."""
    if horizon_step < 1:
        raise ValueError("horizon_step must be >= 1")

    feat = add_features(df, config)
    y_level = pd.to_numeric(feat[config.target_col], errors="coerce")
    # Features at t predict level (or diff-to-level path) at t+h
    y_future = y_level.shift(-horizon_step)
    if config.target_diff:
        # Predict cumulative change from y_t to y_{t+h}: y_{t+h} - y_t
        y_model = y_future - y_level
    else:
        y_model = y_future

    feature_cols = _model_feature_columns(feat, config)
    X = feat[feature_cols]
    mask = y_model.notna() & y_level.notna() & X.notna().all(axis=1)
    return X.loc[mask], y_model.loc[mask], y_level.loc[mask]


def _model_feature_columns(feat: pd.DataFrame, config: TSConfig) -> list[str]:
    banned = {config.date_col, config.target_col}
    cols: list[str] = []
    for c in feat.columns:
        if c in banned:
            continue
        if c.startswith("_"):
            continue
        cols.append(c)
    return cols
