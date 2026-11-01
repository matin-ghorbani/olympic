from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import MetricName, StrategyName, TSConfig
from .model import XGBForecaster


def _score(y_true: np.ndarray, y_pred: np.ndarray, metric: MetricName) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if metric == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if metric == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if metric == "rmsle":
        return float(np.sqrt(mean_squared_error(np.log1p(np.clip(y_true, 0, None)), np.log1p(np.clip(y_pred, 0, None)))))
    if metric == "mape":
        denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
        return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)
    raise ValueError(f"unknown metric: {metric}")


def walk_forward_validate(
    df: pd.DataFrame,
    config: TSConfig,
    *,
    strategies: list[str] | None = None,
) -> dict[str, Any]:
    """Expanding-window walk-forward CV with optional gap and per-horizon errors.

    For each split ``i`` in ``1..n_splits``:
      train = [: cut_i]
      val   = [cut_i + gap : cut_i + gap + horizon]
    """
    strategies: list[StrategyName] = list(strategies or [config.strategy])  # type: ignore[arg-type]
    frame = df.copy()
    frame[config.date_col] = pd.to_datetime(frame[config.date_col])
    frame = frame.sort_values(config.date_col).reset_index(drop=True)

    n = len(frame)
    h = config.horizon
    gap = config.gap
    n_splits = config.n_splits

    # Need room for n_splits validation windows of size h after a growing train
    min_train = max(max(config.lags, default=1) + max(config.rolling_windows, default=1) + 5, 30)
    while n_splits > 1 and n < min_train + n_splits * (gap + h):
        n_splits -= 1
    if n < min_train + gap + h:
        raise ValueError(
            f"series too short for walk-forward (n={n}, need>={min_train + gap + h})"
        )

    results: dict[str, Any] = {}
    for strategy in strategies:
        fold_scores: list[float] = []
        per_h_errors: list[list[float]] = [[] for _ in range(h)]
        fold_rows: list[dict[str, Any]] = []

        for split_i in range(n_splits):
            # Place validation windows at the end, expanding train before each
            val_end = n - (n_splits - split_i - 1) * (gap + h)
            val_start = val_end - h
            train_end = val_start - gap
            if train_end < min_train or val_start < 0:
                continue

            train = frame.iloc[:train_end].copy()
            future = frame.iloc[val_start:val_end].copy()
            y_true = pd.to_numeric(future[config.target_col], errors="coerce").to_numpy()
            if np.isnan(y_true).all():
                continue

            cfg = config.with_updates(strategy=strategy)

            model = XGBForecaster(cfg)
            model.fit(train, cfg)
            pred_df = model.predict(train, future.drop(columns=[config.target_col], errors="ignore"), cfg)
            y_pred = pred_df["prediction"].to_numpy()

            valid = ~np.isnan(y_true)
            score = _score(y_true[valid], y_pred[valid], cfg.metric)
            fold_scores.append(score)
            for j in range(min(h, len(y_true))):
                if valid[j]:
                    per_h_errors[j].append(abs(float(y_true[j] - y_pred[j])))

            fold_rows.append(
                {
                    "strategy": strategy,
                    "fold": split_i,
                    "train_end": int(train_end),
                    "val_start": int(val_start),
                    "val_end": int(val_end),
                    "score": score,
                }
            )

        per_horizon = pd.DataFrame(
            {
                "horizon_step": np.arange(1, h + 1),
                "mae": [float(np.mean(e)) if e else np.nan for e in per_h_errors],
            }
        )
        results[strategy] = {
            "metric": config.metric,
            "mean_score": float(np.mean(fold_scores)) if fold_scores else np.nan,
            "std_score": float(np.std(fold_scores)) if fold_scores else np.nan,
            "folds": pd.DataFrame(fold_rows),
            "per_horizon": per_horizon,
        }

    return results


def summarize_cv(results: dict[str, Any]) -> pd.DataFrame:
    """Flat summary table across strategies."""
    rows = []
    for strategy, payload in results.items():
        rows.append(
            {
                "strategy": strategy,
                "metric": payload["metric"],
                "mean_score": payload["mean_score"],
                "std_score": payload["std_score"],
            }
        )
    return pd.DataFrame(rows)
