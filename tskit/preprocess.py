from __future__ import annotations

import warnings

import pandas as pd

from .config import TSConfig


def prepare(df: pd.DataFrame, config: TSConfig) -> pd.DataFrame:
    """Sort by date, reindex to regular ``freq``, apply NaN policy.

    Target NaNs are **kept** so lag/rolling stay calendar-aligned; supervised
    builders drop incomplete rows at train time. Exogenous cols get limited
    ffill/bfill.
    """
    if config.date_col not in df.columns:
        raise KeyError(f"date_col '{config.date_col}' not in columns: {list(df.columns)}")

    out = df.copy()
    out[config.date_col] = pd.to_datetime(out[config.date_col])
    out = out.sort_values(config.date_col).drop_duplicates(config.date_col, keep="last")
    out = out.set_index(config.date_col)

    full_idx = pd.date_range(out.index.min(), out.index.max(), freq=config.freq)
    n_missing = int(len(full_idx) - len(out.index.intersection(full_idx)))
    if n_missing > 0:
        warnings.warn(
            f"reindex({config.freq}): inserted {n_missing} missing timestamps "
            "(target NaNs kept for calendar-aligned lags)",
            stacklevel=2,
        )
    out = out.reindex(full_idx)
    out.index.name = config.date_col

    for col in config.exog_cols:
        if col not in out.columns:
            raise KeyError(f"exog col '{col}' missing after reindex")
        out[col] = out[col].ffill(limit=config.ffill_limit).bfill(limit=config.ffill_limit)

    if config.target_col in out.columns:
        out[config.target_col] = pd.to_numeric(out[config.target_col], errors="coerce")
        n_na = int(out[config.target_col].isna().sum())
        if n_na:
            warnings.warn(
                f"{n_na} target NaNs retained after reindex (dropped only when building X/y)",
                stacklevel=2,
            )

    return out.reset_index()


def prepare_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: TSConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare train and future/test frames separately."""
    train_p = prepare(train, config)

    test_p = test.copy()
    test_p[config.date_col] = pd.to_datetime(test_p[config.date_col])
    cols = [config.date_col, *config.exog_cols]
    missing = [c for c in cols if c not in test_p.columns]
    if missing:
        raise KeyError(f"test is missing required columns: {missing}")

    test_only = test_p[cols + ([config.target_col] if config.target_col in test_p.columns else [])]
    test_prep = prepare(test_only, config)
    return train_p, test_prep
