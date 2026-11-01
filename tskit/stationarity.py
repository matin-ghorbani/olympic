from __future__ import annotations

import warnings
from typing import Any

import pandas as pd

from .config import TSConfig


def _safe_adf(series: pd.Series) -> dict[str, Any]:
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError as exc:  # pragma: no cover
        raise ImportError("statsmodels is required for ADF/KPSS helpers") from exc

    clean = series.dropna()
    if len(clean) < 16:
        return {"test": "ADF", "n": len(clean), "error": "too few observations"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pvalue, usedlag, nobs, crit, _ = adfuller(clean, autolag="AIC")
    return {
        "test": "ADF",
        "stat": float(stat),
        "pvalue": float(pvalue),
        "usedlag": int(usedlag),
        "nobs": int(nobs),
        "critical": {k: float(v) for k, v in crit.items()},
        "stationary_at_5pct": bool(pvalue < 0.05),
        "null": "unit root (non-stationary)",
    }


def _safe_kpss(series: pd.Series) -> dict[str, Any]:
    try:
        from statsmodels.tsa.stattools import kpss
    except ImportError as exc:  # pragma: no cover
        raise ImportError("statsmodels is required for ADF/KPSS helpers") from exc

    clean = series.dropna()
    if len(clean) < 16:
        return {"test": "KPSS", "n": len(clean), "error": "too few observations"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pvalue, lags, crit = kpss(clean, regression="c", nlags="auto")
    return {
        "test": "KPSS",
        "stat": float(stat),
        "pvalue": float(pvalue),
        "lags": int(lags),
        "critical": {k: float(v) for k, v in crit.items()},
        # KPSS null is stationary; low p-value => reject => non-stationary
        "stationary_at_5pct": bool(pvalue >= 0.05),
        "null": "stationary",
    }


def check_stationarity(df: pd.DataFrame, config: TSConfig) -> pd.DataFrame:
    """Run ADF/KPSS on raw target and first difference. Diagnostic only."""
    y = pd.to_numeric(df[config.target_col], errors="coerce")
    y_diff = y.diff()

    rows = []
    for label, series in (("raw", y), ("diff1", y_diff)):
        adf = _safe_adf(series)
        kpss_res = _safe_kpss(series)
        rows.append({"series": label, **{f"adf_{k}": v for k, v in adf.items() if k != "test"}})
        rows[-1].update({f"kpss_{k}": v for k, v in kpss_res.items() if k != "test"})
    return pd.DataFrame(rows)
