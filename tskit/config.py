from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal, Sequence


MetricName = Literal["mae", "rmse", "rmsle", "mape"]
StrategyName = Literal["recursive", "direct"]


@dataclass
class TSConfig:
    """Contest-day knobs for univariate XGBoost forecasting."""

    date_col: str = "date"
    target_col: str = "y"
    freq: str = "D"
    horizon: int = 14
    lags: Sequence[int] = field(default_factory=lambda: (1, 7, 14))
    rolling_windows: Sequence[int] = field(default_factory=lambda: (7, 14))
    target_diff: bool = True
    strategy: StrategyName = "recursive"
    n_splits: int = 3
    gap: int = 0
    metric: MetricName = "mae"
    exog_cols: Sequence[str] = field(default_factory=tuple)
    ffill_limit: int = 3
    random_state: int = 42

    # XGBoost defaults — strong enough for a first submit
    xgb_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "n_jobs": -1,
            "tree_method": "hist",
        }
    )

    def with_updates(self, **kwargs: Any) -> "TSConfig":
        """Return a copy with selected fields overridden (deep-copies xgb_params)."""
        if "xgb_params" not in kwargs:
            kwargs = {**kwargs, "xgb_params": dict(self.xgb_params)}
        if "lags" in kwargs:
            kwargs["lags"] = tuple(kwargs["lags"])
        if "rolling_windows" in kwargs:
            kwargs["rolling_windows"] = tuple(kwargs["rolling_windows"])
        if "exog_cols" in kwargs:
            kwargs["exog_cols"] = tuple(kwargs["exog_cols"])
        return replace(self, **kwargs)

    def feature_cols(self, frame_columns: Sequence[str]) -> list[str]:
        """Return model feature columns present in a frame (excludes date/target)."""
        banned = {self.date_col, self.target_col, "_y_level", "_y_train"}
        return [c for c in frame_columns if c not in banned and not c.startswith("_")]
