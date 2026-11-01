from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .config import TSConfig
from .features import (
    add_features,
    make_supervised_direct,
    make_supervised_recursive,
)


class XGBForecaster:
    """Univariate XGBoost forecaster with recursive/direct strategies."""

    def __init__(self, config: TSConfig | None = None):
        self.config = config or TSConfig()
        self.model_: XGBRegressor | None = None
        self.models_: dict[int, XGBRegressor] | None = None
        self.feature_names_: list[str] = []
        self.last_level_: float | None = None
        self.fitted_strategy_: str | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame, config: TSConfig | None = None) -> "XGBForecaster":
        cfg = config or self.config
        self.config = cfg
        self.fitted_strategy_ = cfg.strategy

        y_level_full = pd.to_numeric(train[cfg.target_col], errors="coerce").dropna()
        if y_level_full.empty:
            raise ValueError("train target is empty after dropping NaNs")
        self.last_level_ = float(y_level_full.iloc[-1])

        if cfg.strategy == "recursive":
            X, y, _ = make_supervised_recursive(train, cfg)
            if X.empty:
                raise ValueError("no training rows for recursive strategy")
            self.feature_names_ = list(X.columns)
            self.model_ = self._make_estimator(cfg)
            self.model_.fit(X, y)
            self.models_ = None
        elif cfg.strategy == "direct":
            self.models_ = {}
            self.feature_names_ = []
            for h in range(1, cfg.horizon + 1):
                X, y, _ = make_supervised_direct(train, cfg, h)
                if X.empty:
                    raise ValueError(f"no training rows for direct horizon h={h}")
                if not self.feature_names_:
                    self.feature_names_ = list(X.columns)
                est = self._make_estimator(cfg)
                est.fit(X[self.feature_names_], y)
                self.models_[h] = est
            self.model_ = None
        else:
            raise ValueError(f"unknown strategy: {cfg.strategy}")
        return self

    # -------------------------------------------------------------- predict
    def predict(
        self,
        history: pd.DataFrame,
        future: pd.DataFrame,
        config: TSConfig | None = None,
    ) -> pd.DataFrame:
        """Forecast rows in ``future`` using ``history`` as the known past.

        ``future`` must contain ``date_col`` and any ``exog_cols``. Target may be
        absent. Returns a frame with ``date_col`` and ``prediction``.
        """
        cfg = config or self.config
        if self.fitted_strategy_ is None:
            raise RuntimeError("Call fit() before predict()")

        future = future.copy()
        future[cfg.date_col] = pd.to_datetime(future[cfg.date_col])
        future = future.sort_values(cfg.date_col)

        if self.fitted_strategy_ == "recursive":
            preds = self._predict_recursive(history, future, cfg)
        else:
            preds = self._predict_direct(history, future, cfg)

        out = future[[cfg.date_col]].copy()
        out["prediction"] = preds
        return out.reset_index(drop=True)

    def _predict_recursive(
        self,
        history: pd.DataFrame,
        future: pd.DataFrame,
        cfg: TSConfig,
    ) -> np.ndarray:
        assert self.model_ is not None
        hist = history.copy()
        hist[cfg.date_col] = pd.to_datetime(hist[cfg.date_col])
        hist = hist.sort_values(cfg.date_col)

        # Working series of level targets; append predictions as we go
        work_rows = []
        for _, row in hist.iterrows():
            work_rows.append(row.to_dict())

        predictions: list[float] = []
        for _, frow in future.iterrows():
            work_df = pd.DataFrame(work_rows)
            # Ensure target col exists on the stub future row (unknown)
            stub = frow.to_dict()
            stub[cfg.target_col] = np.nan
            combined = pd.concat([work_df, pd.DataFrame([stub])], ignore_index=True)
            featured = add_features(combined, cfg)
            x_row = featured.iloc[[-1]][self.feature_names_]
            if x_row.isna().any(axis=None):
                # Fall back: fill remaining feature NaNs with column medians from hist features
                hist_feat = add_features(work_df, cfg)
                x_row = x_row.fillna(hist_feat[self.feature_names_].median(numeric_only=True))

            raw = float(self.model_.predict(x_row)[0])
            last_level = float(
                pd.to_numeric(work_df[cfg.target_col], errors="coerce").dropna().iloc[-1]
            )
            if cfg.target_diff:
                level_pred = last_level + raw
            else:
                level_pred = raw

            predictions.append(level_pred)
            stub[cfg.target_col] = level_pred
            work_rows.append(stub)

        return np.asarray(predictions, dtype=float)

    def _predict_direct(
        self,
        history: pd.DataFrame,
        future: pd.DataFrame,
        cfg: TSConfig,
    ) -> np.ndarray:
        assert self.models_ is not None
        hist = history.copy()
        hist[cfg.date_col] = pd.to_datetime(hist[cfg.date_col])
        hist = hist.sort_values(cfg.date_col)

        featured = add_features(hist, cfg)
        x_last = featured.iloc[[-1]][self.feature_names_]
        if x_last.isna().any(axis=None):
            x_last = x_last.fillna(featured[self.feature_names_].median(numeric_only=True))

        last_level = float(
            pd.to_numeric(hist[cfg.target_col], errors="coerce").dropna().iloc[-1]
        )

        n = len(future)
        preds = np.empty(n, dtype=float)
        for i in range(n):
            h = i + 1
            if h not in self.models_:
                # Beyond trained horizon: reuse last model
                model = self.models_[max(self.models_)]
            else:
                model = self.models_[h]
            raw = float(model.predict(x_last)[0])
            if cfg.target_diff:
                preds[i] = last_level + raw
            else:
                preds[i] = raw
        return preds

    # ---------------------------------------------------------------- tune
    def tune(
        self,
        train: pd.DataFrame,
        config: TSConfig | None = None,
        *,
        n_trials: int = 40,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Optional short HPO. Uses Optuna if installed, else RandomizedSearch."""
        cfg = deepcopy(config or self.config)
        # Tune against recursive one-step MAE on a holdout tail (fast & stable)
        X, y, _ = make_supervised_recursive(train, cfg)
        if len(X) < 50:
            raise ValueError("need at least ~50 supervised rows to tune")

        split = int(len(X) * 0.8)
        X_tr, X_va = X.iloc[:split], X.iloc[split:]
        y_tr, y_va = y.iloc[:split], y.iloc[split:]

        try:
            best = self._tune_optuna(X_tr, y_tr, X_va, y_va, cfg, n_trials, timeout)
        except ImportError:
            best = self._tune_random(X_tr, y_tr, X_va, y_va, cfg, n_trials)

        self.config.xgb_params.update(best)
        self.fit(train, self.config)
        return best

    def _tune_optuna(self, X_tr, y_tr, X_va, y_va, cfg, n_trials, timeout) -> dict[str, Any]:
        import optuna
        from sklearn.metrics import mean_absolute_error

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: "optuna.Trial") -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            }
            est = self._make_estimator(cfg, overrides=params)
            est.fit(X_tr, y_tr)
            pred = est.predict(X_va)
            return float(mean_absolute_error(y_va, pred))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        return dict(study.best_params)

    def _tune_random(self, X_tr, y_tr, X_va, y_va, cfg, n_iter) -> dict[str, Any]:
        from sklearn.metrics import mean_absolute_error
        from sklearn.model_selection import ParameterSampler

        space = {
            "n_estimators": [150, 300, 500, 700],
            "max_depth": [3, 4, 6, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "min_child_weight": [1, 3, 5, 8],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "reg_lambda": [0.1, 1.0, 5.0],
        }
        best_score = float("inf")
        best_params: dict[str, Any] = {}
        rng = np.random.RandomState(cfg.random_state)
        for params in ParameterSampler(space, n_iter=n_iter, random_state=rng):
            est = self._make_estimator(cfg, overrides=params)
            est.fit(X_tr, y_tr)
            score = float(mean_absolute_error(y_va, est.predict(X_va)))
            if score < best_score:
                best_score = score
                best_params = dict(params)
        return best_params

    def _make_estimator(
        self, cfg: TSConfig, overrides: dict[str, Any] | None = None
    ) -> XGBRegressor:
        params = dict(cfg.xgb_params)
        params["random_state"] = cfg.random_state
        if overrides:
            params.update(overrides)
        return XGBRegressor(**params)
