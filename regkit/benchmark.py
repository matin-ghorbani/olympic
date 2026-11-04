"""Benchmark orchestration: compare tuned regressors on a prepared DataFrame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from regkit.models import get_model_specs
from regkit.tune import tune_model


@dataclass
class BenchmarkResult:
    leaderboard: pd.DataFrame
    best_model: str | None
    best_params: dict[str, Any] | None
    best_score: float | None

    def __repr__(self) -> str:
        lines = ["BenchmarkResult", "--------------"]
        if self.best_model is None:
            lines.append("No successful models.")
        else:
            lines.append(f"best_model : {self.best_model}")
            lines.append(f"best_score : {self.best_score:.6f}  (R²)")
            lines.append(f"best_params: {self.best_params}")
        lines.append("")
        lines.append(self.leaderboard.to_string(index=False))
        return "\n".join(lines)


def benchmark(
    df: pd.DataFrame,
    target: str | int | None = None,
    *,
    cv: int = 5,
    n_jobs: int = -1,
    random_state: int = 42,
    models: Sequence[str] | None = None,
) -> BenchmarkResult:
    """Tune and compare regressors with GridSearchCV; return R² leaderboard.

    Assumes ``df`` is already preprocessed / feature-engineered. Only numeric
    (or already-encoded) feature columns are expected.
    """
    X, y, _target_name = _split_xy(df, target)
    specs = get_model_specs(random_state=random_state)

    if models is not None:
        missing = [name for name in models if name not in specs]
        if missing:
            raise ValueError(
                f"Unknown or unavailable model(s): {missing}. "
                f"Available: {sorted(specs)}"
            )
        selected = {name: specs[name] for name in models}
    else:
        selected = specs

    rows: list[dict[str, Any]] = []
    for name, spec in selected.items():
        try:
            result = tune_model(
                estimator=spec["estimator"],
                param_grid=spec["param_grid"],
                X=X,
                y=y,
                cv=cv,
                n_jobs=n_jobs,
                random_state=random_state,
            )
            rows.append(
                {
                    "model": name,
                    "mean_r2": result["mean_r2"],
                    "std_r2": result["std_r2"],
                    "best_params": result["best_params"],
                    "fit_time_sec": result["fit_time_sec"],
                    "status": result["status"],
                }
            )
        except Exception as exc:  # noqa: BLE001 — contest robustness
            rows.append(
                {
                    "model": name,
                    "mean_r2": np.nan,
                    "std_r2": np.nan,
                    "best_params": None,
                    "fit_time_sec": np.nan,
                    "status": f"error: {type(exc).__name__}: {exc}",
                }
            )

    leaderboard = (
        pd.DataFrame(rows)
        .sort_values(
            by=["mean_r2", "std_r2"],
            ascending=[False, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    ok = leaderboard[leaderboard["status"] == "ok"]
    if ok.empty:
        return BenchmarkResult(
            leaderboard=leaderboard,
            best_model=None,
            best_params=None,
            best_score=None,
        )

    best_row = ok.iloc[0]
    return BenchmarkResult(
        leaderboard=leaderboard,
        best_model=str(best_row["model"]),
        best_params=best_row["best_params"],
        best_score=float(best_row["mean_r2"]),
    )


def _split_xy(
    df: pd.DataFrame,
    target: str | int | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if df.empty:
        raise ValueError("df is empty")

    if target is None:
        target_name = df.columns[-1]
    elif isinstance(target, int):
        target_name = df.columns[target]
    else:
        target_name = target

    if target_name not in df.columns:
        raise KeyError(f"target column {target_name!r} not found in df")

    feature_cols = [c for c in df.columns if c != target_name]
    if not feature_cols:
        raise ValueError("no feature columns left after removing target")

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_name].to_numpy(dtype=float)
    return X, y, str(target_name)
