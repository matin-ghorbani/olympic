"""Hyperparameter search helpers."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.model_selection import GridSearchCV, KFold


def tune_model(
    estimator: Any,
    param_grid: dict[str, list[Any]],
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv: int = 5,
    n_jobs: int = -1,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run GridSearchCV with K-fold CV and R² scoring."""
    n_splits = min(cv, len(y))
    if n_splits < 2:
        raise ValueError("need at least 2 samples for cross-validation")

    cv_splitter = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="r2",
        cv=cv_splitter,
        n_jobs=n_jobs,
        refit=False,
        error_score=np.nan,
    )

    t0 = time.perf_counter()
    search.fit(X, y)
    elapsed = time.perf_counter() - t0

    best_idx = int(np.nanargmax(search.cv_results_["mean_test_score"]))
    return {
        "mean_r2": float(search.cv_results_["mean_test_score"][best_idx]),
        "std_r2": float(search.cv_results_["std_test_score"][best_idx]),
        "best_params": dict(search.cv_results_["params"][best_idx]),
        "fit_time_sec": round(elapsed, 3),
        "status": "ok",
    }
