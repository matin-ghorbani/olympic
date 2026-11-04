"""Hyperparameter search helpers."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold


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
    """Run GridSearchCV with stratified folds and accuracy scoring."""
    n_splits = min(cv, _max_stratified_splits(y))
    cv_splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="accuracy",
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
        "mean_accuracy": float(search.cv_results_["mean_test_score"][best_idx]),
        "std_accuracy": float(search.cv_results_["std_test_score"][best_idx]),
        "best_params": dict(search.cv_results_["params"][best_idx]),
        "fit_time_sec": round(elapsed, 3),
        "status": "ok",
    }


def _max_stratified_splits(y: np.ndarray) -> int:
    _, counts = np.unique(y, return_counts=True)
    return int(counts.min())
