"""Model definitions and hyperparameter grids for GridSearchCV."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

ModelSpec = dict[str, Any]


def get_model_specs(random_state: int = 42) -> dict[str, ModelSpec]:
    """Return available classifiers and compact GridSearchCV param grids.

    Optional boosters (XGBoost / LightGBM / CatBoost) are included only when
    the corresponding package is installed.
    """
    specs: dict[str, ModelSpec] = {
        "LogisticRegression": {
            "estimator": LogisticRegression(
                max_iter=2000,
                random_state=random_state,
            ),
            "param_grid": {
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "solver": ["lbfgs", "saga"],
                "class_weight": [None, "balanced"],
            },
        },
        "RandomForestClassifier": {
            "estimator": RandomForestClassifier(random_state=random_state),
            "param_grid": {
                "n_estimators": [100, 200, 400],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "max_features": ["sqrt", "log2"],
                "class_weight": [None, "balanced"],
            },
        },
        "ExtraTreesClassifier": {
            "estimator": ExtraTreesClassifier(random_state=random_state),
            "param_grid": {
                "n_estimators": [100, 200, 400],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "max_features": ["sqrt", "log2"],
                "class_weight": [None, "balanced"],
            },
        },
        "HistGradientBoostingClassifier": {
            "estimator": HistGradientBoostingClassifier(random_state=random_state),
            "param_grid": {
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 5, None],
                "max_iter": [100, 200],
                "l2_regularization": [0.0, 0.1, 1.0],
            },
        },
        "SVC": {
            "estimator": SVC(random_state=random_state),
            "param_grid": {
                "C": [0.1, 1.0, 10.0, 100.0],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"],
                "class_weight": [None, "balanced"],
            },
        },
        "KNeighborsClassifier": {
            "estimator": KNeighborsClassifier(),
            "param_grid": {
                "n_neighbors": [3, 5, 7, 11, 15],
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            },
        },
    }

    try:
        from xgboost import XGBClassifier

        specs["XGBClassifier"] = {
            "estimator": XGBClassifier(
                tree_method="hist",
                random_state=random_state,
                n_jobs=1,
                verbosity=0,
            ),
            "param_grid": {
                "n_estimators": [100, 200, 400],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.2],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        }
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier

        specs["LGBMClassifier"] = {
            "estimator": LGBMClassifier(
                random_state=random_state,
                n_jobs=1,
                verbosity=-1,
            ),
            "param_grid": {
                "n_estimators": [100, 200, 400],
                "max_depth": [-1, 5, 10],
                "learning_rate": [0.05, 0.1, 0.2],
                "num_leaves": [31, 63],
                "subsample": [0.8, 1.0],
            },
        }
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier

        specs["CatBoostClassifier"] = {
            "estimator": CatBoostClassifier(
                random_seed=random_state,
                verbose=0,
                allow_writing_files=False,
            ),
            "param_grid": {
                "iterations": [100, 200, 400],
                "depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1, 0.2],
                "l2_leaf_reg": [1, 3, 5],
            },
        }
    except ImportError:
        pass

    return specs
