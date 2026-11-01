"""Contest-ready univariate time series toolkit (XGBoost + temporal FE)."""

from .config import TSConfig
from .cv import summarize_cv, walk_forward_validate
from .features import add_features
from .model import XGBForecaster
from .preprocess import prepare, prepare_train_test
from .stationarity import check_stationarity
from .submit import to_submission

__all__ = [
    "TSConfig",
    "XGBForecaster",
    "prepare",
    "prepare_train_test",
    "add_features",
    "check_stationarity",
    "walk_forward_validate",
    "summarize_cv",
    "to_submission",
]

__version__ = "0.1.0"
