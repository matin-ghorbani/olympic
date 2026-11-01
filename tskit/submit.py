from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def to_submission(
    test_df: pd.DataFrame,
    preds: pd.Series | np.ndarray | pd.DataFrame,
    path: str | Path,
    *,
    id_col: str | None = None,
    pred_col: str = "prediction",
    date_col: str = "date",
) -> pd.DataFrame:
    """Write a contest-style submission CSV.

    - If ``id_col`` is given, output columns are ``[id_col, pred_col]``.
    - Else if ``date_col`` exists on test, output ``[date_col, pred_col]``.
    - Else a single ``pred_col`` column aligned to test row order.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(preds, pd.DataFrame):
        if pred_col in preds.columns:
            values = preds[pred_col].to_numpy()
        elif "prediction" in preds.columns:
            values = preds["prediction"].to_numpy()
        else:
            values = preds.iloc[:, -1].to_numpy()
    else:
        values = np.asarray(preds)

    if len(values) != len(test_df):
        raise ValueError(
            f"pred length {len(values)} != test length {len(test_df)}"
        )

    if id_col is not None:
        if id_col not in test_df.columns:
            raise KeyError(f"id_col '{id_col}' not in test columns")
        out = pd.DataFrame({id_col: test_df[id_col].to_numpy(), pred_col: values})
    elif date_col in test_df.columns:
        out = pd.DataFrame({date_col: test_df[date_col].to_numpy(), pred_col: values})
    else:
        out = pd.DataFrame({pred_col: values})

    out.to_csv(path, index=False)
    return out
