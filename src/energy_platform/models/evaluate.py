"""
Model evaluation utilities.

Computes MAE, RMSE, MAPE, R² and produces a prediction dataframe
suitable for logging and visualization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute regression metrics for consumption forecasting.

    Returns
    -------
    dict with mae, rmse, mape, r2
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    # MAPE — guard against zero actuals
    mask = y_true > 0.01
    mape = (
        float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        if mask.sum() > 0
        else float("nan")
    )

    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape), "r2": float(r2)}


def evaluation_dataframe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Return a tidy dataframe of predictions vs actuals for analysis."""
    df = pd.DataFrame(
        {
            "actual": y_true,
            "predicted": y_pred,
            "residual": y_true - y_pred,
            "abs_error": np.abs(y_true - y_pred),
        }
    )
    if timestamps is not None:
        df["timestamp"] = timestamps
    return df
