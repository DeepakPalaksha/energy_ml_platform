"""Sanity checks for model training pipeline."""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from energy_platform.data.generate_sample_data import generate_dataset
from energy_platform.data.feature_engineering import build_feature_matrix, FEATURE_COLUMNS, TARGET_COLUMN
from energy_platform.models.evaluate import compute_metrics


@pytest.fixture
def feature_df():
    raw = generate_dataset(n_households=5, n_months=1, save=False)
    return build_feature_matrix(raw, drop_na=True)


def test_xgboost_trains_and_predicts(feature_df):
    """XGBoost should train without error and produce non-negative predictions."""
    import xgboost as xgb

    X = feature_df[FEATURE_COLUMNS]
    y = feature_df[TARGET_COLUMN]

    model = xgb.XGBRegressor(n_estimators=20, max_depth=3, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == len(y)
    assert (preds >= 0).all(), "Model produced negative predictions"


def test_metrics_are_reasonable(feature_df):
    """R² should be positive for a model trained and tested on same data."""
    import xgboost as xgb

    X = feature_df[FEATURE_COLUMNS]
    y = feature_df[TARGET_COLUMN].values

    model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)

    metrics = compute_metrics(y, preds)
    assert metrics["r2"] > 0.5, f"R² too low on training data: {metrics['r2']}"
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0


def test_compute_metrics_basic():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    m = compute_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
