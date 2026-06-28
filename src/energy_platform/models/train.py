"""
Model training with full MLflow experiment tracking.

Trains an XGBoost baseline for hourly consumption forecasting.
Logs parameters, metrics, feature importance, and model artifact.
"""

from __future__ import annotations

import json
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

from energy_platform.config.settings import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    PROCESSED_DIR,
    RANDOM_SEED,
)
from energy_platform.data.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN
from energy_platform.models.evaluate import compute_metrics


# Default XGBoost hyperparameters — tunable via MLflow runs
DEFAULT_PARAMS: dict = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "tree_method": "hist",
}


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load preprocessed train/test splits from disk."""
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet").squeeze()
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet").squeeze()
    return X_train, X_test, y_train, y_test


def train(
    params: dict | None = None,
    run_name: str = "xgboost-baseline",
    register_model: bool = False,
) -> tuple[xgb.XGBRegressor, dict]:
    """
    Train XGBoost model and log everything to MLflow.

    Parameters
    ----------
    params         : override default hyperparameters
    run_name       : MLflow run display name
    register_model : register best model in MLflow Model Registry

    Returns
    -------
    (fitted model, metrics dict)
    """
    hp = {**DEFAULT_PARAMS, **(params or {})}

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_splits()

    with mlflow.start_run(run_name=run_name) as run:
        # ── Log hyperparameters ──────────────────────────────────────────────
        mlflow.log_params(hp)
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))
        mlflow.log_param("n_features", len(FEATURE_COLUMNS))

        # ── Train ────────────────────────────────────────────────────────────
        model = xgb.XGBRegressor(**hp)
        model.fit(
            X_train[FEATURE_COLUMNS],
            y_train,
            eval_set=[(X_test[FEATURE_COLUMNS], y_test)],
            verbose=False,
        )

        # ── Evaluate ─────────────────────────────────────────────────────────
        y_pred = model.predict(X_test[FEATURE_COLUMNS])
        metrics = compute_metrics(y_test.values, y_pred)

        mlflow.log_metrics(metrics)
        print(f"[{run_name}] MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
              f"MAPE={metrics['mape']:.2f}%  R²={metrics['r2']:.4f}")

        # ── Feature importance ───────────────────────────────────────────────
        importance = dict(
            zip(FEATURE_COLUMNS, model.feature_importances_.tolist())
        )
        top_features = sorted(importance.items(), key=lambda x: -x[1])[:10]
        mlflow.log_dict({"feature_importance": importance}, "feature_importance.json")
        print("Top features:", top_features[:5])

        # ── Log model artifact ───────────────────────────────────────────────
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name="energy-consumption-xgb" if register_model else None,
        )

        run_id = run.info.run_id
        print(f"MLflow run_id: {run_id}")

    return model, metrics


if __name__ == "__main__":
    model, metrics = train(register_model=True)
    print("Final metrics:", metrics)
