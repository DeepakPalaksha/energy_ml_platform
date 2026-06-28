"""
Model prediction and MLflow Model Registry integration.

mlflow is imported lazily inside functions to avoid blocking the test
suite during module collection when no MLflow server is running.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_platform.config.settings import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from energy_platform.data.feature_engineering import FEATURE_COLUMNS


def load_model_from_registry(model_name: str = "energy-consumption-xgb"):
    """Load a registered MLflow model, fallback to latest run."""
    import mlflow
    import mlflow.xgboost
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        model = mlflow.xgboost.load_model(f"models:/{model_name}/latest")
        print(f"Loaded from registry: {model_name}")
        return model
    except Exception as e:
        print(f"Registry load failed ({e}), falling back to latest run.")
        return load_model_from_latest_run()


def load_model_from_latest_run(experiment_name: str = MLFLOW_EXPERIMENT_NAME):
    """Load model artifact from the most recent MLflow run."""
    import mlflow
    import mlflow.xgboost
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"No MLflow experiment '{experiment_name}' found.")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No MLflow runs found.")
    run_id = runs[0].info.run_id
    print(f"Loaded from run: {run_id}")
    return mlflow.xgboost.load_model(f"runs:/{run_id}/model")


def predict(model, features: pd.DataFrame) -> np.ndarray:
    """Run inference. model is any sklearn-compatible regressor."""
    missing = set(FEATURE_COLUMNS) - set(features.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return model.predict(features[FEATURE_COLUMNS])
