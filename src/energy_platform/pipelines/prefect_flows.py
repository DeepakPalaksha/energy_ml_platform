"""
Prefect orchestration flows for the full ML lifecycle.

Flows:
  - generate_data_flow      : create synthetic dataset
  - preprocess_flow         : feature engineering + train/test split
  - train_flow              : train + log to MLflow
  - evaluate_flow           : evaluate latest model
  - full_pipeline_flow      : end-to-end orchestration
  - batch_prediction_flow   : run inference on new data
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from prefect import flow, task, get_run_logger

from energy_platform.config.settings import SAMPLE_DIR, PROCESSED_DIR


# ── Tasks ─────────────────────────────────────────────────────────────────────

@task(name="generate-sample-data", retries=1)
def generate_data_task(n_households: int = 50, n_months: int = 6) -> str:
    from energy_platform.data.generate_sample_data import generate_dataset
    logger = get_run_logger()
    logger.info(f"Generating data: {n_households} households × {n_months} months")
    df = generate_dataset(n_households=n_households, n_months=n_months, save=True)
    logger.info(f"Generated {len(df):,} rows")
    return str(SAMPLE_DIR / "households.parquet")


@task(name="preprocess-data", retries=1)
def preprocess_task(data_path: str) -> str:
    from energy_platform.data.preprocessing import preprocess
    logger = get_run_logger()
    logger.info(f"Preprocessing: {data_path}")
    df = pd.read_parquet(data_path)
    preprocess(df, save=True)
    logger.info("Preprocessed splits saved.")
    return str(PROCESSED_DIR)


@task(name="train-model", retries=1)
def train_task(run_name: str = "prefect-run", register: bool = True) -> dict:
    from energy_platform.models.train import train
    logger = get_run_logger()
    logger.info(f"Training model: {run_name}")
    _, metrics = train(run_name=run_name, register_model=register)
    logger.info(f"Training metrics: {metrics}")
    return metrics


@task(name="batch-predict", retries=1)
def batch_predict_task(data_path: str) -> str:
    import numpy as np
    from energy_platform.models.predict import load_model_from_latest_run, predict
    from energy_platform.data.feature_engineering import FEATURE_COLUMNS

    logger = get_run_logger()
    logger.info("Loading model and running batch inference...")
    model = load_model_from_latest_run()
    df = pd.read_parquet(data_path)
    preds = predict(model, df)
    out = pd.DataFrame({"predicted_consumption_kwh": preds})
    out_path = str(PROCESSED_DIR / "batch_predictions.parquet")
    out.to_parquet(out_path, index=False)
    logger.info(f"Batch predictions saved → {out_path}")
    return out_path


# ── Flows ─────────────────────────────────────────────────────────────────────

@flow(name="generate-data-flow", log_prints=True)
def generate_data_flow(n_households: int = 50, n_months: int = 6) -> str:
    return generate_data_task(n_households=n_households, n_months=n_months)


@flow(name="preprocess-flow", log_prints=True)
def preprocess_flow(data_path: str | None = None) -> str:
    path = data_path or str(SAMPLE_DIR / "households.parquet")
    return preprocess_task(path)


@flow(name="train-flow", log_prints=True)
def train_flow(run_name: str = "prefect-train", register: bool = True) -> dict:
    return train_task(run_name=run_name, register=register)


@flow(name="full-pipeline-flow", log_prints=True)
def full_pipeline_flow(
    n_households: int = 50,
    n_months: int = 6,
    run_name: str = "full-pipeline",
) -> dict:
    """
    End-to-end ML pipeline:
      generate → preprocess → train → (batch predict)
    """
    data_path = generate_data_task(n_households=n_households, n_months=n_months)
    preprocess_task(data_path)
    metrics = train_task(run_name=run_name, register=True)
    return metrics


@flow(name="batch-prediction-flow", log_prints=True)
def batch_prediction_flow(data_path: str | None = None) -> str:
    path = data_path or str(PROCESSED_DIR / "X_test.parquet")
    return batch_predict_task(path)


if __name__ == "__main__":
    full_pipeline_flow()
