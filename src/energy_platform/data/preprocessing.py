"""
Data preprocessing: validation, cleaning, train/test split, and persistence.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from energy_platform.config.settings import (
    PROCESSED_DIR,
    TEST_SIZE,
    RANDOM_SEED,
)
from energy_platform.data.feature_engineering import (
    build_feature_matrix,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def validate_raw(df: pd.DataFrame) -> None:
    """Assert basic schema and value constraints on raw data."""
    required = {"timestamp", "household_id", "target_consumption_kwh", "electricity_price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df["target_consumption_kwh"].lt(0).any():
        raise ValueError("Negative consumption values detected.")
    if df["electricity_price"].isna().mean() > 0.05:
        raise ValueError("More than 5% of electricity prices are NaN.")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers and fill minor gaps."""
    df = df.copy()
    # Cap consumption at 99th percentile per household
    p99 = df.groupby("household_id")["target_consumption_kwh"].transform(
        lambda x: x.quantile(0.99)
    )
    df["target_consumption_kwh"] = df["target_consumption_kwh"].clip(upper=p99)
    # Forward-fill small gaps in price (max 2 hours)
    df["electricity_price"] = df["electricity_price"].ffill(limit=2)
    return df


def preprocess(
    df: pd.DataFrame,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline.

    1. Validate → 2. Clean → 3. Feature engineering → 4. Train/test split

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    validate_raw(df)
    df = clean(df)
    df = build_feature_matrix(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=True
    )

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
        X_train.to_parquet(PROCESSED_DIR / "X_train.parquet", index=False)
        X_test.to_parquet(PROCESSED_DIR / "X_test.parquet", index=False)
        y_train.to_frame().to_parquet(PROCESSED_DIR / "y_train.parquet", index=False)
        y_test.to_frame().to_parquet(PROCESSED_DIR / "y_test.parquet", index=False)
        print(f"Saved splits → {PROCESSED_DIR}")

    return X_train, X_test, y_train, y_test
