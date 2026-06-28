"""
Feature engineering for energy consumption forecasting.

Produces calendar features, lag features, rolling statistics,
price-derived features, and weather interaction terms.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# Features that the model expects at inference time
FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_morning_peak",
    "is_evening_peak",
    "outdoor_temperature",
    "temp_below_zero",
    "electricity_price",
    "price_lag_1h",
    "price_lag_24h",
    "price_rolling_mean_24h",
    "solar_generation_kwh",
    "battery_soc",
    "heat_pump_status",
    "building_area",
    "insulation_score",
    "household_size",
    "has_battery",
    "has_solar",
    "consumption_lag_1h",
    "consumption_lag_24h",
    "consumption_lag_168h",
    "consumption_rolling_mean_24h",
    "consumption_rolling_std_24h",
    "temp_x_insulation",
    "price_x_battery",
]

TARGET_COLUMN = "target_consumption_kwh"


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from the timestamp column."""
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    df["is_morning_peak"] = ((ts.dt.hour >= 6) & (ts.dt.hour <= 9)).astype(int)
    df["is_evening_peak"] = ((ts.dt.hour >= 17) & (ts.dt.hour <= 22)).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and rolling features per household.

    Lags: 1h, 24h, 168h (1 week).
    Rolling: 24h mean and std.
    """
    df = df.copy().sort_values(["household_id", "timestamp"])

    for col, lag in [
        ("target_consumption_kwh", 1),
        ("target_consumption_kwh", 24),
        ("target_consumption_kwh", 168),
        ("electricity_price", 1),
        ("electricity_price", 24),
    ]:
        if col == "target_consumption_kwh":
            new_col = f"consumption_lag_{lag}h"
        else:
            new_col = f"price_lag_{lag}h"
        df[new_col] = df.groupby("household_id")[col].shift(lag)

    # Rolling stats (consumption)
    df["consumption_rolling_mean_24h"] = (
        df.groupby("household_id")["target_consumption_kwh"]
        .transform(lambda x: x.shift(1).rolling(24, min_periods=1).mean())
    )
    df["consumption_rolling_std_24h"] = (
        df.groupby("household_id")["target_consumption_kwh"]
        .transform(lambda x: x.shift(1).rolling(24, min_periods=1).std().fillna(0))
    )

    # Price rolling mean
    df["price_rolling_mean_24h"] = (
        df.groupby("household_id")["electricity_price"]
        .transform(lambda x: x.shift(1).rolling(24, min_periods=1).mean())
    )

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-driven interaction terms."""
    df = df.copy()
    df["temp_below_zero"] = (df["outdoor_temperature"] < 0).astype(int)
    df["temp_x_insulation"] = df["outdoor_temperature"] * df["insulation_score"]
    df["price_x_battery"] = df["electricity_price"] * df["has_battery"]
    return df


def build_feature_matrix(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    df       : raw dataset from generate_sample_data
    drop_na  : drop rows with NaN from lag features (first N rows per household)

    Returns
    -------
    DataFrame with FEATURE_COLUMNS + TARGET_COLUMN
    """
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_interaction_features(df)

    output_cols = FEATURE_COLUMNS + [TARGET_COLUMN, "timestamp", "household_id"]
    available = [c for c in output_cols if c in df.columns]
    df = df[available]

    if drop_na:
        before = len(df)
        df = df.dropna(subset=FEATURE_COLUMNS)
        print(f"Dropped {before - len(df):,} rows with NaN (lag warm-up). Remaining: {len(df):,}")

    return df.reset_index(drop=True)
