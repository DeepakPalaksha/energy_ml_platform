"""Tests for data preprocessing and feature engineering."""

import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from energy_platform.data.generate_sample_data import generate_dataset
from energy_platform.data.feature_engineering import (
    build_feature_matrix,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from energy_platform.data.preprocessing import validate_raw, clean


@pytest.fixture
def raw_df():
    return generate_dataset(n_households=3, n_months=1, save=False)


def test_dataset_shape(raw_df):
    assert len(raw_df) > 0
    assert "timestamp" in raw_df.columns
    assert "target_consumption_kwh" in raw_df.columns


def test_no_negative_consumption(raw_df):
    assert (raw_df["target_consumption_kwh"] >= 0).all()


def test_validate_raw_passes(raw_df):
    validate_raw(raw_df)  # Should not raise


def test_validate_raw_fails_on_negatives(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "target_consumption_kwh"] = -1.0
    with pytest.raises(ValueError):
        validate_raw(bad)


def test_feature_matrix_columns(raw_df):
    df = build_feature_matrix(raw_df, drop_na=True)
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"Missing feature: {col}"
    assert TARGET_COLUMN in df.columns


def test_feature_matrix_no_nan(raw_df):
    df = build_feature_matrix(raw_df, drop_na=True)
    missing = df[FEATURE_COLUMNS].isnull().sum()
    assert missing.sum() == 0, f"NaN found: {missing[missing > 0]}"


def test_clean_caps_outliers(raw_df):
    dirty = raw_df.copy()
    dirty.loc[0, "target_consumption_kwh"] = 9999.0
    cleaned = clean(dirty)
    assert cleaned["target_consumption_kwh"].max() < 9999.0
