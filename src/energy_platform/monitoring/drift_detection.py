"""
Model monitoring — data drift and prediction drift detection.

Uses Population Stability Index (PSI) and simple statistical tests
to flag when incoming data diverges from training distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


PSI_THRESHOLD_WARN = 0.1    # Slight distribution shift
PSI_THRESHOLD_ALERT = 0.25  # Significant shift — retrain candidate


@dataclass
class DriftReport:
    feature: str
    psi: float
    status: str   # "ok", "warn", "alert"


def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Population Stability Index between two distributions.

    PSI < 0.1   : no significant change
    PSI 0.1–0.25: moderate change, monitor
    PSI > 0.25  : significant change, retrain
    """
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf

    expected_pct = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=bins)[0] / len(actual)

    # Avoid log(0)
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)

    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return round(psi, 5)


def check_drift(
    train_df: pd.DataFrame,
    live_df: pd.DataFrame,
    feature_cols: list[str],
) -> list[DriftReport]:
    """
    Check data drift for all numeric features.

    Parameters
    ----------
    train_df     : training-time feature distribution
    live_df      : incoming/live feature distribution
    feature_cols : list of feature names to check

    Returns
    -------
    list of DriftReport, one per feature
    """
    reports = []
    for col in feature_cols:
        if col not in train_df.columns or col not in live_df.columns:
            continue
        psi = compute_psi(
            train_df[col].dropna().values,
            live_df[col].dropna().values,
        )
        if psi < PSI_THRESHOLD_WARN:
            status = "ok"
        elif psi < PSI_THRESHOLD_ALERT:
            status = "warn"
        else:
            status = "alert"
        reports.append(DriftReport(feature=col, psi=psi, status=status))

    return sorted(reports, key=lambda r: -r.psi)
