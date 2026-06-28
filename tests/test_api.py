"""FastAPI endpoint tests — no live MLflow or model required."""

import sys
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from energy_platform.api.routes import router
from energy_platform import __version__


@pytest.fixture
def client():
    """Create a test app with no lifespan (no MLflow dependency)."""
    test_app = FastAPI(title="test")
    test_app.include_router(router)
    test_app.state.model = None  # no model — predict returns 503
    with TestClient(test_app) as c:
        yield c


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "project" in r.json()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_no_model_returns_503(client):
    payload = {
        "hour": 10, "day_of_week": 1, "month": 3, "is_weekend": 0,
        "is_morning_peak": 1, "is_evening_peak": 0,
        "outdoor_temperature": -5.0, "temp_below_zero": 1,
        "electricity_price": 80.0, "price_lag_1h": 78.0,
        "price_lag_24h": 75.0, "price_rolling_mean_24h": 77.0,
        "solar_generation_kwh": 0.0, "battery_soc": 5.0,
        "heat_pump_status": 1, "building_area": 120.0,
        "insulation_score": 3.5, "household_size": 3,
        "has_battery": 1, "has_solar": 0,
        "consumption_lag_1h": 0.8, "consumption_lag_24h": 0.9,
        "consumption_lag_168h": 0.85, "consumption_rolling_mean_24h": 0.82,
        "consumption_rolling_std_24h": 0.05,
    }
    r = client.post("/predict-consumption", json=payload)
    assert r.status_code == 503


def test_estimate_subscription(client):
    payload = {
        "predicted_hourly_kwh": [0.5] * 720,
        "spot_prices_eur_mwh": [60.0] * 720,
        "hardware_amortization_eur": 45.0,
    }
    r = client.post("/estimate-subscription", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["final_price_eur"] > data["base_price_eur"]
    assert data["confidence_band_high"] >= data["confidence_band_low"]


def test_simulate_vpp(client):
    prices = list(np.tile([40.0, 40.0, 40.0, 120.0, 120.0, 120.0], 40))
    payload = {
        "spot_prices_eur_mwh": prices,
        "capacity_kwh": 10.0,
        "max_rate_kw": 5.0,
        "efficiency": 0.92,
        "initial_soc_pct": 0.5,
    }
    r = client.post("/simulate-vpp", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "arbitrage_value_eur" in data
    assert len(data["charge_schedule"]) == len(prices)
