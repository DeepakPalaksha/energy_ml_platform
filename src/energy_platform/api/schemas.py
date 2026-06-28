"""Pydantic request/response schemas for the Energy Platform API."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


# ── Consumption prediction ────────────────────────────────────────────────────

class ConsumptionFeatures(BaseModel):
    """Input features for a single-hour consumption prediction."""

    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    is_weekend: int = Field(..., ge=0, le=1)
    is_morning_peak: int = Field(0, ge=0, le=1)
    is_evening_peak: int = Field(0, ge=0, le=1)
    outdoor_temperature: float
    temp_below_zero: int = Field(0, ge=0, le=1)
    electricity_price: float = Field(..., gt=0)
    price_lag_1h: float
    price_lag_24h: float
    price_rolling_mean_24h: float
    solar_generation_kwh: float = Field(0.0, ge=0)
    battery_soc: float = Field(0.0, ge=0)
    heat_pump_status: int = Field(0, ge=0, le=1)
    building_area: float = Field(..., gt=0)
    insulation_score: float = Field(..., gt=0)
    household_size: int = Field(..., ge=1, le=10)
    has_battery: int = Field(0, ge=0, le=1)
    has_solar: int = Field(0, ge=0, le=1)
    consumption_lag_1h: float = Field(0.0)
    consumption_lag_24h: float = Field(0.0)
    consumption_lag_168h: float = Field(0.0)
    consumption_rolling_mean_24h: float = Field(0.0)
    consumption_rolling_std_24h: float = Field(0.0)
    temp_x_insulation: Optional[float] = None
    price_x_battery: Optional[float] = None

    def to_feature_dict(self) -> dict:
        d = self.model_dump()
        d["temp_x_insulation"] = d.get("temp_x_insulation") or (
            d["outdoor_temperature"] * d["insulation_score"]
        )
        d["price_x_battery"] = d.get("price_x_battery") or (
            d["electricity_price"] * d["has_battery"]
        )
        return d


class ConsumptionPredictionResponse(BaseModel):
    predicted_consumption_kwh: float
    model_version: str = "xgboost-baseline-v1"


# ── Subscription pricing ──────────────────────────────────────────────────────

class SubscriptionRequest(BaseModel):
    """Household profile for subscription price estimation."""

    predicted_hourly_kwh: list[float] = Field(..., min_length=1)
    spot_prices_eur_mwh: list[float] = Field(..., min_length=1)
    hardware_amortization_eur: float = Field(45.0, gt=0)


class SubscriptionResponse(BaseModel):
    base_price_eur: float
    risk_adjusted_price_eur: float
    final_price_eur: float
    expected_margin_eur: float
    confidence_band_low: float
    confidence_band_high: float
    predicted_monthly_kwh: float
    assumed_avg_spot_eur_mwh: float


# ── VPP simulation ────────────────────────────────────────────────────────────

class VPPRequest(BaseModel):
    """Parameters for battery VPP simulation."""

    spot_prices_eur_mwh: list[float] = Field(..., min_length=2)
    capacity_kwh: float = Field(10.0, gt=0)
    max_rate_kw: float = Field(5.0, gt=0)
    efficiency: float = Field(0.92, gt=0, le=1)
    initial_soc_pct: float = Field(0.5, ge=0, le=1)


class VPPResponse(BaseModel):
    arbitrage_value_eur: float
    total_charged_kwh: float
    total_discharged_kwh: float
    cycles: float
    n_hours: int
    charge_schedule: list[float]


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
