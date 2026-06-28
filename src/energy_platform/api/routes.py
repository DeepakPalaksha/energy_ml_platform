"""
API route handlers.

Organises prediction, pricing, and VPP endpoints into a single router.
Model is loaded once at startup and injected via FastAPI's app.state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from energy_platform.api.schemas import (
    ConsumptionFeatures,
    ConsumptionPredictionResponse,
    SubscriptionRequest,
    SubscriptionResponse,
    VPPRequest,
    VPPResponse,
    HealthResponse,
)
from energy_platform.models.predict import predict
from energy_platform.pricing.subscription_pricing import estimate_subscription
from energy_platform.vpp.battery_simulator import simulate_battery
from energy_platform.config.settings import ENV
from energy_platform import __version__

router = APIRouter()


@router.get("/", tags=["status"])
def root():
    return {
        "project": "Elvy Energy — VPP + Subscription Intelligence Platform",
        "version": __version__,
        "docs": "/docs",
    }


@router.get("/health", response_model=HealthResponse, tags=["status"])
def health():
    return HealthResponse(status="ok", version=__version__, environment=ENV)


@router.post(
    "/predict-consumption",
    response_model=ConsumptionPredictionResponse,
    tags=["ml"],
)
def predict_consumption(request: Request, features: ConsumptionFeatures):
    """
    Predict hourly electricity consumption for a single household-hour.

    Returns predicted kWh.
    """
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    feature_dict = features.to_feature_dict()
    df = pd.DataFrame([feature_dict])

    try:
        pred = predict(model, df)
        return ConsumptionPredictionResponse(
            predicted_consumption_kwh=round(float(pred[0]), 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/estimate-subscription",
    response_model=SubscriptionResponse,
    tags=["pricing"],
)
def estimate_subscription_price(body: SubscriptionRequest):
    """
    Estimate monthly subscription price for a household.

    Input: predicted hourly consumption + spot price series.
    Output: subscription quote with confidence band.
    """
    try:
        quote = estimate_subscription(
            predicted_hourly_kwh=body.predicted_hourly_kwh,
            spot_prices_eur_mwh=body.spot_prices_eur_mwh,
            hardware_amortization=body.hardware_amortization_eur,
        )
        return SubscriptionResponse(**quote.__dict__)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/simulate-vpp", response_model=VPPResponse, tags=["vpp"])
def simulate_vpp(body: VPPRequest):
    """
    Simulate battery charge/discharge schedule for VPP arbitrage.

    Returns estimated arbitrage value and per-hour schedule.
    """
    try:
        result = simulate_battery(
            spot_prices_eur_mwh=body.spot_prices_eur_mwh,
            capacity_kwh=body.capacity_kwh,
            max_rate_kw=body.max_rate_kw,
            efficiency=body.efficiency,
            initial_soc_pct=body.initial_soc_pct,
        )
        return VPPResponse(
            arbitrage_value_eur=result.arbitrage_value_eur,
            total_charged_kwh=result.total_charged_kwh,
            total_discharged_kwh=result.total_discharged_kwh,
            cycles=result.cycles,
            n_hours=result.n_hours,
            charge_schedule=result.charge_kwh,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
