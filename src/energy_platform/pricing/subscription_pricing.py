"""
Subscription pricing engine.

Converts predicted household energy consumption into a monthly fixed
subscription price, accounting for spot price exposure, risk buffer,
hardware amortization, and service margin.

This mirrors how Elvy Energy prices long-term contracts — the model
must get pricing right because contracts run for 15 years.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from energy_platform.config.settings import (
    RISK_BUFFER_PCT,
    SERVICE_MARGIN_PCT,
    HARDWARE_AMORTIZATION_MONTHLY,
)


@dataclass
class SubscriptionQuote:
    """Output of the subscription pricing engine."""

    base_price_eur: float           # Expected cost-of-energy, no margin
    risk_adjusted_price_eur: float  # + risk buffer
    final_price_eur: float          # + service margin + hardware amortization
    expected_margin_eur: float      # Revenue – cost
    confidence_band_low: float      # 10th percentile scenario
    confidence_band_high: float     # 90th percentile scenario
    predicted_monthly_kwh: float
    assumed_avg_spot_eur_mwh: float


def estimate_subscription(
    predicted_hourly_kwh: list[float] | np.ndarray,
    spot_prices_eur_mwh: list[float] | np.ndarray,
    hardware_amortization: float = HARDWARE_AMORTIZATION_MONTHLY,
    risk_buffer_pct: float = RISK_BUFFER_PCT,
    service_margin_pct: float = SERVICE_MARGIN_PCT,
    n_simulations: int = 1000,
    seed: int = 42,
) -> SubscriptionQuote:
    """
    Estimate monthly subscription price for a household.

    Parameters
    ----------
    predicted_hourly_kwh  : array of hourly consumption predictions (kWh)
    spot_prices_eur_mwh   : array of hourly electricity spot prices (EUR/MWh)
    hardware_amortization : monthly cost of hardware assets (EUR)
    risk_buffer_pct       : fraction added to cover price volatility risk
    service_margin_pct    : Elvy's gross margin percentage
    n_simulations         : Monte Carlo draws for confidence band

    Returns
    -------
    SubscriptionQuote
    """
    consumption = np.asarray(predicted_hourly_kwh, dtype=float)
    spot = np.asarray(spot_prices_eur_mwh, dtype=float)

    # Scale to 30-day month if input is shorter/longer
    hours_per_month = 24 * 30
    if len(consumption) != hours_per_month:
        # Tile or trim
        tiles = int(np.ceil(hours_per_month / len(consumption)))
        consumption = np.tile(consumption, tiles)[:hours_per_month]
        spot = np.tile(spot, tiles)[:hours_per_month]

    monthly_kwh = float(consumption.sum())
    avg_spot = float(spot.mean())

    # Base energy cost: sum(consumption_kWh * spot_EUR/MWh / 1000)
    base_cost = float((consumption * spot / 1000.0).sum())

    # Risk-adjusted cost
    risk_adjusted_cost = base_cost * (1 + risk_buffer_pct)

    # Final price includes hardware amortization and service margin
    pre_margin = risk_adjusted_cost + hardware_amortization
    final_price = pre_margin / (1 - service_margin_pct)
    expected_margin = final_price - pre_margin

    # Monte Carlo confidence band — vary spot price with ±20% volatility
    rng = np.random.default_rng(seed)
    simulated_costs = []
    for _ in range(n_simulations):
        spot_sim = spot * rng.uniform(0.8, 1.2, size=len(spot))
        sim_cost = (consumption * spot_sim / 1000.0).sum()
        sim_risk = sim_cost * (1 + risk_buffer_pct)
        sim_final = (sim_risk + hardware_amortization) / (1 - service_margin_pct)
        simulated_costs.append(sim_final)

    simulated_costs_arr = np.array(simulated_costs)

    return SubscriptionQuote(
        base_price_eur=round(base_cost, 2),
        risk_adjusted_price_eur=round(risk_adjusted_cost, 2),
        final_price_eur=round(final_price, 2),
        expected_margin_eur=round(expected_margin, 2),
        confidence_band_low=round(float(np.percentile(simulated_costs_arr, 10)), 2),
        confidence_band_high=round(float(np.percentile(simulated_costs_arr, 90)), 2),
        predicted_monthly_kwh=round(monthly_kwh, 2),
        assumed_avg_spot_eur_mwh=round(avg_spot, 2),
    )
