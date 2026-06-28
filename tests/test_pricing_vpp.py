"""Tests for subscription pricing and VPP battery simulation."""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from energy_platform.pricing.subscription_pricing import estimate_subscription
from energy_platform.vpp.battery_simulator import simulate_battery


# ── Pricing tests ─────────────────────────────────────────────────────────────

def test_subscription_quote_structure():
    consumption = [0.5] * 720
    prices = [60.0] * 720
    quote = estimate_subscription(consumption, prices)
    assert quote.final_price_eur > 0
    assert quote.final_price_eur > quote.base_price_eur
    assert quote.confidence_band_high >= quote.confidence_band_low


def test_higher_consumption_higher_price():
    prices = [60.0] * 720
    low_consumption = [0.3] * 720
    high_consumption = [1.2] * 720
    q_low = estimate_subscription(low_consumption, prices)
    q_high = estimate_subscription(high_consumption, prices)
    assert q_high.final_price_eur > q_low.final_price_eur


def test_higher_spot_higher_price():
    consumption = [0.5] * 720
    q_cheap = estimate_subscription(consumption, [30.0] * 720)
    q_expensive = estimate_subscription(consumption, [150.0] * 720)
    assert q_expensive.final_price_eur > q_cheap.final_price_eur


def test_margin_is_positive():
    quote = estimate_subscription([0.5] * 720, [60.0] * 720)
    assert quote.expected_margin_eur > 0


# ── VPP tests ─────────────────────────────────────────────────────────────────

def test_battery_schedule_length():
    prices = [50.0, 50.0, 150.0, 150.0] * 10
    result = simulate_battery(prices)
    assert len(result.charge_kwh) == len(prices)
    assert len(result.soc_kwh) == len(prices)


def test_soc_stays_within_capacity():
    prices = list(np.random.uniform(30, 200, 168))
    cap = 10.0
    result = simulate_battery(prices, capacity_kwh=cap)
    assert all(0 <= s <= cap for s in result.soc_kwh), "SoC exceeded capacity bounds"


def test_arbitrage_positive_with_high_spread():
    """Battery should profit when cheap/expensive hours are clear."""
    prices = [20.0] * 12 + [200.0] * 12   # extreme spread
    result = simulate_battery(prices, capacity_kwh=10.0, max_rate_kw=5.0, initial_soc_pct=0.0)
    assert result.arbitrage_value_eur > 0


def test_zero_capacity_no_arbitrage():
    prices = [20.0, 200.0] * 5
    result = simulate_battery(prices, capacity_kwh=0.0)
    assert result.arbitrage_value_eur == 0.0
