"""
VPP Battery Simulator.

Simulates a residential battery's charge/discharge schedule based on
Nord Pool spot prices, producing an estimated arbitrage value and
flexibility schedule for VPP market participation.

Strategy: charge when price < threshold (cheap hours),
          discharge when price > threshold (expensive hours).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from energy_platform.config.settings import (
    DEFAULT_BATTERY_CAPACITY_KWH,
    DEFAULT_MAX_CHARGE_RATE_KW,
    DEFAULT_EFFICIENCY,
)


@dataclass
class BatterySchedule:
    """Output of the battery simulator."""

    charge_kwh: list[float]         # Positive = charging, negative = discharging
    soc_kwh: list[float]            # State of charge at each timestep
    arbitrage_value_eur: float      # Total estimated profit from price arbitrage
    total_charged_kwh: float
    total_discharged_kwh: float
    cycles: float                   # Number of full charge/discharge cycles
    n_hours: int


def simulate_battery(
    spot_prices_eur_mwh: list[float] | np.ndarray,
    capacity_kwh: float = DEFAULT_BATTERY_CAPACITY_KWH,
    max_rate_kw: float = DEFAULT_MAX_CHARGE_RATE_KW,
    efficiency: float = DEFAULT_EFFICIENCY,
    initial_soc_pct: float = 0.5,
    low_price_pct: float = 0.35,   # Charge when price < 35th percentile
    high_price_pct: float = 0.65,  # Discharge when price > 65th percentile
) -> BatterySchedule:
    """
    Simulate battery charge/discharge schedule over a price series.

    Parameters
    ----------
    spot_prices_eur_mwh : hourly spot prices (EUR/MWh)
    capacity_kwh        : usable battery capacity
    max_rate_kw         : maximum charge or discharge rate (kW)
    efficiency          : round-trip efficiency (e.g. 0.92)
    initial_soc_pct     : starting state of charge as fraction of capacity
    low_price_pct       : percentile threshold for "cheap" hours (charge)
    high_price_pct      : percentile threshold for "expensive" hours (discharge)

    Returns
    -------
    BatterySchedule
    """
    prices = np.asarray(spot_prices_eur_mwh, dtype=float)
    n = len(prices)

    low_threshold = float(np.percentile(prices, low_price_pct * 100))
    high_threshold = float(np.percentile(prices, high_price_pct * 100))

    soc = initial_soc_pct * capacity_kwh
    soc_series = []
    charge_series = []
    arbitrage = 0.0
    total_charged = 0.0
    total_discharged = 0.0

    for i in range(n):
        p = prices[i]

        if p <= low_threshold and soc < capacity_kwh:
            # Charge
            room = capacity_kwh - soc
            charge = min(max_rate_kw * 1.0, room)   # 1h timestep
            soc += charge * efficiency
            soc = min(soc, capacity_kwh)
            arbitrage -= charge * p / 1000.0         # cost to charge
            total_charged += charge
            charge_series.append(charge)

        elif p >= high_threshold and soc > 0:
            # Discharge
            available = soc
            discharge = min(max_rate_kw * 1.0, available)
            soc -= discharge
            soc = max(soc, 0.0)
            arbitrage += discharge * efficiency * p / 1000.0   # revenue
            total_discharged += discharge
            charge_series.append(-discharge)

        else:
            charge_series.append(0.0)

        soc_series.append(round(soc, 4))

    cycles = total_discharged / capacity_kwh if capacity_kwh > 0 else 0.0

    return BatterySchedule(
        charge_kwh=charge_series,
        soc_kwh=soc_series,
        arbitrage_value_eur=round(arbitrage, 4),
        total_charged_kwh=round(total_charged, 4),
        total_discharged_kwh=round(total_discharged, 4),
        cycles=round(cycles, 2),
        n_hours=n,
    )
