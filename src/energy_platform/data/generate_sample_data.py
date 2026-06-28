"""
Synthetic household energy dataset generator.

Produces realistic time-series data for N households over M months,
including IoT telemetry signals, weather, spot prices, and building metadata.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from energy_platform.config.settings import (
    RANDOM_SEED,
    N_HOUSEHOLDS,
    N_MONTHS,
    SAMPLE_DIR,
)


def _spot_price_series(timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Simulate Nord Pool-style hourly spot prices (EUR/MWh)."""
    n = len(timestamps)
    base = 60.0
    daily_pattern = 15 * np.sin(2 * np.pi * (timestamps.hour - 8) / 24)
    seasonal = 20 * np.sin(2 * np.pi * timestamps.dayofyear / 365)
    noise = rng.normal(0, 5, n)
    prices = base + daily_pattern + seasonal + noise
    return np.clip(prices, 5.0, 300.0)


def _outdoor_temperature(timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Simulate Swedish outdoor temperatures (°C)."""
    n = len(timestamps)
    seasonal = -10 * np.cos(2 * np.pi * timestamps.dayofyear / 365)
    diurnal = 3 * np.sin(2 * np.pi * (timestamps.hour - 6) / 24)
    noise = rng.normal(0, 2, n)
    return seasonal + diurnal + noise


def _solar_generation(
    timestamps: pd.DatetimeIndex,
    building_area: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Estimate solar PV generation (kWh) based on hour and season."""
    n = len(timestamps)
    hour = timestamps.hour
    dayofyear = timestamps.dayofyear
    daylight = np.clip(np.sin(np.pi * (hour - 5) / 14), 0, 1)
    seasonal_irr = 0.5 + 0.5 * np.sin(2 * np.pi * (dayofyear - 80) / 365)
    panel_kw = building_area * 0.003  # ~3W per m²
    noise = rng.uniform(0.85, 1.05, n)
    return np.clip(panel_kw * daylight * seasonal_irr * noise, 0, None)


def _consumption(
    timestamps: pd.DatetimeIndex,
    temperature: np.ndarray,
    solar: np.ndarray,
    household_size: int,
    insulation_score: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate hourly electricity consumption (kWh).

    Driven by: base load, heat pump demand (temperature), solar self-consumption,
    occupancy pattern (hour of day), and household size.
    """
    n = len(timestamps)
    hour = timestamps.hour

    # Base load by time of day
    base = 0.3 + 0.5 * np.clip(np.sin(np.pi * (hour.values - 6) / 12), 0, None)

    # Heat pump load — inversely proportional to insulation, driven by cold temp
    heat_demand = np.maximum(0, (18 - temperature)) * (1.0 / insulation_score) * 0.08

    # Occupancy bump: morning and evening peaks
    occupancy = 0.4 * ((hour >= 6) & (hour <= 9)) + 0.3 * ((hour >= 17) & (hour <= 22))

    # Household size scaling
    size_factor = 0.5 + 0.15 * household_size

    # Solar reduces net consumption
    solar_offset = solar * 0.7

    noise = rng.normal(0, 0.05, n)
    consumption = (base + heat_demand + occupancy) * size_factor - solar_offset + noise
    return np.clip(consumption, 0.05, None)


def generate_dataset(
    n_households: int = N_HOUSEHOLDS,
    n_months: int = N_MONTHS,
    seed: int = RANDOM_SEED,
    save: bool = True,
) -> pd.DataFrame:
    """
    Generate synthetic energy dataset for N households over N months.

    Parameters
    ----------
    n_households : number of synthetic households
    n_months     : number of months of hourly data
    seed         : random seed for reproducibility
    save         : if True, write parquet to data/sample/

    Returns
    -------
    pd.DataFrame with columns matching the 41-parameter telemetry schema.
    """
    rng = np.random.default_rng(seed)

    start = pd.Timestamp("2024-01-01")
    end = start + pd.DateOffset(months=n_months)
    timestamps = pd.date_range(start, end, freq="h", inclusive="left")
    n_timesteps = len(timestamps)

    # Shared market signals
    spot_prices = _spot_price_series(timestamps, rng)
    outdoor_temp = _outdoor_temperature(timestamps, rng)

    records: list[pd.DataFrame] = []

    for hid in range(n_households):
        # Per-household fixed metadata
        building_area = rng.uniform(60, 220)          # m²
        insulation_score = rng.uniform(1.5, 5.0)      # higher = better
        household_size = rng.integers(1, 6)
        has_battery = rng.random() > 0.4
        has_solar = rng.random() > 0.3
        battery_capacity = rng.uniform(5, 15) if has_battery else 0.0
        tariff_type = rng.choice(["fixed", "variable", "spot"])

        solar = _solar_generation(timestamps, building_area, rng) if has_solar else np.zeros(n_timesteps)

        consumption = _consumption(
            timestamps, outdoor_temp, solar, int(household_size), insulation_score, rng
        )

        # Battery state of charge (simple heuristic — charge cheap, discharge expensive)
        battery_soc = np.zeros(n_timesteps)
        if has_battery and battery_capacity > 0:
            soc = battery_capacity * 0.5
            price_median = np.median(spot_prices)
            for i in range(n_timesteps):
                if spot_prices[i] < price_median * 0.9 and soc < battery_capacity:
                    soc = min(soc + 1.0, battery_capacity)
                elif spot_prices[i] > price_median * 1.1 and soc > 0:
                    soc = max(soc - 1.0, 0)
                battery_soc[i] = soc

        heat_pump_status = (outdoor_temp < 5).astype(int)

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "household_id": f"HH_{hid:04d}",
                "outdoor_temperature": outdoor_temp,
                "electricity_price": spot_prices,
                "solar_generation_kwh": solar,
                "battery_soc": battery_soc,
                "heat_pump_status": heat_pump_status,
                "building_area": building_area,
                "insulation_score": insulation_score,
                "household_size": int(household_size),
                "has_battery": int(has_battery),
                "has_solar": int(has_solar),
                "battery_capacity_kwh": battery_capacity,
                "tariff_type": tariff_type,
                "target_consumption_kwh": consumption,
            }
        )
        records.append(df)

    dataset = pd.concat(records, ignore_index=True)

    if save:
        SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        out = SAMPLE_DIR / "households.parquet"
        dataset.to_parquet(out, index=False)
        print(f"Saved {len(dataset):,} rows → {out}")

    return dataset


if __name__ == "__main__":
    df = generate_dataset()
    print(df.head())
    print(df.dtypes)
