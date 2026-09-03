"""
data_generator.py
------------------
Simulates multi-source urban weather data for the Smart Urban Weather
Analytics project.

In a production deployment this module would be replaced by real API
clients (e.g. OpenWeatherMap, government weather-station feeds, IoT
sensor networks). For development, testing, and demo purposes it
generates realistic synthetic data with day/night and seasonal cycles,
sensor noise, missing values, and injected anomalies so the rest of
the pipeline (cleaning, forecasting, anomaly detection, dashboard) has
something meaningful to work with.

Three "sources" are simulated per city zone:
  1. ground_station  - official weather station (low noise, hourly)
  2. iot_sensor       - cheap IoT sensor network (higher noise, more gaps)
  3. satellite        - coarse satellite-derived estimate (smooth, biased)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

ZONES = ["Downtown", "Riverside", "Industrial_Park", "Suburb_North", "Suburb_South"]
SOURCES = ["ground_station", "iot_sensor", "satellite"]


def _daily_seasonal_signal(hours: np.ndarray) -> np.ndarray:
    """Base temperature curve combining daily + seasonal cycles (deg C)."""
    day_of_year = (hours / 24) % 365
    seasonal = 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # summer peak
    diurnal = 6 * np.sin(2 * np.pi * (hours % 24 - 9) / 24)      # afternoon peak
    return 22 + seasonal + diurnal


def generate_raw_dataset(n_days: int = 30, freq_hours: int = 1, seed: int = 42) -> pd.DataFrame:
    """Generate a raw, messy multi-source weather dataset.

    Returns a DataFrame with columns:
      timestamp, zone, source, temperature_c, humidity_pct,
      wind_speed_kmh, pressure_hpa, rainfall_mm
    Includes injected missing values and a handful of extreme anomalies
    to exercise the cleaning and anomaly-detection stages.
    """
    global RNG
    RNG = np.random.default_rng(seed)

    n_steps = int(n_days * 24 / freq_hours)
    timestamps = pd.date_range("2026-01-01", periods=n_steps, freq=f"{freq_hours}h")
    hours = np.arange(n_steps) * freq_hours

    rows = []
    for zone in ZONES:
        zone_bias = RNG.uniform(-1.5, 1.5)          # microclimate offset per zone
        base_temp = _daily_seasonal_signal(hours) + zone_bias

        for source in SOURCES:
            noise_scale = {"ground_station": 0.3, "iot_sensor": 1.2, "satellite": 0.6}[source]
            source_bias = {"ground_station": 0.0, "iot_sensor": RNG.uniform(-0.5, 0.5),
                            "satellite": RNG.uniform(-1.0, 1.0)}[source]

            temp = base_temp + source_bias + RNG.normal(0, noise_scale, n_steps)
            humidity = np.clip(60 + 20 * np.sin(2 * np.pi * (hours % 24) / 24 - 2)
                                + RNG.normal(0, 5, n_steps), 5, 100)
            wind = np.clip(RNG.gamma(2.0, 3.0, n_steps), 0, None)
            pressure = 1013 + RNG.normal(0, 4, n_steps) - 0.05 * (temp - 22)
            rainfall = np.clip(RNG.exponential(0.4, n_steps) - 0.3, 0, None)
            rainfall[RNG.random(n_steps) > 0.15] = 0.0  # rain only ~15% of readings

            df = pd.DataFrame({
                "timestamp": timestamps,
                "zone": zone,
                "source": source,
                "temperature_c": temp,
                "humidity_pct": humidity,
                "wind_speed_kmh": wind,
                "pressure_hpa": pressure,
                "rainfall_mm": rainfall,
            })
            rows.append(df)

    data = pd.concat(rows, ignore_index=True)

    # Inject missing values (simulating sensor dropouts / API failures)
    for col in ["temperature_c", "humidity_pct", "wind_speed_kmh", "pressure_hpa"]:
        mask = RNG.random(len(data)) < 0.02
        data.loc[mask, col] = np.nan

    # Inject a handful of extreme anomalies (sensor faults / real extreme events)
    anomaly_idx = RNG.choice(len(data), size=max(5, len(data) // 400), replace=False)
    data.loc[anomaly_idx, "temperature_c"] += RNG.choice([-1, 1], size=len(anomaly_idx)) * RNG.uniform(15, 25, len(anomaly_idx))

    return data.sort_values(["zone", "source", "timestamp"]).reset_index(drop=True)


if __name__ == "__main__":
    df = generate_raw_dataset(n_days=30)
    out_path = "data/raw/raw_weather_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} rows across {df['zone'].nunique()} zones "
          f"and {df['source'].nunique()} sources -> {out_path}")
