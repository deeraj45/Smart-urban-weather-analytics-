"""
data_pipeline.py
-----------------
Cleans and fuses raw multi-source weather readings into a single,
analysis-ready hourly dataset per zone.

Steps:
  1. Load raw multi-source data
  2. Handle missing values (interpolate within each source/zone series)
  3. Remove impossible / out-of-range physical readings
  4. Fuse the 3 sources per zone into one reading using a confidence-
     weighted average (ground_station is trusted most, satellite least)
  5. Engineer time-based features used later for forecasting
  6. Save cleaned dataset to data/processed/
"""

import numpy as np
import pandas as pd

# Confidence weights used when fusing multiple sources for the same zone/time
SOURCE_WEIGHTS = {"ground_station": 0.6, "iot_sensor": 0.25, "satellite": 0.15}

PHYSICAL_BOUNDS = {
    "temperature_c": (-40, 55),
    "humidity_pct": (0, 100),
    "wind_speed_kmh": (0, 200),
    "pressure_hpa": (870, 1085),
    "rainfall_mm": (0, 500),
}


def load_raw(path: str = "data/raw/raw_weather_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def clip_out_of_range(df: pd.DataFrame) -> pd.DataFrame:
    """Turn physically impossible readings into NaN so they get interpolated."""
    df = df.copy()
    for col, (low, high) in PHYSICAL_BOUNDS.items():
        if col in df.columns:
            out_of_range = ~df[col].between(low, high)
            df.loc[out_of_range, col] = np.nan
    return df


def interpolate_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate missing values per zone+source time series."""
    df = df.sort_values(["zone", "source", "timestamp"]).copy()
    numeric_cols = ["temperature_c", "humidity_pct", "wind_speed_kmh", "pressure_hpa", "rainfall_mm"]

    group_keys = df.groupby(["zone", "source"], sort=False).ngroup()
    interpolated = (
        df[numeric_cols]
        .groupby(group_keys)
        .transform(lambda s: s.interpolate(limit_direction="both"))
    )
    df[numeric_cols] = interpolated
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df


def fuse_sources(df: pd.DataFrame) -> pd.DataFrame:
    """Combine the 3 sources per zone/timestamp into one confidence-weighted reading."""
    df = df.copy()
    df["weight"] = df["source"].map(SOURCE_WEIGHTS)

    weighted_cols = ["temperature_c", "humidity_pct", "wind_speed_kmh", "pressure_hpa", "rainfall_mm"]
    for col in weighted_cols:
        df[f"w_{col}"] = df[col] * df["weight"]

    grouped = df.groupby(["zone", "timestamp"])
    fused = grouped[[f"w_{c}" for c in weighted_cols]].sum()
    weight_sum = grouped["weight"].sum()
    fused = fused.div(weight_sum, axis=0)
    fused.columns = weighted_cols
    fused = fused.reset_index()
    return fused


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features used by the forecasting model."""
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # Lag features per zone, useful for short-horizon forecasting
    df = df.sort_values(["zone", "timestamp"])
    for lag in (1, 3, 24):
        df[f"temp_lag_{lag}h"] = df.groupby("zone")["temperature_c"].shift(lag)
    df = df.dropna().reset_index(drop=True)
    return df


def run_pipeline(raw_path: str = "data/raw/raw_weather_data.csv",
                  out_path: str = "data/processed/clean_weather_data.csv") -> pd.DataFrame:
    df = load_raw(raw_path)
    df = clip_out_of_range(df)
    df = interpolate_gaps(df)
    fused = fuse_sources(df)
    featured = engineer_features(fused)
    featured.to_csv(out_path, index=False)
    print(f"Cleaned & fused dataset saved -> {out_path} ({len(featured):,} rows)")
    return featured


if __name__ == "__main__":
    run_pipeline()
