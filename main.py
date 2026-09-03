"""
main.py
-------
End-to-end runner for the Smart Urban Weather Analytics pipeline:
  1. Generate (or load) raw multi-source weather data
  2. Clean + fuse it into one hourly dataset per zone
  3. Train the forecasting model and report accuracy
  4. Run anomaly detection and report flagged readings
  5. Save all artifacts under data/ and models/ for the dashboard to use

Usage:
    python main.py
"""

import os

from src.data_generator import generate_raw_dataset
from src.data_pipeline import run_pipeline
from src.forecasting_model import train_model, forecast_next_hours
from src.anomaly_detection import detect_anomalies


def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("Step 1/4: Generating multi-source raw weather data...")
    raw_df = generate_raw_dataset(n_days=45)
    raw_df.to_csv("data/raw/raw_weather_data.csv", index=False)
    print(f"  -> {len(raw_df):,} raw readings from {raw_df['source'].nunique()} sources")

    print("Step 2/4: Cleaning and fusing sources...")
    clean_df = run_pipeline()

    print("Step 3/4: Training forecasting model...")
    model, metrics = train_model(clean_df)
    print(f"  -> MAE: {metrics['mae_c']} C | R2: {metrics['r2']}")
    latest = clean_df.sort_values("timestamp").iloc[-1]
    forecast = forecast_next_hours(model, latest, hours_ahead=6)
    print(f"  -> Next 6h forecast for last zone in data: {forecast}")

    print("Step 4/4: Running anomaly detection...")
    flagged = detect_anomalies(clean_df)
    flagged.to_csv("data/processed/flagged_weather_data.csv", index=False)
    n_anom = int(flagged["is_anomaly"].sum())
    print(f"  -> {n_anom} anomalies flagged out of {len(flagged):,} readings")

    print("\nDone. Run 'python dashboard/app.py' to view results in the dashboard.")


if __name__ == "__main__":
    main()
