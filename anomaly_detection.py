"""
anomaly_detection.py
----------------------
Flags unusual weather readings (sensor faults, extreme/rare weather
events) using an Isolation Forest over the cleaned, fused dataset.

An anomaly score close to -1 means "very unusual"; we expose both the
raw score and a boolean `is_anomaly` flag using a configurable
contamination rate.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest

ANOMALY_FEATURES = ["temperature_c", "humidity_pct", "wind_speed_kmh", "pressure_hpa", "rainfall_mm"]


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.02) -> pd.DataFrame:
    """Adds `anomaly_score` and `is_anomaly` columns to a copy of df."""
    df = df.copy()
    model = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=42
    )
    model.fit(df[ANOMALY_FEATURES])

    df["anomaly_score"] = model.decision_function(df[ANOMALY_FEATURES])
    df["is_anomaly"] = model.predict(df[ANOMALY_FEATURES]) == -1
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/clean_weather_data.csv", parse_dates=["timestamp"])
    flagged = detect_anomalies(df)
    n_anomalies = int(flagged["is_anomaly"].sum())
    print(f"Detected {n_anomalies} anomalies out of {len(flagged):,} readings "
          f"({n_anomalies / len(flagged):.2%})")
    print(flagged[flagged["is_anomaly"]].sort_values("anomaly_score")
          [["timestamp", "zone", "temperature_c", "anomaly_score"]].head(10))
