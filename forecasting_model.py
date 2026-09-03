"""
forecasting_model.py
---------------------
Trains a short-horizon temperature forecasting model per zone using
scikit-learn's RandomForestRegressor over engineered time features
and lag features produced by data_pipeline.py.

This keeps the project dependency-light (no deep learning framework
required) while still producing a genuinely useful, evaluable model.
Swap in an LSTM/Prophet model later without changing the rest of the
pipeline's interface.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

FEATURE_COLS = [
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "day_of_week", "humidity_pct", "wind_speed_kmh", "pressure_hpa",
    "temp_lag_1h", "temp_lag_3h", "temp_lag_24h",
]
TARGET_COL = "temperature_c"


def train_model(df: pd.DataFrame, model_path: str = "models/forecast_model.joblib"):
    """Train one global model (zone is implicitly captured via lag/humidity/pressure
    patterns); returns the fitted model and a metrics dict."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    model = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "mae_c": round(mean_absolute_error(y_test, preds), 3),
        "r2": round(r2_score(y_test, preds), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    joblib.dump(model, model_path)
    return model, metrics


def forecast_next_hours(model, latest_row: pd.Series, hours_ahead: int = 6) -> list:
    """Naive iterative forecaster: re-uses the latest known features and
    walks the hour forward, updating cyclical hour features and lag_1h
    with the previous prediction. Good enough for a short demo horizon."""
    preds = []
    row = latest_row.copy()
    for step in range(1, hours_ahead + 1):
        next_hour = (row["hour"] + step) % 24
        row_features = row.copy()
        row_features["hour_sin"] = np.sin(2 * np.pi * next_hour / 24)
        row_features["hour_cos"] = np.cos(2 * np.pi * next_hour / 24)
        X = pd.DataFrame([row_features[FEATURE_COLS]], columns=FEATURE_COLS)
        pred = model.predict(X)[0]
        preds.append(round(float(pred), 2))
        row["temp_lag_1h"] = pred  # feed prediction back in as next lag
    return preds


if __name__ == "__main__":
    df = pd.read_csv("data/processed/clean_weather_data.csv", parse_dates=["timestamp"])
    model, metrics = train_model(df)
    print("Forecast model trained:", metrics)

    latest = df.sort_values("timestamp").iloc[-1]
    print("Next 6h forecast (deg C):", forecast_next_hours(model, latest, hours_ahead=6))
