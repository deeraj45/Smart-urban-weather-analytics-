"""
dashboard/app.py
-----------------
Lightweight Flask dashboard for the Smart Urban Weather Analytics
project. Serves:
  - /                 overview page (zone picker, latest readings, charts)
  - /api/zones         list of available zones
  - /api/timeseries    historical + fused readings for a zone (JSON, for charts)
  - /api/forecast      6h temperature forecast for a zone
  - /api/anomalies     recent flagged anomalies for a zone

Run with:  python dashboard/app.py
Then open: http://127.0.0.1:5000
"""

import os
import sys

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.forecasting_model import forecast_next_hours  # noqa: E402

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "flagged_weather_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "forecast_model.joblib")

app = Flask(__name__)


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "No processed data found. Run `python main.py` from the project root first."
        )
    return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])


@app.route("/")
def index():
    df = load_data()
    zones = sorted(df["zone"].unique().tolist())
    return render_template("index.html", zones=zones)


@app.route("/api/zones")
def api_zones():
    df = load_data()
    return jsonify(sorted(df["zone"].unique().tolist()))


@app.route("/api/timeseries")
def api_timeseries():
    zone = request.args.get("zone")
    df = load_data()
    zone_df = df[df["zone"] == zone].sort_values("timestamp").tail(24 * 7)  # last 7 days
    return jsonify({
        "timestamps": zone_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
        "temperature_c": zone_df["temperature_c"].round(2).tolist(),
        "humidity_pct": zone_df["humidity_pct"].round(1).tolist(),
        "wind_speed_kmh": zone_df["wind_speed_kmh"].round(1).tolist(),
        "is_anomaly": zone_df["is_anomaly"].tolist(),
    })


@app.route("/api/forecast")
def api_forecast():
    zone = request.args.get("zone")
    df = load_data()
    zone_df = df[df["zone"] == zone].sort_values("timestamp")
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not trained yet. Run `python main.py` first."}), 400

    model = joblib.load(MODEL_PATH)
    latest_row = zone_df.iloc[-1]
    forecast = forecast_next_hours(model, latest_row, hours_ahead=6)
    return jsonify({"zone": zone, "forecast_c": forecast})


@app.route("/api/anomalies")
def api_anomalies():
    zone = request.args.get("zone")
    df = load_data()
    zone_df = df[(df["zone"] == zone) & (df["is_anomaly"])].sort_values("timestamp", ascending=False)
    return jsonify(
        zone_df[["timestamp", "temperature_c", "humidity_pct", "anomaly_score"]]
        .assign(timestamp=lambda d: d["timestamp"].dt.strftime("%Y-%m-%d %H:%M"))
        .head(15)
        .to_dict(orient="records")
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
