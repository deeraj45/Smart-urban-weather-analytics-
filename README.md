# 🌦️ Smart Urban Weather Analytics

An end-to-end data pipeline and dashboard that fuses multi-source urban
weather data (ground weather stations, IoT sensors, and satellite
estimates), cleans and merges it, forecasts short-term temperature
trends with machine learning, and flags anomalous readings in
real time — built for smart-city weather monitoring use cases.

## ✨ Features

- **Multi-source data ingestion** — simulates 3 independent data
  sources (ground station, IoT sensor network, satellite) across 5
  city zones, each with realistic noise, bias, and missing data.
- **Data cleaning & fusion** — removes physically impossible readings,
  interpolates gaps, and fuses sources using confidence-weighted
  averaging into one trustworthy reading per zone/hour.
- **ML-based forecasting** — a `RandomForestRegressor` trained on
  cyclical time features + lag features predicts temperature up to
  6 hours ahead (MAE ≈ 0.4 °C, R² ≈ 0.97 on synthetic data).
- **Anomaly detection** — an `IsolationForest` model flags unusual
  readings (sensor faults or genuine extreme-weather events) across
  temperature, humidity, wind, pressure, and rainfall.
- **Interactive dashboard** — a Flask + Chart.js web app to explore
  per-zone history, forecasts, and flagged anomalies.

## 🏗️ Architecture

```
Raw multi-source data  ─┐
 (ground/IoT/satellite) │
                         ▼
                 data_pipeline.py
             (clean → interpolate → fuse)
                         │
                         ▼
              clean_weather_data.csv
                 │                │
                 ▼                ▼
     forecasting_model.py   anomaly_detection.py
      (RandomForest)          (IsolationForest)
                 │                │
                 └───────┬────────┘
                         ▼
                  dashboard/app.py
                (Flask + Chart.js UI)
```

## 📁 Project Structure

```
smart-urban-weather-analytics/
├── main.py                      # Runs the full pipeline end-to-end
├── requirements.txt
├── src/
│   ├── data_generator.py        # Simulates multi-source raw weather data
│   ├── data_pipeline.py         # Cleaning, interpolation, source fusion, features
│   ├── forecasting_model.py     # RandomForest temperature forecasting
│   └── anomaly_detection.py     # IsolationForest anomaly flagging
├── dashboard/
│   ├── app.py                   # Flask app + REST API
│   ├── templates/index.html
│   └── static/{style.css, dashboard.js}
├── tests/
│   └── test_pipeline.py         # Unit tests for the pipeline
├── data/                        # raw/ and processed/ CSVs (generated, gitignored)
└── models/                      # trained model artifacts (generated, gitignored)
```

## 🚀 Getting Started

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/smart-urban-weather-analytics.git
cd smart-urban-weather-analytics
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
python main.py
```

This generates raw data, cleans and fuses it, trains the forecasting
model, runs anomaly detection, and saves everything under `data/` and
`models/`.

### 3. Launch the dashboard

```bash
python dashboard/app.py
```

Then open **http://127.0.0.1:5000** in your browser. Pick a zone from
the dropdown to see its temperature/humidity/wind history, a 6-hour
forecast, and any flagged anomalies.

### 4. Run tests

```bash
pytest tests/
```

## 🔧 Tech Stack

| Layer          | Tools                                  |
|----------------|-----------------------------------------|
| Data & cleaning| Python, Pandas, NumPy                   |
| Machine learning| scikit-learn (RandomForest, IsolationForest) |
| Backend/API    | Flask                                    |
| Frontend       | HTML, CSS, Chart.js                      |
| Testing        | pytest                                   |

## 🔄 Swapping in Real Data

`src/data_generator.py` is a stand-in for real APIs. To go live, replace
its output with data pulled from sources such as OpenWeatherMap,
national meteorological APIs, or your own IoT sensor network — as long
as the resulting DataFrame keeps the same columns
(`timestamp, zone, source, temperature_c, humidity_pct, wind_speed_kmh,
pressure_hpa, rainfall_mm`), the rest of the pipeline works unchanged.

## 📌 Notes

- All data in this repo is **synthetically generated** for
  demonstration purposes — no real proprietary weather data is used.
- The forecasting model uses a simple iterative multi-step approach
  suited for short horizons; for longer-horizon forecasting consider
  a dedicated time-series model (e.g. Prophet, ARIMA, or an LSTM).
