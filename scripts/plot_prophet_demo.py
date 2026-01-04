#!/usr/bin/env python3
"""
Quick demo: fit Prophet on debug crime data and save two PNGs:
- docs/figures/prophet_forecast_demo.png
- docs/figures/prophet_components_demo.png

Run:
python scripts/plot_prophet_demo.py
"""
import os
import sys
import matplotlib.pyplot as plt
from prophet import Prophet

# Ensure project root is on sys.path so imports like `backend.*` work when
# running the script directly (python scripts/plot_prophet_demo.py).
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.analytics.predict import debug_load_crime_data

OUT_DIR = os.path.join("docs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Load a small debug dataset (fast)
df = debug_load_crime_data(max_files=30, expand_to_daily=True)
if df.empty:
    raise SystemExit("No debug crime data available. Ensure Data/UK DATA CRIME... exists or increase max_files")

# Ensure columns are correct for Prophet
if 'ds' not in df.columns or 'y' not in df.columns:
    df = df.rename(columns={'date': 'ds', 'count': 'y'})

national_daily = df.groupby('ds')['y'].sum().reset_index()

m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
            changepoint_prior_scale=0.05, seasonality_prior_scale=10.0)

m.fit(national_daily)
future = m.make_future_dataframe(periods=90, freq='D')
forecast = m.predict(future)

# Forecast plot
fig1 = m.plot(forecast)
fig1.savefig(os.path.join(OUT_DIR, 'prophet_forecast_demo.png'), dpi=150)
plt.close(fig1)

# Components (trend + weekly/annual seasonality)
fig2 = m.plot_components(forecast)
fig2.savefig(os.path.join(OUT_DIR, 'prophet_components_demo.png'), dpi=150)
plt.close(fig2)

print(f"Saved forecast and components to {OUT_DIR}")
