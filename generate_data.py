import pandas as pd
import numpy as np

# Reproducibility
np.random.seed(42)

# --------------------------------
# 1. Create 2 years of hourly data
# --------------------------------

timestamps = pd.date_range(
    start="2024-01-01",
    end="2025-12-31 23:00:00",
    freq="h"
)

n = len(timestamps)

df = pd.DataFrame({
    "ts": timestamps
})

# --------------------------------
# 2. Time features
# --------------------------------

hour = df["ts"].dt.hour
day_of_week = df["ts"].dt.dayofweek
day_of_year = df["ts"].dt.dayofyear

# --------------------------------
# 3. Electricity demand / load
# --------------------------------

base_load = 18000

daily_load = (
    3500 * np.sin((hour - 7) * 2 * np.pi / 24)
)

evening_peak = (
    3000 * np.exp(-((hour - 19) ** 2) / 10)
)

weekly_effect = np.where(
    day_of_week >= 5,
    -1500,
    0
)

noise_load = np.random.normal(0, 500, n)

df["load_mw"] = (
    base_load
    + daily_load
    + evening_peak
    + weekly_effect
    + noise_load
)

df["load_mw"] = df["load_mw"].clip(lower=8000)

# --------------------------------
# 4. Renewable generation
# --------------------------------

solar = (
    6000
    * np.maximum(
        0,
        np.sin((hour - 6) * np.pi / 12)
    )
)

wind = (
    2500
    + 1200 * np.sin(day_of_year * 2 * np.pi / 30)
    + np.random.normal(0, 600, n)
)

df["renewable_mw"] = solar + wind

df["renewable_mw"] = df["renewable_mw"].clip(lower=0)

# --------------------------------
# 5. Electricity price
# --------------------------------

# Base price
base_price = 70

# Daily price pattern
daily_pattern = (
    20 * np.sin((hour - 8) * 2 * np.pi / 24)
)

# Evening price spike
evening_price_peak = (
    35 * np.exp(-((hour - 19) ** 2) / 8)
)

# Weekend effect
weekend_effect = np.where(
    day_of_week >= 5,
    -10,
    0
)

# Fuel-cost drift over time
fuel_drift = np.linspace(0, 25, n)

# Renewable effect
renewable_effect = (
    -0.003 * df["renewable_mw"]
)

# Load effect
load_effect = (
    0.0015 * (df["load_mw"] - 15000)
)

# Random volatility
price_noise = np.random.normal(0, 12, n)

df["price_per_mwh"] = (
    base_price
    + daily_pattern
    + evening_price_peak
    + weekend_effect
    + fuel_drift
    + renewable_effect
    + load_effect
    + price_noise
)

# --------------------------------
# 6. Add occasional price spikes
# --------------------------------

spike_probability = 0.005

spikes = (
    np.random.random(n) < spike_probability
)

df.loc[spikes, "price_per_mwh"] += np.random.uniform(
    80,
    250,
    spikes.sum()
)

# --------------------------------
# 7. Add occasional negative prices
# --------------------------------

negative_probability = 0.003

negative_events = (
    np.random.random(n) < negative_probability
)

df.loc[negative_events, "price_per_mwh"] = np.random.uniform(
    -30,
    -5,
    negative_events.sum()
)

# --------------------------------
# 8. Round values
# --------------------------------

df["load_mw"] = df["load_mw"].round(2)
df["renewable_mw"] = df["renewable_mw"].round(2)
df["price_per_mwh"] = df["price_per_mwh"].round(2)

# --------------------------------
# 9. Save dataset
# --------------------------------

df.to_csv(
    "data/battery_prices.csv",
    index=False
)

print("Dataset created successfully!")
print()
print("Rows:", len(df))
print()
print(df.head())
print()
print(df.describe())