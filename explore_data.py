import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/battery_prices.csv")

# Convert timestamp
df["ts"] = pd.to_datetime(df["ts"])

print("========== DATASET INFO ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\n========== PRICE INFORMATION ==========")
print("Minimum price:", df["price_per_mwh"].min())
print("Maximum price:", df["price_per_mwh"].max())
print("Average price:", round(df["price_per_mwh"].mean(), 2))

print("\n========== BATTERY DATA ==========")
print("Minimum load:", df["load_mw"].min())
print("Maximum load:", df["load_mw"].max())

print("Minimum renewable generation:",
      df["renewable_mw"].min())

print("Maximum renewable generation:",
      df["renewable_mw"].max())

# Plot price
plt.figure(figsize=(12, 5))

plt.plot(
    df["ts"],
    df["price_per_mwh"]
)

plt.title("Electricity Price Over Time")
plt.xlabel("Time")
plt.ylabel("Price per MWh")
plt.grid(True)

plt.tight_layout()
plt.show()