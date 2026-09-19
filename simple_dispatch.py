import pandas as pd
from battery import Battery

# Load data
df = pd.read_csv("data/battery_prices.csv")
df["ts"] = pd.to_datetime(df["ts"])

# Use first 24 hours for our first test
df = df.head(24).copy()

# Create battery
battery = Battery(
    capacity_mwh=100,
    power_mw=20,
    efficiency=0.90,
    soc_min=0.20,
    soc_max=0.90,
    initial_soc=0.50,
    degradation_cost=2
)

# Calculate cheap and expensive price thresholds
cheap_price = df["price_per_mwh"].quantile(0.30)
expensive_price = df["price_per_mwh"].quantile(0.70)

print("Cheap price threshold:", cheap_price)
print("Expensive price threshold:", expensive_price)

results = []

total_profit = 0

for _, row in df.iterrows():

    price = row["price_per_mwh"]

    charge = 0
    discharge = 0
    revenue = 0
    energy_cost = 0
    degradation_cost = 0
    action = "IDLE"

    # Cheap electricity → charge
    if price <= cheap_price:

        charge = battery.power_mw

        stored = battery.charge(charge)

        energy_cost = charge * price

        degradation_cost = stored * battery.degradation_cost

        total_profit -= energy_cost
        total_profit -= degradation_cost

        action = "CHARGE"

    # Expensive electricity → discharge
    elif price >= expensive_price:

        discharge = battery.power_mw

        delivered = battery.discharge(discharge)

        revenue = delivered * price

        degradation_cost = delivered * battery.degradation_cost

        total_profit += revenue
        total_profit -= degradation_cost

        action = "DISCHARGE"

    results.append({
        "ts": row["ts"],
        "price": price,
        "action": action,
        "charge_mw": charge,
        "discharge_mw": discharge,
        "soc_mwh": battery.soc_mwh,
        "revenue": revenue,
        "energy_cost": energy_cost,
        "degradation_cost": degradation_cost
    })


results_df = pd.DataFrame(results)

print("\n========== DISPATCH ==========")
print(
    results_df[
        [
            "ts",
            "price",
            "action",
            "charge_mw",
            "discharge_mw",
            "soc_mwh"
        ]
    ].to_string(index=False)
)

print("\n========== RESULT ==========")
print("Total profit:", round(total_profit, 2))