from forecast import load_data, create_forecast
from optimizer import optimize_dispatch


# ---------------------------------------
# 1. Load historical data
# ---------------------------------------

df = load_data()

# Start after first 24 hours
start_index = 24

# ---------------------------------------
# 2. Generate 24-hour forecast
# ---------------------------------------

forecast = create_forecast(
    df,
    start_index
)

predicted_prices = forecast[
    "predicted_price"
].tolist()

# ---------------------------------------
# 3. Send forecast to optimizer
# ---------------------------------------

dispatch = optimize_dispatch(
    predicted_prices
)

# Add timestamps and actual prices
dispatch["ts"] = forecast["ts"].values
dispatch["actual_price"] = forecast[
    "actual_price"
].values

# ---------------------------------------
# 4. Display result
# ---------------------------------------

print("\n========== END-TO-END PIPELINE ==========\n")

print(
    dispatch[
        [
            "ts",
            "price",
            "actual_price",
            "charge_mw",
            "discharge_mw",
            "soc_mwh"
        ]
    ].to_string(index=False)
)

print("\nPipeline completed successfully!")