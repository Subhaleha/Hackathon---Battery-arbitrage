from ml_forecast import (
    load_data,
    create_features,
    train_model,
    forecast_next_24_hours
)

from optimizer import optimize_dispatch


df = load_data()

feature_df = create_features(df)

print("Training ML model...")

model, mae, rmse = train_model(feature_df)

print("ML model trained.")

start_index = 168 + 24

forecast = forecast_next_24_hours(
    df,
    model,
    start_index
)

predicted_prices = (
    forecast["predicted_price"].tolist()
)

print("\nRunning battery optimizer...")

dispatch = optimize_dispatch(
    predicted_prices
)

dispatch["ts"] = forecast["ts"].values

dispatch["predicted_price"] = (
    forecast["predicted_price"].values
)

dispatch["actual_price"] = (
    forecast["actual_price"].values
)


print("\n========== ML + OPTIMIZER ==========\n")

print(
    dispatch[
        [
            "ts",
            "predicted_price",
            "actual_price",
            "charge_mw",
            "discharge_mw",
            "soc_mwh"
        ]
    ].to_string(index=False)
)

print("\nML + Optimization pipeline completed!")