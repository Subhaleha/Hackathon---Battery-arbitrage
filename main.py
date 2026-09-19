import pandas as pd

from ml_forecast import (
    load_data,
    create_features,
    train_model,
    forecast_next_24_hours
)

from agent import run_agent


# ============================================================
# MAIN END-TO-END PIPELINE
# ============================================================

def main():

    print()
    print("=" * 70)
    print("       BATTERY STORAGE PRICE-ARBITRAGE AGENT")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    print("\n[1/4] Loading market data...")

    df = load_data()

    print(
        f"Loaded {len(df):,} hourly observations."
    )

    # --------------------------------------------------------
    # 2. TRAIN ML MODEL
    # --------------------------------------------------------

    print("\n[2/4] Training LightGBM price forecasting model...")

    feature_df = create_features(df)

    model, mae, rmse = train_model(
        feature_df
    )

    print(
        f"ML Model MAE: {mae:.2f}"
    )

    print(
        f"ML Model RMSE: {rmse:.2f}"
    )

    # --------------------------------------------------------
    # 3. GENERATE 24-HOUR FORECAST
    # --------------------------------------------------------

    print("\n[3/4] Generating 24-hour price forecast...")

    # Need sufficient historical data
    # for 168-hour lag features.
    start_index = 168 + 24

    forecast = forecast_next_24_hours(
        df,
        model,
        start_index
    )

    predicted_prices = (
        forecast["predicted_price"]
        .tolist()
    )

    print(
        "\n24-hour forecast generated."
    )

    print(
        forecast[
            [
                "ts",
                "predicted_price",
                "actual_price"
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # 4. RUN SCENARIO AGENT
    # --------------------------------------------------------

    print("\n[4/4] Running scenario agent...")

    results = run_agent(
        predicted_prices
    )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("                    SCENARIO RESULTS")
    print("=" * 70)

    summary = []

    for result in results:

        print("\n" + "-" * 70)

        print(
            "Scenario:",
            result["scenario"]
        )

        print(
            "Estimated Profit: $",
            round(result["profit"], 2)
        )

        print(
            "Total Charge:",
            round(
                result["total_charge"],
                2
            ),
            "MWh"
        )

        print(
            "Total Discharge:",
            round(
                result["total_discharge"],
                2
            ),
            "MWh"
        )

        print(
            "Minimum SOC:",
            round(
                result["minimum_soc"],
                2
            ),
            "MWh"
        )

        print(
            "Risk Reserve:",
            result["risk_reserve"] * 100,
            "%"
        )

        print(
            "Risk Note:",
            result["risk_note"]
        )

        summary.append({
            "scenario": result["scenario"],
            "profit": result["profit"],
            "charge_mwh": result["total_charge"],
            "discharge_mwh": result["total_discharge"],
            "minimum_soc_mwh": result["minimum_soc"],
            "risk_reserve": result["risk_reserve"]
        })

    # --------------------------------------------------------
    # SAVE OUTPUT
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        summary
    )

    summary_df.to_csv(
        "scenario_results.csv",
        index=False
    )

    forecast.to_csv(
        "ml_forecast_results.csv",
        index=False
    )

    print("\n")
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        "\nSaved:"
    )

    print(
        "  → ml_forecast_results.csv"
    )

    print(
        "  → scenario_results.csv"
    )


if __name__ == "__main__":
    main()