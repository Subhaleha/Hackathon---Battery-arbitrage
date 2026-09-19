import pandas as pd

from ml_forecast import (
    load_data,
    create_features,
    train_model,
    forecast_next_24_hours
)

from optimizer import optimize_dispatch


def run_closed_loop(df, model, start_index, days=7):

    results = []

    current_soc = 0.50

    for day in range(days):

        current_index = start_index + day * 24

        if current_index + 24 > len(df):
            break

        # ------------------------------------------
        # 1. Forecast next 24 hours
        # ------------------------------------------

        forecast = forecast_next_24_hours(
            df,
            model,
            current_index
        )

        predicted_prices = (
            forecast["predicted_price"].tolist()
        )

        # ------------------------------------------
        # 2. Optimize using forecast
        # ------------------------------------------

        dispatch = optimize_dispatch(
            predicted_prices,
            initial_soc=current_soc
        )

        # ------------------------------------------
        # 3. Execute only the FIRST action
        # ------------------------------------------

        first_action = dispatch.iloc[0]

        charge = first_action["charge_mw"]
        discharge = first_action["discharge_mw"]

        actual_price = (
            df.iloc[current_index]["price_per_mwh"]
        )

        # ------------------------------------------
        # 4. Calculate actual profit
        # ------------------------------------------

        revenue = actual_price * discharge

        charging_cost = actual_price * charge

        degradation_cost = (
            2 * (charge + discharge)
        )

        profit = (
            revenue
            - charging_cost
            - degradation_cost
        )

        # ------------------------------------------
        # 5. Update SOC
        # ------------------------------------------

        capacity = 100
        efficiency = 0.90

        current_soc = (
            current_soc * capacity
            + charge * efficiency
            - discharge / efficiency
        ) / capacity

        # Keep SOC within physical limits
        current_soc = max(
            0.20,
            min(0.90, current_soc)
        )

        results.append({
            "ts": df.iloc[current_index]["ts"],
            "actual_price": actual_price,
            "predicted_price": first_action["price"],
            "charge_mw": charge,
            "discharge_mw": discharge,
            "soc": current_soc,
            "profit": profit
        })

    return pd.DataFrame(results)


if __name__ == "__main__":

    print("Loading data...")

    df = load_data()

    print("Creating features...")

    feature_df = create_features(df)

    print("Training ML model...")

    model, mae, rmse = train_model(feature_df)

    print("\nML model trained.")

    start_index = 168 + 24

    print("\nRunning closed-loop backtest...")

    results = run_closed_loop(
        df,
        model,
        start_index,
        days=7
    )

    print("\n========== CLOSED-LOOP RESULTS ==========")

    print(
        results.to_string(index=False)
    )

    total_profit = results["profit"].sum()

    print("\nTotal closed-loop profit:",
          round(total_profit, 2))

    print(
        "\nClosed-loop backtest completed!"
    )

    results.to_csv(
        "closed_loop_results.csv",
        index=False
    )

    print(
        "Results saved to closed_loop_results.csv"
    )