import pandas as pd
from forecast import load_data, create_forecast
from optimizer import optimize_dispatch


def calculate_profit(dispatch):
    revenue = (
        dispatch["actual_price"] * dispatch["discharge_mw"]
    ).sum()

    charging_cost = (
        dispatch["actual_price"] * dispatch["charge_mw"]
    ).sum()

    degradation_cost = (
        2 * (dispatch["charge_mw"] + dispatch["discharge_mw"])
    ).sum()

    return revenue - charging_cost - degradation_cost


def run_forecast_backtest(df, start_index=24, days=30):

    all_results = []

    for day in range(days):

        current_index = start_index + day * 24

        if current_index + 24 > len(df):
            break

        forecast = create_forecast(
            df,
            current_index
        )

        predicted_prices = (
            forecast["predicted_price"].tolist()
        )

        dispatch = optimize_dispatch(
            predicted_prices
        )

        dispatch["ts"] = forecast["ts"].values

        dispatch["actual_price"] = (
            forecast["actual_price"].values
        )

        dispatch["predicted_price"] = (
            forecast["predicted_price"].values
        )

        dispatch["forecast_error"] = (
            dispatch["actual_price"]
            - dispatch["predicted_price"]
        )

        daily_profit = calculate_profit(dispatch)

        # Store daily profit only for the first row
        # so it is not counted 24 times.
        dispatch["daily_profit"] = 0.0
        dispatch.loc[0, "daily_profit"] = daily_profit

        all_results.append(dispatch)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(
        all_results,
        ignore_index=True
    )


def calculate_perfect_foresight(
    df,
    start_index=24,
    days=30
):

    all_results = []

    for day in range(days):

        current_index = start_index + day * 24

        if current_index + 24 > len(df):
            break

        actual_prices = df.loc[
            current_index:current_index + 23,
            "price_per_mwh"
        ].tolist()

        dispatch = optimize_dispatch(
            actual_prices
        )

        dispatch["ts"] = df.loc[
            current_index:current_index + 23,
            "ts"
        ].values

        dispatch["actual_price"] = actual_prices

        profit = calculate_profit(dispatch)

        dispatch["daily_profit"] = 0.0
        dispatch.loc[0, "daily_profit"] = profit

        all_results.append(dispatch)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(
        all_results,
        ignore_index=True
    )


if __name__ == "__main__":

    df = load_data()

    print("Running forecast-based backtest...")

    forecast_results = run_forecast_backtest(
        df,
        start_index=24,
        days=30
    )

    print("Running perfect-foresight benchmark...")

    perfect_results = calculate_perfect_foresight(
        df,
        start_index=24,
        days=30
    )

    forecast_profit = (
        forecast_results["daily_profit"].sum()
    )

    perfect_profit = (
        perfect_results["daily_profit"].sum()
    )

    regret = (
        perfect_profit - forecast_profit
    )

    mae = (
        forecast_results["forecast_error"]
        .abs()
        .mean()
    )

    print("\n================================")
    print("BACKTEST RESULTS")
    print("================================")

    print(
        "Forecast-based profit:",
        round(forecast_profit, 2)
    )

    print(
        "Perfect-foresight profit:",
        round(perfect_profit, 2)
    )

    print(
        "Regret:",
        round(regret, 2)
    )

    print(
        "Mean Absolute Forecast Error:",
        round(mae, 2)
    )

    print("================================")

    forecast_results.to_csv(
        "forecast_backtest_results.csv",
        index=False
    )

    perfect_results.to_csv(
        "perfect_foresight_results.csv",
        index=False
    )

    print("\nResults saved successfully!")