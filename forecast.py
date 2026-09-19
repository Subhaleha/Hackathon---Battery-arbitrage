import pandas as pd


def load_data():
    df = pd.read_csv("data/battery_prices.csv")
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def create_forecast(df, start_index):
    """
    Simple baseline forecast:
    predicted price for each hour = price from 24 hours earlier.
    """

    forecast = []

    for i in range(start_index, start_index + 24):

        previous_day_index = i - 24

        predicted_price = df.loc[
            previous_day_index,
            "price_per_mwh"
        ]

        forecast.append({
            "ts": df.loc[i, "ts"],
            "predicted_price": predicted_price,
            "actual_price": df.loc[i, "price_per_mwh"]
        })

    return pd.DataFrame(forecast)


if __name__ == "__main__":

    df = load_data()

    # Start after the first 24 hours
    start_index = 24

    forecast = create_forecast(df, start_index)

    print("\n========== 24-HOUR FORECAST ==========\n")
    print(forecast.to_string(index=False))