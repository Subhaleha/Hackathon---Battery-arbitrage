import pandas as pd
import numpy as np

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_data():
    df = pd.read_csv("data/battery_prices.csv")
    df["ts"] = pd.to_datetime(df["ts"])
    return df.sort_values("ts").reset_index(drop=True)


def create_features(df):

    data = df.copy()

    data["hour"] = data["ts"].dt.hour
    data["day_of_week"] = data["ts"].dt.dayofweek
    data["day_of_year"] = data["ts"].dt.dayofyear

    data["price_lag_1"] = data["price_per_mwh"].shift(1)
    data["price_lag_24"] = data["price_per_mwh"].shift(24)
    data["price_lag_48"] = data["price_per_mwh"].shift(48)
    data["price_lag_168"] = data["price_per_mwh"].shift(168)

    data["price_rolling_24"] = (
        data["price_per_mwh"]
        .shift(1)
        .rolling(24)
        .mean()
    )

    data["price_rolling_168"] = (
        data["price_per_mwh"]
        .shift(1)
        .rolling(168)
        .mean()
    )

    return data.dropna().reset_index(drop=True)


FEATURES = [
    "hour",
    "day_of_week",
    "day_of_year",
    "price_lag_1",
    "price_lag_24",
    "price_lag_48",
    "price_lag_168",
    "price_rolling_24",
    "price_rolling_168",
    "load_mw",
    "renewable_mw"
]


def train_model(df):

    X = df[FEATURES]
    y = df["price_per_mwh"]

    split = int(len(df) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    return model, mae, rmse


def forecast_next_24_hours(df, model, start_index):

    rows = []

    for i in range(start_index, start_index + 24):

        if i >= len(df):
            break

        row = df.iloc[i].copy()

        # Build features using information available
        # before the forecasted hour.
        row_data = {
            "hour": row["ts"].hour,
            "day_of_week": row["ts"].dayofweek,
            "day_of_year": row["ts"].dayofyear,
            "price_lag_1": df.iloc[i - 1]["price_per_mwh"],
            "price_lag_24": df.iloc[i - 24]["price_per_mwh"],
            "price_lag_48": df.iloc[i - 48]["price_per_mwh"],
            "price_lag_168": df.iloc[i - 168]["price_per_mwh"],
            "price_rolling_24": df.iloc[i - 24:i]["price_per_mwh"].mean(),
            "price_rolling_168": df.iloc[i - 168:i]["price_per_mwh"].mean(),
            "load_mw": row["load_mw"],
            "renewable_mw": row["renewable_mw"]
        }

        X = pd.DataFrame([row_data])[FEATURES]

        prediction = model.predict(X)[0]

        rows.append({
            "ts": row["ts"],
            "predicted_price": prediction,
            "actual_price": row["price_per_mwh"]
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":

    print("Loading data...")

    df = load_data()

    print("Creating features...")

    feature_df = create_features(df)

    print("Training LightGBM...")

    model, mae, rmse = train_model(feature_df)

    print("\n========== ML MODEL RESULTS ==========")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))

    # Need 168 previous hours for lag features
    start_index = 168 + 24

    forecast = forecast_next_24_hours(
        df,
        model,
        start_index
    )

    print("\n========== 24-HOUR ML FORECAST ==========")
    print(
        forecast.to_string(index=False)
    )