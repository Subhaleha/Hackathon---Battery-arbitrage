import streamlit as st
import pandas as pd
import plotly.express as px

from ml_forecast import (
    load_data,
    create_features,
    train_model,
    forecast_next_24_hours
)

from agent import run_agent


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Battery Arbitrage Agent",
    page_icon="🔋",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🔋 Battery Storage Dispatch & Price-Arbitrage Agent")

st.markdown(
    """
    **AI/ML Energy Optimization System**

    The system forecasts electricity prices using LightGBM,
    optimizes battery dispatch using physical constraints,
    evaluates operating scenarios, and presents the results
    to the operator.
    """
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def get_data():

    df = load_data()

    return df


@st.cache_data
def train_ml_model(df):

    feature_df = create_features(df)

    model, mae, rmse = train_model(
        feature_df
    )

    return model, mae, rmse


df = get_data()


# ============================================================
# BATTERY SPECIFICATION
# ============================================================

spec = pd.read_csv(
    "data/battery_spec.csv"
).iloc[0]

capacity = float(
    spec["capacity_mwh"]
)

power = float(
    spec["power_mw"]
)

efficiency = float(
    spec["round_trip_efficiency"]
)

soc_min = float(
    spec["soc_min"]
)

soc_max = float(
    spec["soc_max"]
)

degradation = float(
    spec["cycle_cost_per_mwh"]
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🔋 Battery Configuration")

st.sidebar.write(
    f"**Capacity:** {capacity:.0f} MWh"
)

st.sidebar.write(
    f"**Power:** {power:.0f} MW"
)

st.sidebar.write(
    f"**Round-trip efficiency:** "
    f"{efficiency * 100:.0f}%"
)

st.sidebar.write(
    f"**SOC range:** "
    f"{soc_min * 100:.0f}% – "
    f"{soc_max * 100:.0f}%"
)

st.sidebar.write(
    f"**Degradation cost:** "
    f"${degradation:.2f}/MWh"
)


# ============================================================
# MARKET OVERVIEW
# ============================================================

st.header("📊 Market Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Data Points",
        f"{len(df):,}"
    )

with col2:

    st.metric(
        "Average Price",
        f"${df['price_per_mwh'].mean():.2f}/MWh"
    )

with col3:

    st.metric(
        "Minimum Price",
        f"${df['price_per_mwh'].min():.2f}/MWh"
    )

with col4:

    st.metric(
        "Maximum Price",
        f"${df['price_per_mwh'].max():.2f}/MWh"
    )


# ============================================================
# PRICE HISTORY
# ============================================================

st.subheader("Electricity Price History")

recent_data = df.tail(24 * 7)

price_chart = px.line(
    recent_data,
    x="ts",
    y="price_per_mwh",
    title="Electricity Prices — Recent 7 Days",
    labels={
        "ts": "Time",
        "price_per_mwh": "Price ($/MWh)"
    }
)

st.plotly_chart(
    price_chart,
    use_container_width=True
)


# ============================================================
# LOAD VS RENEWABLE
# ============================================================

st.subheader(
    "Electricity Load vs Renewable Generation"
)

market_chart = px.line(
    recent_data,
    x="ts",
    y=[
        "load_mw",
        "renewable_mw"
    ],
    labels={
        "ts": "Time",
        "value": "MW",
        "variable": "Metric"
    }
)

st.plotly_chart(
    market_chart,
    use_container_width=True
)


# ============================================================
# TRAIN ML MODEL
# ============================================================

st.header("🤖 ML Price Forecast")

with st.spinner(
    "Training LightGBM price forecasting model..."
):

    model, mae, rmse = train_ml_model(
        df
    )


col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Forecast MAE",
        f"{mae:.2f}"
    )

with col2:

    st.metric(
        "Forecast RMSE",
        f"{rmse:.2f}"
    )


# ============================================================
# GENERATE 24-HOUR FORECAST
# ============================================================

start_index = 168 + 24

forecast = forecast_next_24_hours(
    df,
    model,
    start_index
)

predicted_prices = (
    forecast[
        "predicted_price"
    ].tolist()
)


st.subheader(
    "24-Hour Electricity Price Forecast"
)

forecast_chart = px.line(
    forecast,
    x="ts",
    y=[
        "predicted_price",
        "actual_price"
    ],
    labels={
        "ts": "Time",
        "value": "Price ($/MWh)",
        "variable": "Price"
    },
    title="Predicted vs Actual Price"
)

st.plotly_chart(
    forecast_chart,
    use_container_width=True
)


# ============================================================
# FORECAST TABLE
# ============================================================

with st.expander(
    "View 24-hour forecast table"
):

    display_forecast = forecast.copy()

    display_forecast[
        "predicted_price"
    ] = display_forecast[
        "predicted_price"
    ].round(2)

    display_forecast[
        "actual_price"
    ] = display_forecast[
        "actual_price"
    ].round(2)

    st.dataframe(
        display_forecast,
        use_container_width=True
    )


# ============================================================
# SCENARIO AGENT
# ============================================================

st.header("🧠 Scenario Agent")

st.write(
    """
    The scenario agent evaluates different operating
    assumptions and sends them to the optimization model.
    """
)

with st.spinner(
    "Running battery scenarios..."
):

    scenario_results = run_agent(
        predicted_prices
    )


# ============================================================
# SCENARIO SUMMARY
# ============================================================

summary = []

for result in scenario_results:

    summary.append({

        "Scenario":
            result["scenario"],

        "Estimated Profit":
            round(
                result["profit"],
                2
            ),

        "Charge (MWh)":
            round(
                result["total_charge"],
                2
            ),

        "Discharge (MWh)":
            round(
                result["total_discharge"],
                2
            ),

        "Minimum SOC (MWh)":
            round(
                result["minimum_soc"],
                2
            ),

        "Risk Reserve":
            f"{result['risk_reserve'] * 100:.0f}%"

    })


summary_df = pd.DataFrame(
    summary
)


st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SCENARIO PROFIT CHART
# ============================================================

profit_chart = px.bar(
    summary_df,
    x="Scenario",
    y="Estimated Profit",
    title="Estimated Profit by Scenario"
)

st.plotly_chart(
    profit_chart,
    use_container_width=True
)


# ============================================================
# SELECT SCENARIO
# ============================================================

st.subheader(
    "🔎 Scenario Details"
)

scenario_names = [
    result["scenario"]
    for result in scenario_results
]

selected_scenario = st.selectbox(
    "Select a scenario",
    scenario_names
)


selected_result = next(
    result
    for result in scenario_results
    if result["scenario"]
    == selected_scenario
)


# ============================================================
# SELECTED SCENARIO METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Estimated Profit",
        f"${selected_result['profit']:,.2f}"
    )

with col2:

    st.metric(
        "Total Charge",
        f"{selected_result['total_charge']:.2f} MWh"
    )

with col3:

    st.metric(
        "Total Discharge",
        f"{selected_result['total_discharge']:.2f} MWh"
    )

with col4:

    st.metric(
        "Minimum SOC",
        f"{selected_result['minimum_soc']:.2f} MWh"
    )


# ============================================================
# RISK NOTE
# ============================================================

st.subheader("⚠️ Operator Risk Note")

st.info(
    selected_result["risk_note"]
)


# ============================================================
# DISPATCH PLAN
# ============================================================

st.subheader(
    "⚡ Battery Dispatch Plan"
)

dispatch = selected_result[
    "dispatch"
].copy()

dispatch["ts"] = forecast["ts"].values


dispatch_chart = px.line(
    dispatch,
    x="ts",
    y=[
        "charge_mw",
        "discharge_mw"
    ],
    title="Battery Charge / Discharge Schedule",
    labels={
        "ts": "Time",
        "value": "Power (MW)",
        "variable": "Action"
    }
)

st.plotly_chart(
    dispatch_chart,
    use_container_width=True
)


# ============================================================
# SOC CHART
# ============================================================

st.subheader(
    "🔋 Battery State of Charge"
)

soc_chart = px.line(
    dispatch,
    x="ts",
    y="soc_mwh",
    title="Battery SOC During Dispatch",
    labels={
        "ts": "Time",
        "soc_mwh": "SOC (MWh)"
    }
)

st.plotly_chart(
    soc_chart,
    use_container_width=True
)


# ============================================================
# DISPATCH TABLE
# ============================================================

with st.expander(
    "View detailed dispatch plan"
):

    st.dataframe(
        dispatch,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SYSTEM ARCHITECTURE
# ============================================================

st.header(
    "🏗️ System Architecture"
)

st.code(
    """
Historical Energy Data
        │
        ▼
Feature Engineering
        │
        ▼
LightGBM Price Forecast
        │
        ▼
24-Hour Price Prediction
        │
        ▼
Scenario Agent
        │
        ▼
PuLP Battery Optimizer
        │
        ├── Capacity Constraint
        ├── Power Constraint
        ├── SOC Constraint
        ├── Efficiency
        ├── Degradation Cost
        └── Risk Reserve
        │
        ▼
Battery Dispatch
        │
        ▼
Operator Dashboard
    """,
    language="text"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Battery Storage Dispatch & Price-Arbitrage Agent | "
    "AIML Campus Drive 2026"
)