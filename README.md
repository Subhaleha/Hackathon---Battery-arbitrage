Battery Storage Dispatch and Price-Arbitrage Agent

Overview

The Battery Storage Dispatch and Price-Arbitrage Agent is an AI/ML-based energy management system designed to optimize the operation of a grid-connected battery under changing electricity prices.
The system forecasts electricity prices using machine learning and then uses mathematical optimization to determine when the battery should charge or discharge. It also evaluates different operating scenarios and provides risk information to support operator decisions.

Problem Statement

Electricity prices can vary significantly throughout the day due to demand, renewable generation, fuel costs and market conditions.
A battery can potentially benefit from these price differences by charging when prices are relatively low and discharging when prices are relatively high.
However, battery operation is constrained by:
- Battery capacity
- Maximum charging and discharging power
- State of Charge (SOC)
- Charging and discharging efficiency
- Battery degradation
- Operational risk
The project aims to develop an intelligent system that considers these factors while planning battery dispatch.

Proposed Solution

Our system follows a multi-stage approach:
1. Historical electricity price data is prepared.
2. Time-based and historical price features are created.
3. A LightGBM model forecasts future electricity prices.
4. A PuLP-based optimization model generates the battery charging and discharging schedule.
5. Battery constraints, efficiency, degradation cost and SOC reserve are considered.
6. Different operating scenarios are evaluated.
7. Results are presented through a Streamlit dashboard.

System Architecture

Historical Data
      |
      v
Feature Engineering
      |
      v
LightGBM Price Forecast
      |
      v
Predicted Electricity Prices
      |
      v
PuLP Optimization
      |
      v
Battery Dispatch Plan
      |
      v
Scenario Analysis / Agent
      |
      v
Streamlit Dashboard
