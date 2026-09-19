import pandas as pd

from optimizer import optimize_dispatch


# ============================================================
# SCENARIO AGENT
# ============================================================

def run_scenario(predicted_prices, scenario_name):
    """
    Run the battery optimizer under a specific scenario.

    The agent can modify:
    - Price volatility assumption
    - Risk reserve
    - Degradation cost

    Physical battery constraints remain controlled by
    the optimizer.
    """

    prices = list(predicted_prices)

    # Default scenario parameters
    risk_reserve = 0.30
    degradation_multiplier = 1.0

    # --------------------------------------------------------
    # SCENARIO 1: NORMAL
    # --------------------------------------------------------

    if scenario_name == "Normal":

        risk_reserve = 0.30
        degradation_multiplier = 1.0

    # --------------------------------------------------------
    # SCENARIO 2: HIGH VOLATILITY
    # --------------------------------------------------------

    elif scenario_name == "High Volatility":

        risk_reserve = 0.40
        degradation_multiplier = 1.0

        average_price = sum(prices) / len(prices)

        # Increase deviations from the average price
        prices = [
            average_price
            + (price - average_price) * 1.30
            for price in prices
        ]

    # --------------------------------------------------------
    # SCENARIO 3: HIGH DEGRADATION COST
    # --------------------------------------------------------

    elif scenario_name == "High Degradation Cost":

        risk_reserve = 0.30
        degradation_multiplier = 2.0

    # --------------------------------------------------------
    # SCENARIO 4: CONSERVATIVE RISK
    # --------------------------------------------------------

    elif scenario_name == "Conservative Risk":

        risk_reserve = 0.50
        degradation_multiplier = 1.0

    # --------------------------------------------------------
    # UNKNOWN SCENARIO
    # --------------------------------------------------------

    else:

        raise ValueError(
            f"Unknown scenario: {scenario_name}"
        )

    # --------------------------------------------------------
    # DEGRADATION COST
    # --------------------------------------------------------

    base_degradation_cost = 2.0

    scenario_degradation_cost = (
        base_degradation_cost
        * degradation_multiplier
    )

    # --------------------------------------------------------
    # CALL OPTIMIZER
    # --------------------------------------------------------

    dispatch = optimize_dispatch(
        prices,
        initial_soc=0.50,
        risk_reserve=risk_reserve,
        degradation_cost=scenario_degradation_cost
    )

    # --------------------------------------------------------
    # ECONOMIC CALCULATION
    # --------------------------------------------------------

    original_prices = pd.Series(
        predicted_prices
    )

    revenue = (
        original_prices
        * dispatch["discharge_mw"]
    ).sum()

    charging_cost = (
        original_prices
        * dispatch["charge_mw"]
    ).sum()

    degradation_cost = (
        scenario_degradation_cost
        * (
            dispatch["charge_mw"]
            + dispatch["discharge_mw"]
        )
    ).sum()

    profit = (
        revenue
        - charging_cost
        - degradation_cost
    )

    # --------------------------------------------------------
    # SCENARIO RESULT
    # --------------------------------------------------------

    result = {
        "scenario": scenario_name,

        "profit": profit,

        "total_charge": (
            dispatch["charge_mw"].sum()
        ),

        "total_discharge": (
            dispatch["discharge_mw"].sum()
        ),

        "minimum_soc": (
            dispatch["soc_mwh"].min()
        ),

        "maximum_soc": (
            dispatch["soc_mwh"].max()
        ),

        "risk_reserve": risk_reserve,

        "degradation_cost": (
            scenario_degradation_cost
        ),

        "dispatch": dispatch
    }

    return result


# ============================================================
# RISK NOTE GENERATOR
# ============================================================

def generate_risk_note(result):

    scenario = result["scenario"]

    if scenario == "Normal":

        return (
            "Standard operating policy with a "
            "30% SOC reserve."
        )

    elif scenario == "High Volatility":

        return (
            "Higher price volatility is assumed. "
            "The optimizer maintains a larger SOC reserve "
            "to preserve flexibility for future price "
            "movements."
        )

    elif scenario == "High Degradation Cost":

        return (
            "Higher degradation cost increases the penalty "
            "for battery throughput and can reduce "
            "unnecessary cycling."
        )

    elif scenario == "Conservative Risk":

        return (
            "A 50% SOC reserve is enforced. This provides "
            "a larger operating buffer but reduces the "
            "energy available for arbitrage."
        )

    return "Scenario completed successfully."


# ============================================================
# RUN ALL SCENARIOS
# ============================================================

def run_agent(predicted_prices):

    scenarios = [
        "Normal",
        "High Volatility",
        "High Degradation Cost",
        "Conservative Risk"
    ]

    results = []

    for scenario in scenarios:

        result = run_scenario(
            predicted_prices,
            scenario
        )

        result["risk_note"] = (
            generate_risk_note(result)
        )

        results.append(result)

    return results


# ============================================================
# CREATE SUMMARY TABLE
# ============================================================

def create_summary(results):

    summary = []

    for result in results:

        summary.append({
            "Scenario": result["scenario"],

            "Profit": round(
                result["profit"],
                2
            ),

            "Total Charge (MWh)": round(
                result["total_charge"],
                2
            ),

            "Total Discharge (MWh)": round(
                result["total_discharge"],
                2
            ),

            "Minimum SOC (MWh)": round(
                result["minimum_soc"],
                2
            ),

            "Maximum SOC (MWh)": round(
                result["maximum_soc"],
                2
            ),

            "Risk Reserve (%)": round(
                result["risk_reserve"] * 100,
                0
            ),

            "Degradation Cost ($/MWh)": round(
                result["degradation_cost"],
                2
            ),

            "Risk Note": result["risk_note"]
        })

    return pd.DataFrame(summary)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("        BATTERY SCENARIO AGENT")
    print("=" * 60)

    # --------------------------------------------------------
    # Example 24-hour predicted prices
    # --------------------------------------------------------

    predicted_prices = [
        30, 25, 20, 25, 35, 40,
        50, 70, 100, 120, 90, 60,
        40, 30, 25, 20, 30, 50,
        80, 110, 130, 100, 60, 40
    ]

    print()
    print("Running scenarios...")
    print()

    # --------------------------------------------------------
    # Run agent
    # --------------------------------------------------------

    results = run_agent(
        predicted_prices
    )

    # --------------------------------------------------------
    # Display individual results
    # --------------------------------------------------------

    for result in results:

        print("-" * 60)

        print(
            "Scenario:",
            result["scenario"]
        )

        print(
            "Profit: $",
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
            "Maximum SOC:",
            round(
                result["maximum_soc"],
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
            "Degradation Cost: $",
            result["degradation_cost"],
            "/MWh"
        )

        print(
            "Risk Note:",
            result["risk_note"]
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = create_summary(
        results
    )

    print()
    print("=" * 60)
    print("                 SCENARIO SUMMARY")
    print("=" * 60)
    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 60)
    print("Agent execution completed successfully!")
    print("=" * 60)