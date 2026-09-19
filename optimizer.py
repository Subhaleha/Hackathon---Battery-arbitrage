import pandas as pd
import pulp
import math


def load_battery_spec():

    spec = pd.read_csv(
        "data/battery_spec.csv"
    ).iloc[0]

    return {
        "capacity": float(spec["capacity_mwh"]),
        "power": float(spec["power_mw"]),
        "round_trip_efficiency": float(
            spec["round_trip_efficiency"]
        ),
        "soc_min": float(spec["soc_min"]),
        "soc_max": float(spec["soc_max"]),
        "degradation_cost": float(
            spec["cycle_cost_per_mwh"]
        )
    }


def optimize_dispatch(
    prices,
    initial_soc=0.50,
    risk_reserve=0.30,
    degradation_cost=None
):

    battery = load_battery_spec()

    capacity = battery["capacity"]
    power = battery["power"]

    round_trip_efficiency = (
        battery["round_trip_efficiency"]
    )

    # Convert round-trip efficiency into
    # efficiency for each direction.
    efficiency = math.sqrt(
        round_trip_efficiency
    )

    soc_min = (
        battery["soc_min"] * capacity
    )

    soc_max = (
        battery["soc_max"] * capacity
    )

    # Use battery specification's degradation
    # cost unless a scenario provides another value.
    if degradation_cost is None:
        degradation_cost = battery["degradation_cost"]

    # Risk reserve
    reserve = max(
        soc_min,
        risk_reserve * capacity
    )

    hours = len(prices)

    # --------------------------------------------------
    # CREATE OPTIMIZATION MODEL
    # --------------------------------------------------

    model = pulp.LpProblem(
        "Battery_Arbitrage",
        pulp.LpMaximize
    )

    # Charging power
    charge = pulp.LpVariable.dicts(
        "charge",
        range(hours),
        lowBound=0,
        upBound=power
    )

    # Discharging power
    discharge = pulp.LpVariable.dicts(
        "discharge",
        range(hours),
        lowBound=0,
        upBound=power
    )

    # State of charge
    soc = pulp.LpVariable.dicts(
        "soc",
        range(hours),
        lowBound=soc_min,
        upBound=soc_max
    )

    # --------------------------------------------------
    # OBJECTIVE FUNCTION
    # --------------------------------------------------

    model += pulp.lpSum(
        prices[t] * discharge[t]
        - prices[t] * charge[t]
        - degradation_cost
        * (charge[t] + discharge[t])
        for t in range(hours)
    )

    # Initial battery energy
    initial_energy = (
        initial_soc * capacity
    )

    # --------------------------------------------------
    # BATTERY SOC CONSTRAINT
    # --------------------------------------------------

    for t in range(hours):

        if t == 0:

            model += (
                soc[t]
                ==
                initial_energy
                + charge[t] * efficiency
                - discharge[t] / efficiency
            )

        else:

            model += (
                soc[t]
                ==
                soc[t - 1]
                + charge[t] * efficiency
                - discharge[t] / efficiency
            )

        # Risk reserve constraint
        model += (
            soc[t] >= reserve
        )

    # --------------------------------------------------
    # SOLVE
    # --------------------------------------------------

    model.solve(
        pulp.PULP_CBC_CMD(msg=False)
    )

    # --------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------

    results = []

    for t in range(hours):

        results.append({
            "hour": t,
            "price": prices[t],
            "charge_mw": charge[t].value(),
            "discharge_mw": discharge[t].value(),
            "soc_mwh": soc[t].value()
        })

    return pd.DataFrame(results)


# ======================================================
# TEST THE OPTIMIZER
# ======================================================

if __name__ == "__main__":

    prices = [
        30, 25, 20, 25, 35, 40,
        50, 70, 100, 120, 90, 60,
        40, 30, 25, 20, 30, 50,
        80, 110, 130, 100, 60, 40
    ]

    result = optimize_dispatch(
        prices,
        initial_soc=0.50,
        risk_reserve=0.30
    )

    print(
        "\n========== OPTIMAL DISPATCH ==========\n"
    )

    print(
        result.to_string(index=False)
    )