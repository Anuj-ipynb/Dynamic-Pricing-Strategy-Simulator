import numpy as np
import pandas as pd


# =========================================================
# CORE METRICS
# =========================================================
def calculate_performance_metrics(history_dict):

    df = pd.DataFrame(history_dict)

    if len(df) == 0:

        return {
            "Total Revenue": 0.0,
            "Avg Revenue per Step": 0.0,
            "Total Sales Units": 0.0,
            "Mean Applied Price": 0.0,
            "Price Volatility (StdDev)": 0.0,
            "Reward Stability": 0.0,
            "Policy Smoothness": 0.0,
            "Final Terminal Inventory": 0.0,
            "Inventory Utilization (%)": 0.0,
            "Demand Fulfillment Rate (%)": 0.0,
            "Revenue Efficiency": 0.0
        }

    # -----------------------------------------------------
    # BASIC
    # -----------------------------------------------------
    total_revenue = df["revenue"].sum()

    total_sales = df["sales"].sum()

    total_demand = df["demand"].sum()

    avg_price = df["price"].mean()

    price_std = (
        df["price"].std()
        if len(df) > 1 else 0.0
    )

    final_inventory = df["inventory"].iloc[-1]

    avg_revenue_per_step = (
        total_revenue / len(df)
    )

    # -----------------------------------------------------
    # INVENTORY
    # -----------------------------------------------------
    inventory_used = total_sales

    initial_inventory = (
        total_sales + final_inventory
    )

    inventory_utilization = (
        (
            inventory_used
            / initial_inventory
        ) * 100.0
        if initial_inventory > 0 else 0.0
    )

    # -----------------------------------------------------
    # FULFILLMENT
    # -----------------------------------------------------
    demand_fulfillment = (
        (
            total_sales
            / total_demand
        ) * 100.0
        if total_demand > 0 else 0.0
    )

    # -----------------------------------------------------
    # POLICY SMOOTHNESS
    # -----------------------------------------------------
    if len(df) > 1:

        price_changes = np.diff(df["price"])

        policy_smoothness = (
            1.0 / (1.0 + np.mean(np.abs(price_changes)))
        )

    else:
        policy_smoothness = 1.0

    # -----------------------------------------------------
    # REWARD STABILITY
    # -----------------------------------------------------
    step_rewards = (
        df["sales"] * df["price"]
    )

    reward_std = np.std(step_rewards)

    reward_stability = float(
    np.exp(-reward_std / 1000.0)
    )

    # -----------------------------------------------------
    # REVENUE EFFICIENCY
    # -----------------------------------------------------
    revenue_efficiency = (
        total_revenue
        /
        (inventory_used + 1e-6)
    )

    # -----------------------------------------------------
    # FINAL METRICS
    # -----------------------------------------------------
    metrics = {

        "Total Revenue": float(total_revenue),

        "Avg Revenue per Step": float(
            avg_revenue_per_step
        ),

        "Total Sales Units": float(
            total_sales
        ),

        "Mean Applied Price": float(
            avg_price
        ),

        "Price Volatility (StdDev)": float(
            price_std
        ),

        "Reward Stability": float(
            reward_stability
        ),

        "Policy Smoothness": float(
            policy_smoothness
        ),

        "Final Terminal Inventory": float(
            final_inventory
        ),

        "Inventory Utilization (%)": float(
            inventory_utilization
        ),

        "Demand Fulfillment Rate (%)": float(
            demand_fulfillment
        ),

        "Revenue Efficiency": float(
            revenue_efficiency
        )
    }

    return metrics