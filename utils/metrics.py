import numpy as np
import pandas as pd


def calculate_performance_metrics(history_dict):
    df = pd.DataFrame(history_dict)

    if len(df) == 0:
        return {
            "Total Revenue": 0.0,
            "Avg Revenue per Step": 0.0,
            "Total Sales Units": 0.0,
            "Mean Applied Price": 0.0,
            "Price Volatility (StdDev)": 0.0,
            "Final Terminal Inventory": 0.0,
            "Inventory Utilization (%)": 0.0,
            "Demand Fulfillment Rate (%)": 0.0
        }

    total_revenue = df["revenue"].sum()
    total_sales = df["sales"].sum()
    total_demand = df["demand"].sum()

    avg_price = df["price"].mean()
    price_std = df["price"].std() if len(df) > 1 else 0.0

    final_inventory = df["inventory"].iloc[-1]

    # ---------------------------
    # Additional meaningful metrics
    # ---------------------------

    avg_revenue_per_step = total_revenue / len(df)

    inventory_used = total_sales
    initial_inventory = total_sales + final_inventory

    inventory_utilization = (
        (inventory_used / initial_inventory) * 100.0
        if initial_inventory > 0 else 0.0
    )

    demand_fulfillment = (
        (total_sales / total_demand) * 100.0
        if total_demand > 0 else 0.0
    )

    metrics = {
        "Total Revenue": float(total_revenue),
        "Avg Revenue per Step": float(avg_revenue_per_step),
        "Total Sales Units": float(total_sales),
        "Mean Applied Price": float(avg_price),
        "Price Volatility (StdDev)": float(price_std),
        "Final Terminal Inventory": float(final_inventory),
        "Inventory Utilization (%)": float(inventory_utilization),
        "Demand Fulfillment Rate (%)": float(demand_fulfillment)
    }

    return metrics