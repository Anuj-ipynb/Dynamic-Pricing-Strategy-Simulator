import numpy as np
import pandas as pd

def calculate_performance_metrics(history_dict):
    df = pd.DataFrame(history_dict)
    metrics = {
        "Total Revenue": float(df["revenue"].sum()),
        "Total Sales Units": float(df["sales"].sum()),
        "Mean Applied Price": float(df["price"].mean()),
        "Price Volatility (StdDev)": float(df["price"].std()) if len(df) > 1 else 0.0,
        "Final Terminal Inventory": float(df["inventory"].iloc[-1]) if len(df) > 0 else 0.0
    }
    return metrics
