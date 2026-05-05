import matplotlib.pyplot as plt
import seaborn as sns
import os


def generate_evaluation_plots(metrics_logs, save_dir="artifacts/plots/"):
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Convert dict → sorted lists
    models = list(metrics_logs.keys())
    revenues = [metrics_logs[m]["Total Revenue"] for m in models]
    volatility = [metrics_logs[m]["Price Volatility (StdDev)"] for m in models]
    inventory_util = [metrics_logs[m]["Inventory Utilization (%)"] for m in models]
    demand_fulfill = [metrics_logs[m]["Demand Fulfillment Rate (%)"] for m in models]

    # ---------------------------
    # 1. Revenue Comparison
    # ---------------------------
    plt.figure(figsize=(10, 5))
    sns.barplot(x=models, y=revenues, palette="viridis")
    plt.title("Total Revenue Comparison Across Models")
    plt.ylabel("Total Revenue")
    plt.xlabel("Models")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "revenue_comparison.png"))
    plt.close()

    # ---------------------------
    # 2. Stability vs Revenue
    # ---------------------------
    plt.figure(figsize=(8, 6))
    for i, model in enumerate(models):
        plt.scatter(
            volatility[i],
            revenues[i],
            s=150,
            label=model
        )
        plt.text(volatility[i], revenues[i], model)

    plt.title("Price Stability vs Revenue Trade-off")
    plt.xlabel("Price Volatility (StdDev)")
    plt.ylabel("Total Revenue")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "stability_tradeoff.png"))
    plt.close()

    # ---------------------------
    # 3. Inventory Utilization
    # ---------------------------
    plt.figure(figsize=(10, 5))
    sns.barplot(x=models, y=inventory_util, palette="magma")
    plt.title("Inventory Utilization Efficiency (%)")
    plt.ylabel("Utilization (%)")
    plt.xlabel("Models")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "inventory_utilization.png"))
    plt.close()

    # ---------------------------
    # 4. Demand Fulfillment
    # ---------------------------
    plt.figure(figsize=(10, 5))
    sns.barplot(x=models, y=demand_fulfill, palette="coolwarm")
    plt.title("Demand Fulfillment Rate (%)")
    plt.ylabel("Fulfillment (%)")
    plt.xlabel("Models")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "demand_fulfillment.png"))
    plt.close()