import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_evaluation_plots(metrics_logs, save_dir="artifacts/plots/"):
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(10, 5))
    models = list(metrics_logs.keys())
    revenues = [metrics_logs[m]["Total Revenue"] for m in models]
    sns.barplot(x=models, y=revenues, palette="viridis")
    plt.title("Total Revenue Generation Comparison Matrix")
    plt.ylabel("Accumulated Revenue ($)")
    plt.xlabel("Tested Architectural Models")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "revenue_comparison.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    for model in metrics_logs:
        plt.scatter(
            metrics_logs[model]["Price Volatility (StdDev)"],
            metrics_logs[model]["Total Revenue"],
            s=200, label=model, alpha=0.8, edgecolors='w'
        )
    plt.title("System Stability Metrics: Volatility vs Cumulative Revenue Tradeoff")
    plt.xlabel("Price Volatility Vector (Standard Deviation)")
    plt.ylabel("Total Achieved Revenue ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "stability_tradeoff.png"))
    plt.close()
