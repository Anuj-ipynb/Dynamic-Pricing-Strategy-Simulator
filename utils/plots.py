import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# EVALUATION PLOTS
# =========================================================
def generate_evaluation_plots(
    metrics_logs,
    save_dir="artifacts/plots/"
):

    os.makedirs(save_dir, exist_ok=True)

    models = list(metrics_logs.keys())

    revenues = [
        metrics_logs[m]["Total Revenue"]
        for m in models
    ]

    volatility = [
        metrics_logs[m]["Price Volatility (StdDev)"]
        for m in models
    ]

    inventory_util = [
        metrics_logs[m]["Inventory Utilization (%)"]
        for m in models
    ]

    fulfillment = [
        metrics_logs[m]["Demand Fulfillment Rate (%)"]
        for m in models
    ]

    smoothness = [
        metrics_logs[m]["Policy Smoothness"]
        for m in models
    ]

    # =====================================================
    # REVENUE COMPARISON
    # =====================================================
    plt.figure(figsize=(12, 6))

    plt.bar(models, revenues)

    plt.title("Revenue Comparison Across Models")

    plt.ylabel("Total Revenue")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            "revenue_comparison.png"
        )
    )

    plt.close()

    # =====================================================
    # VOLATILITY VS REVENUE
    # =====================================================
    plt.figure(figsize=(8, 6))

    for i, model in enumerate(models):

        plt.scatter(
            volatility[i],
            revenues[i],
            s=180
        )

        plt.text(
            volatility[i],
            revenues[i],
            model
        )

    plt.xlabel("Price Volatility")

    plt.ylabel("Revenue")

    plt.title(
        "Revenue vs Price Volatility"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            "stability_tradeoff.png"
        )
    )

    plt.close()

    # =====================================================
    # INVENTORY UTILIZATION
    # =====================================================
    plt.figure(figsize=(12, 6))

    plt.bar(models, inventory_util)

    plt.title("Inventory Utilization")

    plt.ylabel("Utilization (%)")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            "inventory_utilization.png"
        )
    )

    plt.close()

    # =====================================================
    # DEMAND FULFILLMENT
    # =====================================================
    plt.figure(figsize=(12, 6))

    plt.bar(models, fulfillment)

    plt.title("Demand Fulfillment Rate")

    plt.ylabel("Fulfillment (%)")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            "demand_fulfillment.png"
        )
    )

    plt.close()

    # =====================================================
    # POLICY SMOOTHNESS
    # =====================================================
    plt.figure(figsize=(12, 6))

    plt.bar(models, smoothness)

    plt.title("Policy Smoothness Comparison")

    plt.ylabel("Smoothness Score")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            "policy_smoothness.png"
        )
    )

    plt.close()


# =========================================================
# TRAINING HISTORY PLOTS
# =========================================================
def generate_training_history_plots(
    history_path,
    save_dir="artifacts/plots/"
):

    os.makedirs(save_dir, exist_ok=True)

    with open(history_path, "r") as f:
        history = json.load(f)

    # -----------------------------------------------------
    # PPO REWARDS
    # -----------------------------------------------------
    ppo_rewards = history["ppo_rewards"]

    plt.figure(figsize=(12, 6))

    plt.plot(ppo_rewards)

    plt.title("PPO Training Reward Curve")

    plt.xlabel("Episode")

    plt.ylabel("Reward")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            "ppo_training_curve.png"
        )
    )

    plt.close()

    # -----------------------------------------------------
    # DQN REWARDS
    # -----------------------------------------------------
    dqn_rewards = history["dqn_rewards"]

    plt.figure(figsize=(12, 6))

    plt.plot(dqn_rewards)

    plt.title("DQN Training Reward Curve")

    plt.xlabel("Episode")

    plt.ylabel("Reward")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            "dqn_training_curve.png"
        )
    )

    plt.close()

    # -----------------------------------------------------
    # PPO vs DQN
    # -----------------------------------------------------
    plt.figure(figsize=(12, 6))

    plt.plot(
        ppo_rewards,
        label="PPO"
    )

    plt.plot(
        dqn_rewards,
        label="DQN"
    )

    plt.title("PPO vs DQN Reward Convergence")

    plt.xlabel("Episode")

    plt.ylabel("Reward")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            "ppo_vs_dqn_convergence.png"
        )
    )

    plt.close()