import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np

import mlflow

from env.pricing_env import DynamicPricingEnv
from agents.ppo_agent import PPOAgent
from agents.dqn_agent import DQNAgent

from baselines.fixed import FixedPricingBaseline
from baselines.rule_based import RuleBasedPricingBaseline
from baselines.demand_model import DemandModelOptimizationBaseline
from baselines.moving_avg import MovingAveragePricingBaseline
from baselines.bandit import ContextualBanditBaseline

from utils.metrics import calculate_performance_metrics
from utils.plots import generate_evaluation_plots


# ---------------------------
# Run Single Episode
# ---------------------------
def run_episode(agent_name, agent, seed):

    env = DynamicPricingEnv()

    np.random.seed(seed)

    state = env.reset()

    done = False

    history = {
        "revenue": [],
        "sales": [],
        "demand": [],
        "inventory": [],
        "price": []
    }

    while not done:

        if agent_name == "PPO":

            action, _, _ = agent.select_action(state)

        elif agent_name == "DQN":

            action, _ = agent.select_action(
                state,
                eval_mode=True
            )

        else:

            action = agent.select_action(state)

        next_state, reward, done, info = env.step(action)

        if agent_name == "Contextual Bandit":

            agent.update(reward)

        for key in history:

            history[key].append(info[key])

        state = next_state

    return calculate_performance_metrics(history)


# ---------------------------
# Main Evaluation
# ---------------------------
def run_evaluation_suite(num_episodes=10):

    # =====================================================
    # DIRECTORIES
    # =====================================================
    os.makedirs("artifacts", exist_ok=True)

    os.makedirs(
        "artifacts/plots",
        exist_ok=True
    )

    # =====================================================
    # MLFLOW SETUP
    # =====================================================
    mlflow.set_tracking_uri(
        "file:./mlruns"
    )

    mlflow.set_experiment(
        "DynamicPricing-Evaluation"
    )

    with mlflow.start_run():

        # -------------------------------------------------
        # LOG PARAMS
        # -------------------------------------------------
        mlflow.log_param(
            "evaluation_episodes",
            num_episodes
        )

        # -------------------------------------------------
        # LOAD TRAINED AGENTS
        # -------------------------------------------------
        ppo_agent = PPOAgent()

        if os.path.exists(
            "artifacts/models/ppo_model.pt"
        ):

            ppo_agent.load(
                "artifacts/models/ppo_model.pt"
            )

        dqn_agent = DQNAgent()

        if os.path.exists(
            "artifacts/models/dqn_model.pt"
        ):

            dqn_agent.load(
                "artifacts/models/dqn_model.pt"
            )

        # -------------------------------------------------
        # MODELS
        # -------------------------------------------------
        models = {

            "PPO":
                ppo_agent,

            "DQN":
                dqn_agent,

            "Fixed Pricing":
                FixedPricingBaseline(),

            "Rule-Based":
                RuleBasedPricingBaseline(),

            "Demand Optimization":
                DemandModelOptimizationBaseline(),

            "Moving Average":
                MovingAveragePricingBaseline(),

            "Contextual Bandit":
                ContextualBanditBaseline()
        }

        global_results = {}

        print(
            "\nRunning evaluation "
            "over multiple episodes...\n"
        )

        # =================================================
        # EVALUATION LOOP
        # =================================================
        for name, agent in models.items():

            all_metrics = []

            for ep in range(num_episodes):

                metrics = run_episode(
                    name,
                    agent,
                    seed=ep
                )

                all_metrics.append(metrics)

            # ---------------------------------------------
            # AVERAGE METRICS
            # ---------------------------------------------
            avg_metrics = {}

            for key in all_metrics[0]:

                avg_metrics[key] = np.mean(
                    [m[key] for m in all_metrics]
                )

            global_results[name] = avg_metrics

            # ---------------------------------------------
            # LOG TO MLFLOW
            # ---------------------------------------------
            for metric_name, value in avg_metrics.items():

                metric_key = (
                    f"{name}_{metric_name}"
                )

                metric_key = (
                    metric_key
                    .replace(" ", "_")
                    .replace("-", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("%", "")
                    .lower()
                )

                mlflow.log_metric(
                    metric_key,
                    float(value)
                )

        # =================================================
        # SAVE RESULTS
        # =================================================
        evaluation_path = (
            "artifacts/evaluation_summary.json"
        )

        with open(
            evaluation_path,
            "w"
        ) as f:

            json.dump(
                global_results,
                f,
                indent=4
            )

        # =================================================
        # GENERATE PLOTS
        # =================================================
        saved_plots = generate_evaluation_plots(
            global_results
        )

        # =================================================
        # LOG ARTIFACTS
        # =================================================
        mlflow.log_artifact(
            evaluation_path
        )

        # -------------------------------------------------
        # LOG GENERATED PLOTS
        # -------------------------------------------------
        for plot_path in saved_plots:

            if os.path.exists(plot_path):

                mlflow.log_artifact(
                    plot_path
                )

        # =================================================
        # PRINT RESULTS
        # =================================================
        print("\n" + "=" * 55)

        print(
            "EVALUATION PERFORMANCE SCOREBOARD"
        )

        print("=" * 55)

        for model, metrics in global_results.items():

            print(f"\nModel: {model}")

            for metric_name, value in metrics.items():

                print(
                    f"  {metric_name}: "
                    f"{value:.2f}"
                )

        print("\n========================================")
        print("EVALUATION COMPLETE")
        print("========================================")

        print("\nSaved:")
        print("✓ Evaluation summary")
        print("✓ Evaluation plots")
        print("✓ MLflow evaluation logs")

        return global_results


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":

    run_evaluation_suite()