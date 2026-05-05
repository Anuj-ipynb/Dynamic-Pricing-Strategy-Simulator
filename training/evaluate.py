import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np

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
            action, _ = agent.select_action(state, eval_mode=True)

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

    # Load trained agents
    ppo_agent = PPOAgent()
    if os.path.exists("artifacts/models/ppo_model.pt"):
        ppo_agent.load("artifacts/models/ppo_model.pt")

    dqn_agent = DQNAgent()
    if os.path.exists("artifacts/models/dqn_model.pt"):
        dqn_agent.load("artifacts/models/dqn_model.pt")

    models = {
        "PPO": ppo_agent,
        "DQN": dqn_agent,
        "Fixed Pricing": FixedPricingBaseline(),
        "Rule-Based": RuleBasedPricingBaseline(),
        "Demand Optimization": DemandModelOptimizationBaseline(),
        "Moving Average": MovingAveragePricingBaseline(),
        "Contextual Bandit": ContextualBanditBaseline()
    }

    global_results = {}

    print("\nRunning evaluation over multiple episodes...\n")

    for name, agent in models.items():

        all_metrics = []

        for ep in range(num_episodes):
            metrics = run_episode(name, agent, seed=ep)
            all_metrics.append(metrics)

        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0]:
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        global_results[name] = avg_metrics

    # Save results
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/evaluation_summary.json", "w") as f:
        json.dump(global_results, f, indent=4)

    # Generate plots
    generate_evaluation_plots(global_results)

    # Print results
    print("\n" + "="*50)
    print("EVALUATION PERFORMANCE SCOREBOARD")
    print("="*50)

    for model, metrics in global_results.items():
        print(f"\nModel: {model}")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.2f}")

    return global_results


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    run_evaluation_suite()