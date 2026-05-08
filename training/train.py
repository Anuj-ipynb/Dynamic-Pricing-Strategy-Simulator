import sys
import os
import json
import uuid
import random
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yaml
import torch

from env.pricing_env import DynamicPricingEnv
from agents.ppo_agent import PPOAgent
from agents.dqn_agent import DQNAgent


# =========================================================
# GLOBAL SEEDING
# =========================================================
def seed_everything(seed: int = 42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# GAE
# =========================================================
def compute_gae(
    rewards,
    values,
    dones,
    gamma,
    lam
):

    advantages = []
    gae = 0

    for step in reversed(range(len(rewards))):

        delta = (
            rewards[step]
            + gamma * values[step + 1] * (1 - dones[step])
            - values[step]
        )

        gae = (
            delta
            + gamma
            * lam
            * (1 - dones[step])
            * gae
        )

        advantages.insert(0, gae)

    returns = [
        adv + val
        for adv, val in zip(advantages, values[:-1])
    ]

    return returns, advantages


# =========================================================
# MOVING AVERAGE
# =========================================================
def moving_average(values, window=25):

    if len(values) < window:
        return np.mean(values)

    return np.mean(values[-window:])


# =========================================================
# TRAINING PIPELINE
# =========================================================
def execute_training_pipeline():

    # -----------------------------------------------------
    # DIRECTORIES
    # -----------------------------------------------------
    os.makedirs("artifacts/models", exist_ok=True)
    os.makedirs("artifacts/metadata", exist_ok=True)
    os.makedirs("artifacts/history", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)

    # -----------------------------------------------------
    # CONFIG
    # -----------------------------------------------------
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    seed = config["simulation"]["seed"]

    seed_everything(seed)

    total_episodes = config["training"]["episodes"]

    # -----------------------------------------------------
    # EXPERIMENT METADATA
    # -----------------------------------------------------
    run_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    ) + "_" + str(uuid.uuid4())[:8]

    experiment_metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "episodes": total_episodes,
        "ppo_config": config["ppo"],
        "dqn_config": config["dqn"],
        "simulation_config": config["simulation"]
    }

    with open(
        f"artifacts/metadata/{run_id}_metadata.json",
        "w"
    ) as f:
        json.dump(experiment_metadata, f, indent=4)

    # -----------------------------------------------------
    # ENVIRONMENT
    # -----------------------------------------------------
    env = DynamicPricingEnv()

    # -----------------------------------------------------
    # AGENTS
    # -----------------------------------------------------
    ppo = PPOAgent()
    dqn = DQNAgent()

    gamma = config["ppo"]["gamma"]
    lam = config["ppo"]["gae_lambda"]

    # -----------------------------------------------------
    # TRAINING TRACKERS
    # -----------------------------------------------------
    experiment_rows = []

    ppo_rewards = []
    ppo_revenues = []

    dqn_rewards = []
    dqn_revenues = []

    # =====================================================
    # PPO TRAINING
    # =====================================================
    print("\n================ PPO TRAINING ================\n")

    best_ppo_reward = -1e9

    for ep in range(total_episodes):

        state = env.reset()

        done = False

        ep_reward = 0
        ep_revenue = 0

        states = []
        actions = []
        log_probs = []

        rewards = []
        values = []
        dones = []

        price_trace = []
        inventory_trace = []

        regime_trace = []

        while not done:

            action, log_prob, value = ppo.select_action(state)

            action = np.clip(action, 5.0, 100.0)

            next_state, reward, done, info = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)

            rewards.append(reward)
            values.append(value)
            dones.append(float(done))

            ep_reward += reward
            ep_revenue += info["revenue"]

            price_trace.append(info["price"])
            inventory_trace.append(info["inventory"])

            regime_trace.append(info["regime"])

            state = next_state

        # -------------------------------------------------
        # FINAL BOOTSTRAP
        # -------------------------------------------------
        with torch.no_grad():

            state_t = (
                torch.FloatTensor(state)
                .unsqueeze(0)
                .to(ppo.device)
            )

            _, next_value = ppo.policy(state_t)

            next_value = next_value.item()

        values.append(next_value)

        # -------------------------------------------------
        # GAE
        # -------------------------------------------------
        returns, advantages = compute_gae(
            rewards,
            values,
            dones,
            gamma,
            lam
        )

        # -------------------------------------------------
        # PPO UPDATE
        # -------------------------------------------------
        ppo.update(
            states,
            actions,
            log_probs,
            returns,
            advantages
        )

        # -------------------------------------------------
        # TRACKING
        # -------------------------------------------------
        ppo_rewards.append(ep_reward)
        ppo_revenues.append(ep_revenue)

        reward_ma = moving_average(ppo_rewards)

        # -------------------------------------------------
        # SAVE BEST MODEL
        # -------------------------------------------------
        if ep_reward > best_ppo_reward:

            best_ppo_reward = ep_reward

            ppo.save(
                "artifacts/models/ppo_model.pt"
            )

        # -------------------------------------------------
        # EXPERIMENT ROW
        # -------------------------------------------------
        experiment_rows.append({

            "run_id": run_id,

            "episode": ep,
            "model": "PPO",

            "reward": ep_reward,
            "reward_moving_avg": reward_ma,

            "revenue": ep_revenue,

            "avg_price": float(np.mean(price_trace)),
            "price_volatility": float(np.std(price_trace)),

            "final_inventory": float(inventory_trace[-1]),

            "inventory_utilization": float(
                (
                    config["simulation"]["initial_inventory"]
                    - inventory_trace[-1]
                )
                /
                config["simulation"]["initial_inventory"]
                * 100.0
            ),

            "episode_length": len(price_trace),

            "trust_final": float(info["trust"]),

            "market_regime": max(
            set(regime_trace),
            key=regime_trace.count
                    ),

            "learning_rate": config["ppo"]["learning_rate"],
            "gamma": gamma
        })

        # -------------------------------------------------
        # LOGGING
        # -------------------------------------------------
        if (ep + 1) % 20 == 0:

            print(
                f"[PPO] Episode {ep+1}/{total_episodes}"
                f" | Reward: {ep_reward:.2f}"
                f" | Revenue: {ep_revenue:.2f}"
                f" | Avg(25): {reward_ma:.2f}"
                f" | Regime: {regime_trace[0]}"
            )

    # =====================================================
    # DQN TRAINING
    # =====================================================
    print("\n================ DQN TRAINING ================\n")

    best_dqn_reward = -1e9

    for ep in range(total_episodes):

        state = env.reset()

        done = False

        ep_reward = 0
        ep_revenue = 0

        price_trace = []
        inventory_trace = []

        regime_trace = []

        while not done:

            action_value, action_idx = dqn.select_action(state)

            next_state, reward, done, info = env.step(action_value)

            dqn.store_transition(
                state,
                action_idx,
                reward,
                next_state,
                done
            )

            dqn.update()

            ep_reward += reward
            ep_revenue += info["revenue"]

            price_trace.append(info["price"])
            inventory_trace.append(info["inventory"])

            regime_trace.append(info["regime"])

            state = next_state

        # -------------------------------------------------
        # TRACKING
        # -------------------------------------------------
        dqn_rewards.append(ep_reward)
        dqn_revenues.append(ep_revenue)

        reward_ma = moving_average(dqn_rewards)

        # -------------------------------------------------
        # SAVE BEST MODEL
        # -------------------------------------------------
        if ep_reward > best_dqn_reward:

            best_dqn_reward = ep_reward

            dqn.save(
                "artifacts/models/dqn_model.pt"
            )

        # -------------------------------------------------
        # EXPERIMENT ROW
        # -------------------------------------------------
        experiment_rows.append({

            "run_id": run_id,

            "episode": ep,
            "model": "DQN",

            "reward": ep_reward,
            "reward_moving_avg": reward_ma,

            "revenue": ep_revenue,

            "avg_price": float(np.mean(price_trace)),
            "price_volatility": float(np.std(price_trace)),

            "final_inventory": float(inventory_trace[-1]),

            "inventory_utilization": float(
                (
                    config["simulation"]["initial_inventory"]
                    - inventory_trace[-1]
                )
                /
                config["simulation"]["initial_inventory"]
                * 100.0
            ),

            "episode_length": len(price_trace),

            "trust_final": float(info["trust"]),

            "market_regime": regime_trace[0],

            "epsilon": float(dqn.epsilon),

            "learning_rate": config["dqn"]["learning_rate"],
            "gamma": config["dqn"]["gamma"]
        })

        # -------------------------------------------------
        # LOGGING
        # -------------------------------------------------
        if (ep + 1) % 20 == 0:

            print(
                f"[DQN] Episode {ep+1}/{total_episodes}"
                f" | Reward: {ep_reward:.2f}"
                f" | Revenue: {ep_revenue:.2f}"
                f" | Avg(25): {reward_ma:.2f}"
                f" | Epsilon: {dqn.epsilon:.4f}"
            )

    # =====================================================
    # SAVE HISTORY
    # =====================================================
    history_payload = {
        "run_id": run_id,

        "ppo_rewards": ppo_rewards,
        "ppo_revenues": ppo_revenues,

        "dqn_rewards": dqn_rewards,
        "dqn_revenues": dqn_revenues
    }

    with open(
        f"artifacts/history/{run_id}_training_history.json",
        "w"
    ) as f:
        json.dump(history_payload, f, indent=4)

    # =====================================================
    # SAVE EXPERIMENTS CSV
    # =====================================================
    df = pd.DataFrame(experiment_rows)

    df.to_csv(
        "experiments/results.csv",
        index=False
    )

    print("\n================================================")
    print("TRAINING COMPLETE")
    print("================================================")

    print(f"\nRun ID: {run_id}")

    print("\nSaved:")
    print("✓ PPO model")
    print("✓ DQN model")
    print("✓ Experiment CSV")
    print("✓ Training history")
    print("✓ Metadata snapshot")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    execute_training_pipeline()