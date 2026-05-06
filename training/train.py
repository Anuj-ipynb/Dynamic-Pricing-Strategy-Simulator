import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yaml
import torch

from env.pricing_env import DynamicPricingEnv
from agents.ppo_agent import PPOAgent
from agents.dqn_agent import DQNAgent


# ---------------------------------------------------
# SEEDING
# ---------------------------------------------------
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------
# GENERALIZED ADVANTAGE ESTIMATION
# ---------------------------------------------------
def compute_gae(rewards, values, masks, gamma, lam):

    returns = []
    gae = 0

    for step in reversed(range(len(rewards))):

        delta = (
            rewards[step]
            + gamma * values[step + 1] * masks[step]
            - values[step]
        )

        gae = (
            delta
            + gamma * lam * masks[step] * gae
        )

        returns.insert(0, gae + values[step])

    return returns


# ---------------------------------------------------
# REWARD NORMALIZATION
# ---------------------------------------------------
def normalize_rewards(rewards):

    rewards = np.array(rewards)

    return (
        (rewards - rewards.mean())
        / (rewards.std() + 1e-8)
    )


# ---------------------------------------------------
# TRAINING PIPELINE
# ---------------------------------------------------
def execute_training_pipeline():

    seed_everything()

    os.makedirs("artifacts/models", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)

    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    total_episodes = config["training"]["episodes"]

    env = DynamicPricingEnv()

    ppo = PPOAgent()
    dqn = DQNAgent()

    gamma = config["ppo"]["gamma"]
    lam = config["ppo"]["gae_lambda"]

    experiment_rows = []

    # =================================================
    # PPO TRAINING
    # =================================================
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
        masks = []

        regime_trace = []

        while not done:

            action, log_prob, value = ppo.select_action(state)

            next_state, reward, done, info = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)

            rewards.append(reward)
            values.append(value)
            masks.append(1.0 - done)

            ep_reward += reward
            ep_revenue += info["revenue"]

            regime_trace.append(info["regime"])

            state = next_state

        # ---------------------------------------------
        # Final value bootstrap
        # ---------------------------------------------
        with torch.no_grad():

            next_state_t = (
                torch.FloatTensor(state)
                .unsqueeze(0)
                .to(ppo.device)
            )

            _, next_value = ppo.policy(next_state_t)

            next_value = next_value.item()

        values.append(next_value)

        # ---------------------------------------------
        # Normalize rewards
        # ---------------------------------------------
        normalized_rewards = normalize_rewards(rewards)

        # ---------------------------------------------
        # Compute returns + advantages
        # ---------------------------------------------
        returns = compute_gae(
            normalized_rewards,
            values,
            masks,
            gamma,
            lam
        )

        advantages = (
            np.array(returns)
            - np.array(values[:-1])
        )

        # ---------------------------------------------
        # PPO update
        # ---------------------------------------------
        ppo.update(
            states,
            actions,
            log_probs,
            returns,
            advantages
        )

        # ---------------------------------------------
        # Save best PPO
        # ---------------------------------------------
        if ep_reward > best_ppo_reward:

            best_ppo_reward = ep_reward

            ppo.save(
                "artifacts/models/ppo_model.pt"
            )

        # ---------------------------------------------
        # Logging
        # ---------------------------------------------
        experiment_rows.append({
            "run_id": f"ppo_ep_{ep}",
            "model": "PPO",
            "reward": ep_reward,
            "revenue": ep_revenue,
            "regime": regime_trace[0],
            "trust_final": info["trust"]
        })

        if (ep + 1) % 20 == 0:

            print(
                f"PPO Episode {ep+1}/{total_episodes}"
                f" | Reward: {ep_reward:.2f}"
                f" | Revenue: {ep_revenue:.2f}"
                f" | Regime: {regime_trace[0]}"
                f" | Trust: {info['trust']:.2f}"
            )

    # =================================================
    # DQN TRAINING
    # =================================================
    print("\n================ DQN TRAINING ================\n")

    best_dqn_reward = -1e9

    for ep in range(total_episodes):

        state = env.reset()

        done = False

        ep_reward = 0
        ep_revenue = 0

        regime_trace = []

        while not done:

            action_val, action_idx = dqn.select_action(state)

            next_state, reward, done, info = env.step(action_val)

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

            regime_trace.append(info["regime"])

            state = next_state

        if ep_reward > best_dqn_reward:

            best_dqn_reward = ep_reward

            dqn.save(
                "artifacts/models/dqn_model.pt"
            )

        experiment_rows.append({
            "run_id": f"dqn_ep_{ep}",
            "model": "DQN",
            "reward": ep_reward,
            "revenue": ep_revenue,
            "epsilon": dqn.epsilon,
            "regime": regime_trace[0],
            "trust_final": info["trust"]
        })

        if (ep + 1) % 20 == 0:

            print(
                f"DQN Episode {ep+1}/{total_episodes}"
                f" | Reward: {ep_reward:.2f}"
                f" | Revenue: {ep_revenue:.2f}"
                f" | Epsilon: {dqn.epsilon:.3f}"
                f" | Regime: {regime_trace[0]}"
            )

    # =================================================
    # SAVE EXPERIMENTS
    # =================================================
    df = pd.DataFrame(experiment_rows)

    df.to_csv(
        "experiments/results.csv",
        index=False
    )

    print("\nTraining complete.")
    print("Models saved to artifacts/models/")
    print("Experiment logs saved to experiments/results.csv")


# ---------------------------------------------------
# RUN
# ---------------------------------------------------
if __name__ == "__main__":
    execute_training_pipeline()