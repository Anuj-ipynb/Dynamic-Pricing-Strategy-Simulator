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


# ---------------------------
# Seed Everything
# ---------------------------
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------
# Compute GAE (clean version)
# ---------------------------
def compute_gae(rewards, values, masks, gamma, lam):
    returns = []
    gae = 0

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * gae * masks[step]
        returns.insert(0, gae + values[step])

    return returns


# ---------------------------
# Main Training Pipeline
# ---------------------------
def execute_training_pipeline():

    seed_everything()

    os.makedirs("artifacts/models", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)

    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    env = DynamicPricingEnv()
    ppo = PPOAgent()
    dqn = DQNAgent()

    total_episodes = config['training']['episodes']
    gamma = config['ppo']['gamma']
    lam = config['ppo'].get('gae_lambda', 0.95)

    experiment_rows = []

    # ===========================
    # PPO TRAINING
    # ===========================
    print("Training PPO Agent...")

    for ep in range(total_episodes):

        state = env.reset()
        done = False

        ep_reward = 0
        ep_revenue = 0

        states, actions, log_probs = [], [], []
        rewards, values, masks = [], [], []

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
            ep_revenue += info['revenue']

            state = next_state

        # Correct value for last state
        with torch.no_grad():
            next_state_t = torch.FloatTensor(state).unsqueeze(0).to(ppo.device)
            _, next_value = ppo.policy(next_state_t)
            next_value = next_value.item()

        values.append(next_value)

        # Compute returns using GAE
        returns = compute_gae(rewards, values, masks, gamma, lam)
        advantages = np.array(returns) - np.array(values[:-1])

        ppo.update(states, actions, log_probs, returns, advantages)

        experiment_rows.append({
            "run_id": f"ppo_ep_{ep}",
            "model": "PPO",
            "reward": ep_reward,
            "revenue": ep_revenue
        })

        if (ep + 1) % 20 == 0:
            print(f"PPO Episode {ep+1}/{total_episodes} | Reward: {ep_reward:.2f} | Revenue: {ep_revenue:.2f}")

    ppo.save("artifacts/models/ppo_model.pt")

    # ===========================
    # DQN TRAINING
    # ===========================
    print("\nTraining DQN Agent...")

    for ep in range(total_episodes):

        state = env.reset()
        done = False

        ep_reward = 0
        ep_revenue = 0

        while not done:
            action_val, action_idx = dqn.select_action(state)
            next_state, reward, done, info = env.step(action_val)

            dqn.store_transition(state, action_idx, reward, next_state, done)
            loss = dqn.update()

            ep_reward += reward
            ep_revenue += info['revenue']

            state = next_state

        experiment_rows.append({
            "run_id": f"dqn_ep_{ep}",
            "model": "DQN",
            "reward": ep_reward,
            "revenue": ep_revenue,
            "epsilon": dqn.epsilon
        })

        if (ep + 1) % 20 == 0:
            print(f"DQN Episode {ep+1}/{total_episodes} | Revenue: {ep_revenue:.2f} | Epsilon: {dqn.epsilon:.3f}")

    dqn.save("artifacts/models/dqn_model.pt")

    # ===========================
    # SAVE RESULTS
    # ===========================
    df = pd.DataFrame(experiment_rows)
    df.to_csv("experiments/results.csv", index=False)

    print("\nTraining complete. Models saved to /artifacts/models/")


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    execute_training_pipeline()