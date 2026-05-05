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

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

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
    experiment_rows = []
    
    print("Executing Proximal Policy Optimization (PPO) Deep Training Loops...")
    for ep in range(total_episodes):
        state = env.reset()
        ep_reward, ep_revenue = 0, 0
        states, actions, log_probs, rewards, values, masks = [], [], [], [], [], []
        
        done = False
        while not done:
            action, log_prob, val = ppo.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(val)
            masks.append(1.0 - done)
            
            ep_reward += reward
            ep_revenue += info['revenue']
            state = next_state
            
        returns = []
        gae = 0
        next_value = 0 if done else ppo.select_action(state)[2]
        values = values + [next_value]
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + config['ppo']['gamma'] * values[step + 1] * masks[step] - values[step]
            gae = delta + config['ppo']['gamma'] * 0.95 * gae * masks[step]
            returns.insert(0, gae + values[step])
            
        advantages = np.array(returns) - np.array(values[:-1])
        ppo.update(states, actions, log_probs, returns, advantages)
        
        if (ep + 1) % 50 == 0:
            print(f"PPO Episode {ep+1}/{total_episodes} | Achieved Reward: {ep_reward:.2f} | Cumulative Revenue: ${ep_revenue:.2f}")

    ppo.save("artifacts/models/ppo_model.pt")
    experiment_rows.append({"run_id": "ppo_final", "model": "PPO", "reward": ep_reward, "revenue": ep_revenue, "parameters": str(config['ppo'])})

    print("\nExecuting Deep Q-Network (DQN) Training Modules...")
    for ep in range(total_episodes):
        state = env.reset()
        ep_reward, ep_revenue = 0, 0
        
        done = False
        while not done:
            action_val, action_idx = dqn.select_action(state)
            next_state, reward, done, info = env.step(action_val)
            
            dqn.store_transition(state, action_idx, reward, next_state, done)
            dqn.update()
            
            ep_reward += reward
            ep_revenue += info['revenue']
            state = next_state
            
        if (ep + 1) % 50 == 0:
            print(f"DQN Episode {ep+1}/{total_episodes} | Target Epsilon: {dqn.epsilon:.3f} | Cumulative Revenue: ${ep_revenue:.2f}")

    dqn.save("artifacts/models/dqn_model.pt")
    experiment_rows.append({"run_id": "dqn_final", "model": "DQN", "reward": ep_reward, "revenue": ep_revenue, "parameters": str(config['dqn'])})
    
    df = pd.DataFrame(experiment_rows)
    df.to_csv("experiments/results.csv", index=False)
    print("\nTraining completed. Models saved to /artifacts/models/")

if __name__ == "__main__":
    execute_training_pipeline()
