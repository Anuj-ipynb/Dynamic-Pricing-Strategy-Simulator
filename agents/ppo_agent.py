import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import yaml


# ---------------------------
# Actor-Critic Network
# ---------------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim=5):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # FIX: Set initial log_std to 1.5 to provide a healthy initial exploration variance (~$4.48)
        self.log_std = nn.Parameter(torch.ones(1) * 1.5)

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        mean = self.actor(state)

        # Squash to [5, 100] price range
        mean = torch.sigmoid(mean) * 95.0 + 5.0

        std = torch.exp(self.log_std)
        dist = Normal(mean, std)

        value = self.critic(state)

        return dist, value

# ---------------------------
# PPO Agent
# ---------------------------
class PPOAgent:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)['ppo']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = ActorCritic().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg['learning_rate'])

        self.gamma = self.cfg['gamma']
        self.clip_epsilon = self.cfg['clip_epsilon']
        self.epochs = self.cfg['epochs']
        self.batch_size = self.cfg['batch_size']
        self.c1 = self.cfg['c1']
        self.c2 = self.cfg['c2']

    # ---------------------------
    # Action Selection
    # ---------------------------
    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist, value = self.policy(state_t)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        # Clamp action to valid range
        action = torch.clamp(action, 5.0, 100.0)

        return action.item(), log_prob.item(), value.item()

    # ---------------------------
    # PPO Update
    # ---------------------------
    def update(self, states, actions, log_probs, returns, advantages):

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(actions).unsqueeze(1).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).unsqueeze(1).to(self.device)
        returns = torch.FloatTensor(returns).unsqueeze(1).to(self.device)
        advantages = torch.FloatTensor(advantages).unsqueeze(1).to(self.device)

        # Normalize advantages (IMPORTANT)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = states.size(0)

        for _ in range(self.epochs):

            permutation = torch.randperm(dataset_size)

            for i in range(0, dataset_size, self.batch_size):
                idx = permutation[i:i + self.batch_size]

                dist, values = self.policy(states[idx])

                new_log_probs = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()

                # Ratio for PPO
                ratios = torch.exp(new_log_probs - old_log_probs[idx])

                surr1 = ratios * advantages[idx]
                surr2 = torch.clamp(
                    ratios,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon
                ) * advantages[idx]

                # Losses
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values, returns[idx])

                loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy

                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)

                self.optimizer.step()

    # ---------------------------
    # Save / Load
    # ---------------------------
    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy.eval()