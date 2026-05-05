import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import yaml


class ActorCritic(nn.Module):
    def __init__(self, state_dim=5):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.log_std = nn.Parameter(torch.zeros(1))  # FIXED (was too large)

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        mean = torch.sigmoid(self.actor(state)) * 95.0 + 5.0
        std = torch.exp(self.log_std).clamp(1e-3, 10.0)
        dist = Normal(mean, std)
        value = self.critic(state)
        return dist, value


class PPOAgent:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)['ppo']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = ActorCritic().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg['learning_rate'])

        self.clip = self.cfg['clip_epsilon']
        self.c1 = self.cfg['c1']
        self.c2 = self.cfg['c2']
        self.epochs = self.cfg['epochs']
        self.batch_size = self.cfg['batch_size']

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, value = self.policy(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def update(self, states, actions, old_log_probs, returns, advantages):

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).unsqueeze(1).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).unsqueeze(1).to(self.device)
        returns = torch.FloatTensor(returns).unsqueeze(1).to(self.device)
        advantages = torch.FloatTensor(advantages).unsqueeze(1).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            idx = torch.randperm(states.size(0))

            for i in range(0, states.size(0), self.batch_size):
                batch = idx[i:i+self.batch_size]

                dist, value = self.policy(states[batch])
                new_log_probs = dist.log_prob(actions[batch])

                ratio = torch.exp(new_log_probs - old_log_probs[batch])

                surr1 = ratio * advantages[batch]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages[batch]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (returns[batch] - value).pow(2).mean()

                loss = actor_loss + self.c1 * critic_loss - self.c2 * dist.entropy().mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy.eval()