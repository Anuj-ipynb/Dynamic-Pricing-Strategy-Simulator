import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import yaml

class ActorCritic(nn.Module):
    def __init__(self, state_dim=5):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(1, 1))
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        mean = torch.sigmoid(self.actor(state)) * 95.0 + 5.0
        std = torch.exp(self.log_std)
        value = self.critic(state)
        return Normal(mean, std), value

class PPOAgent:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)['ppo']
        self.policy = ActorCritic()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg['learning_rate'])
        
    def select_action(self, state):
        state_t = torch.FloatTensor(state)
        dist, value = self.policy(state_t)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()
        
    def update(self, states, actions, log_probs, returns, advantages):
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.FloatTensor(np.array(actions)).unsqueeze(1)
        log_probs_t = torch.FloatTensor(np.array(log_probs)).unsqueeze(1)
        returns_t = torch.FloatTensor(np.array(returns)).unsqueeze(1)
        advantages_t = torch.FloatTensor(np.array(advantages)).unsqueeze(1)
        
        for _ in range(self.cfg['epochs']):
            permutation = torch.randperm(states_t.size(0))
            for i in range(0, states_t.size(0), self.cfg['batch_size']):
                idx = permutation[i:i+self.cfg['batch_size']]
                
                dist, value = self.policy(states_t[idx])
                new_log_probs = dist.log_prob(actions_t[idx])
                entropy = dist.entropy().mean()
                
                ratios = torch.exp(new_log_probs - log_probs_t[idx])
                surr1 = ratios * advantages_t[idx]
                surr2 = torch.clamp(ratios, 1.0 - self.cfg['clip_epsilon'], 1.0 + self.cfg['clip_epsilon']) * advantages_t[idx]
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(value, returns_t[idx])
                loss = actor_loss + self.cfg['c1'] * critic_loss - self.cfg['c2'] * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
    def save(self, path):
        torch.save(self.policy.state_dict(), path)
        
    def load(self, path):
        self.policy.load_state_dict(torch.load(path))
        self.policy.eval()
