import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import yaml
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_dim=5, action_dim=20):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, state):
        return self.net(state)

class DQNAgent:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)['dqn']
            
        self.action_buckets = np.linspace(5.0, 100.0, 20)
        self.action_dim = len(self.action_buckets)
        
        self.q_net = QNetwork(action_dim=self.action_dim)
        self.target_net = QNetwork(action_dim=self.action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.cfg['learning_rate'])
        self.memory = deque(maxlen=self.cfg['memory_size'])
        self.epsilon = self.cfg['epsilon_start']
        self.steps_done = 0

    def select_action(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                action_idx = self.q_net(state_t).argmax(dim=1).item()
        return self.action_buckets[action_idx], action_idx

    def store_transition(self, s, a_idx, r, s_prime, done):
        self.memory.append((s, a_idx, r, s_prime, done))

    def update(self):
        if len(self.memory) < self.cfg['batch_size']:
            return
            
        batch = random.sample(self.memory, self.cfg['batch_size'])
        states, action_indices, rewards, next_states, dones = zip(*batch)
        
        states_t = torch.FloatTensor(np.array(states))
        action_indices_t = torch.LongTensor(action_indices).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(dones).unsqueeze(1)
        
        current_q = self.q_net(states_t).gather(1, action_indices_t)
        max_next_q = self.target_net(next_states_t).max(1)[0].unsqueeze(1)
        target_q = rewards_t + (1 - dones_t) * self.cfg['gamma'] * max_next_q
        
        loss = nn.MSELoss()(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps_done += 1
        if self.steps_done % self.cfg['target_update_freq'] == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            
        self.epsilon = max(self.cfg['epsilon_end'], self.epsilon * self.cfg['epsilon_decay'])

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.q_net.eval()
