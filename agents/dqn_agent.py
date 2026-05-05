import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import yaml
from collections import deque


# ---------------------------
# Q-Network
# ---------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim=5, action_dim=20):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        return self.net(state)


# ---------------------------
# DQN Agent
# ---------------------------
class DQNAgent:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)['dqn']

        # Device (CPU / GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Action space (price buckets)
        self.action_buckets = np.linspace(5.0, 100.0, self.cfg.get("num_actions", 20))
        self.action_dim = len(self.action_buckets)

        # Networks
        self.q_net = QNetwork(action_dim=self.action_dim).to(self.device)
        self.target_net = QNetwork(action_dim=self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.cfg['learning_rate'])

        # Replay buffer
        self.memory = deque(maxlen=self.cfg['memory_size'])

        # Exploration
        self.epsilon = self.cfg['epsilon_start']
        self.epsilon_end = self.cfg['epsilon_end']
        self.epsilon_decay = self.cfg['epsilon_decay']

        # Hyperparameters
        self.gamma = self.cfg['gamma']
        self.batch_size = self.cfg['batch_size']
        self.tau = self.cfg.get("tau", 0.005)  # for soft updates
        self.use_double_dqn = self.cfg.get("double_dqn", True)

    # ---------------------------
    # Action Selection
    # ---------------------------
    def select_action(self, state, eval_mode=False):
        if (not eval_mode) and (random.random() < self.epsilon):
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state_t)
                action_idx = torch.argmax(q_values, dim=1).item()

        return self.action_buckets[action_idx], action_idx

    # ---------------------------
    # Store Experience
    # ---------------------------
    def store_transition(self, state, action_idx, reward, next_state, done):
        self.memory.append((state, action_idx, reward, next_state, done))

    # ---------------------------
    # Training Step
    # ---------------------------
    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q values
        current_q = self.q_net(states).gather(1, actions)

        # Target Q values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN
                next_actions = self.q_net(next_states).argmax(1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions)
            else:
                # Standard DQN
                next_q = self.target_net(next_states).max(1, keepdim=True)[0]

            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Loss
        loss = nn.MSELoss()(current_q, target_q)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (stability)
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Soft update of target network
        self._soft_update()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    # ---------------------------
    # Soft Target Update
    # ---------------------------
    def _soft_update(self):
        for target_param, local_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    # ---------------------------
    # Save / Load
    # ---------------------------
    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.q_net.eval()