import numpy as np

class ContextualBanditBaseline:
    def __init__(self, epsilon=0.15):
        self.epsilon = epsilon
        self.actions = np.linspace(10.0, 80.0, 8)
        self.counts = np.zeros(len(self.actions))
        self.q_values = np.zeros(len(self.actions))
        self.last_action_idx = 0

    def select_action(self, state):
        if np.random.random() < self.epsilon or np.sum(self.counts) == 0:
            self.last_action_idx = np.random.choice(len(self.actions))
        else:
            self.last_action_idx = np.argmax(self.q_values)
        return self.actions[self.last_action_idx]

    def update(self, reward):
        idx = self.last_action_idx
        self.counts[idx] += 1
        self.q_values[idx] += (reward - self.q_values[idx]) / self.counts[idx]
