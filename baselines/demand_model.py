import numpy as np

class DemandModelOptimizationBaseline:
    def __init__(self, base_demand=50, elasticity=1.5):
        self.base_demand = base_demand
        self.elasticity = elasticity

    def select_action(self, state):
        opt_price = self.base_demand / (2 * self.elasticity)
        return float(np.clip(opt_price, 5.0, 100.0))
