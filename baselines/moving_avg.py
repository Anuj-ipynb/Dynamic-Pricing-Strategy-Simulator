import numpy as np

class MovingAveragePricingBaseline:
    def __init__(self):
        pass

    def select_action(self, state):
        last_demand = state[1]
        prev_demand = state[2]
        current_normalized_price = state[0] * 100.0
        
        if last_demand > prev_demand:
            return float(np.clip(current_normalized_price * 1.05, 5.0, 100.0))
        return float(np.clip(current_normalized_price * 0.95, 5.0, 100.0))
