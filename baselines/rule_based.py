class RuleBasedPricingBaseline:
    def __init__(self):
        pass

    def select_action(self, state):
        last_demand = state[1]
        inventory = state[3]
        
        if inventory < 0.2:
            return 75.0
        if last_demand > 0.6:
            return 45.0
        return 30.0
