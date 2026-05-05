class FixedPricingBaseline:
    def __init__(self, fixed_price=35.0):
        self.fixed_price = fixed_price

    def select_action(self, state):
        return self.fixed_price
