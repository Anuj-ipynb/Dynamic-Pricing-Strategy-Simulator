import numpy as np
import yaml


class DynamicPricingEnv:
    def __init__(self, config_path="configs/config.yaml"):

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["simulation"]

        self.max_steps = self.config["max_steps"]
        self.initial_inventory = self.config["initial_inventory"]

        self.base_demand = self.config["base_demand"]
        self.base_elasticity = self.config["price_elasticity"]

        self.noise = self.config["demand_noise"]

        self.reset()

    # -------------------------------------------------
    # RESET
    # -------------------------------------------------
    def reset(self):

        self.current_step = 0
        self.inventory = self.initial_inventory

        self.last_price = 20.0

        self.price_history = [20.0, 20.0]
        self.demand_history = [0.0, 0.0]

        # Hidden market regime
        self.market_regime = np.random.choice([
            "high_demand",
            "recession",
            "price_sensitive",
            "loyal_customers"
        ])

        # Customer trust score
        self.customer_trust = 1.0

        return self._get_state()

    # -------------------------------------------------
    # STATE
    # -------------------------------------------------
    def _get_state(self):

        return np.array([
            self.last_price / 100.0,
            self.demand_history[-1] / (self.base_demand + 1e-6),
            self.demand_history[-2] / (self.base_demand + 1e-6),
            self.inventory / (self.initial_inventory + 1e-6),
            self.customer_trust
        ], dtype=np.float32)

    # -------------------------------------------------
    # MARKET REGIME DYNAMICS
    # -------------------------------------------------
    def _get_market_parameters(self):

        if self.market_regime == "high_demand":
            return 1.4, 1.2

        elif self.market_regime == "recession":
            return 0.7, 1.8

        elif self.market_regime == "price_sensitive":
            return 1.0, 2.5

        elif self.market_regime == "loyal_customers":
            return 1.1, 0.8

        return 1.0, self.base_elasticity

    # -------------------------------------------------
    # STEP
    # -------------------------------------------------
    def step(self, action_price):

        price = float(np.clip(action_price, 5.0, 100.0))

        demand_multiplier, elasticity = self._get_market_parameters()

        # Dynamic market cycle
        seasonal_factor = 1.0 + 0.2 * np.sin(
            2 * np.pi * self.current_step / 20.0
        )

        current_base_demand = (
            self.base_demand
            * demand_multiplier
            * seasonal_factor
            * self.customer_trust
        )

        # Demand model
        expected_demand = (
            current_base_demand
            - elasticity * price
        )

        expected_demand = max(0.0, expected_demand)

        # Stochasticity
        actual_demand = max(
            0.0,
            expected_demand + np.random.normal(0, self.noise)
        )

        # Inventory constraint
        actual_sales = min(actual_demand, self.inventory)

        self.inventory -= actual_sales

        # Revenue
        revenue = actual_sales * price

        # -------------------------------------------------
        # CUSTOMER TRUST DYNAMICS
        # -------------------------------------------------
        historical_avg_price = np.mean(self.price_history)

        if price > historical_avg_price * 1.25:
            self.customer_trust *= 0.97

        else:
            self.customer_trust *= 1.01

        self.customer_trust = np.clip(
            self.customer_trust,
            0.5,
            1.5
        )

        # -------------------------------------------------
        # REWARD
        # -------------------------------------------------

        revenue_reward = revenue / 100.0

        inventory_penalty = 0.003 * self.inventory

        volatility_penalty = (
            0.03 * abs(price - self.last_price)
        )

        stockout_penalty = (
            0.05 * max(0.0, actual_demand - actual_sales)
        )

        trust_reward = (
            2.0 * (self.customer_trust - 1.0)
        )

        reward = (
            revenue_reward
            + trust_reward
            - inventory_penalty
            - volatility_penalty
            - stockout_penalty
        )

        reward = np.clip(reward, -15.0, 15.0)

        # -------------------------------------------------
        # UPDATE HISTORY
        # -------------------------------------------------

        self.price_history.append(price)
        if len(self.price_history) > 5:
            self.price_history.pop(0)

        self.demand_history.append(actual_sales)
        if len(self.demand_history) > 5:
            self.demand_history.pop(0)

        self.last_price = price

        self.current_step += 1

        done = (
            self.current_step >= self.max_steps
            or self.inventory <= 0
        )

        info = {
            "revenue": revenue,
            "sales": actual_sales,
            "demand": actual_demand,
            "inventory": self.inventory,
            "price": price,
            "trust": self.customer_trust,
            "regime": self.market_regime
        }

        return self._get_state(), reward, done, info