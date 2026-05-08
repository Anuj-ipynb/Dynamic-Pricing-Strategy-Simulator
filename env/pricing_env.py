import numpy as np
import yaml


class DynamicPricingEnv:

    def __init__(self, config_path="configs/config.yaml"):

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        sim_cfg = cfg["simulation"]

        # =====================================================
        # CONFIG
        # =====================================================
        self.max_steps = sim_cfg["max_steps"]

        self.initial_inventory = sim_cfg["initial_inventory"]

        self.base_demand = sim_cfg["base_demand"]

        self.elasticity = sim_cfg["price_elasticity"]

        self.memory_beta = sim_cfg["memory_beta"]

        self.demand_noise = sim_cfg["demand_noise"]

        self.seed = sim_cfg["seed"]

        # =====================================================
        # RNG
        # =====================================================
        self.rng = np.random.default_rng(self.seed)

        # =====================================================
        # RESET
        # =====================================================
        self.reset()

    # =========================================================
    # RESET
    # =========================================================
    def reset(self):

        self.timestep = 0

        self.inventory = self.initial_inventory

        self.last_price = 35.0

        self.last_demand = self.base_demand

        self.prev_demand = self.base_demand

        self.customer_trust = 1.0

        self.regime = "normal"

        return self._get_state()

    # =========================================================
    # MARKET REGIME
    # =========================================================
    def _market_multiplier(self):

        cycle = np.sin(
            2 * np.pi * self.timestep / 50
        )

        if cycle > 0.45:

            self.regime = "peak"

            return 1.45

        elif cycle < -0.45:

            self.regime = "low"

            return 0.60

        self.regime = "normal"

        return 1.0

    # =========================================================
    # STATE
    # =========================================================
    def _get_state(self):

        normalized_price = (
            self.last_price / 100.0
        )

        normalized_last_demand = (
            self.last_demand / 100.0
        )

        normalized_prev_demand = (
            self.prev_demand / 100.0
        )

        normalized_inventory = (
            self.inventory
            / self.initial_inventory
        )

        normalized_timestep = (
            self.timestep
            / self.max_steps
        )

        return np.array([

            normalized_price,

            normalized_last_demand,

            normalized_prev_demand,

            normalized_inventory,

            normalized_timestep

        ], dtype=np.float32)

    # =========================================================
    # STEP
    # =========================================================
    def step(self, action):

        # -----------------------------------------------------
        # PRICE
        # -----------------------------------------------------
        price = float(
            np.clip(action, 5.0, 100.0)
        )

        # -----------------------------------------------------
        # REGIME
        # -----------------------------------------------------
        regime_multiplier = (
            self._market_multiplier()
        )

        # -----------------------------------------------------
        # CUSTOMER MEMORY
        # -----------------------------------------------------
        price_jump = abs(
            price - self.last_price
        )

        memory_penalty = np.exp(
            -self.memory_beta
            * (price_jump / 10.0)
        )

        # MUCH STRONGER TRUST DYNAMICS
        self.customer_trust = float(
            np.clip(

                0.75 * self.customer_trust
                + 0.25 * memory_penalty,

                0.05,
                1.0
            )
        )

        # -----------------------------------------------------
        # DEMAND
        # -----------------------------------------------------
        deterministic_demand = (

            self.base_demand

            * regime_multiplier

            * self.customer_trust

            * np.exp(
                -self.elasticity
                * (price / 100.0)
            )
        )

        # -----------------------------------------------------
        # PRICE COLLAPSE REGION
        # -----------------------------------------------------
        if price > 80.0:

            deterministic_demand *= 0.45

        # -----------------------------------------------------
        # EXTREME PRICE COLLAPSE
        # -----------------------------------------------------
        if price > 90.0:

            deterministic_demand *= 0.35

        # -----------------------------------------------------
        # NOISE
        # -----------------------------------------------------
        noisy_demand = (

            deterministic_demand

            + self.rng.normal(
                0,
                self.demand_noise
            )
        )

        demand = max(0.0, noisy_demand)

        # -----------------------------------------------------
        # SALES
        # -----------------------------------------------------
        sales = min(
            demand,
            self.inventory
        )

        revenue = sales * price

        # -----------------------------------------------------
        # INVENTORY
        # -----------------------------------------------------
        self.inventory -= sales

        # -----------------------------------------------------
        # PENALTIES
        # -----------------------------------------------------

        # stockout
        stockout_penalty = (
            max(0.0, demand - sales)
            * 6.0
        )

        # MUCH STRONGER VOLATILITY
        volatility_penalty = (
            abs(price - self.last_price)
            * 0.45
        )

        # NONLINEAR OVERPRICING
        overpricing_penalty = (

            max(0.0, price - 60.0) ** 2
        ) * 0.08

        # inventory holding
        holding_penalty = (
            self.inventory
            / self.initial_inventory
        ) * 2.0

        # -----------------------------------------------------
        # RETENTION
        # -----------------------------------------------------
        retention_reward = (
            self.customer_trust * 8.0
        )

        # -----------------------------------------------------
        # STRATEGIC INVENTORY PRESSURE
        # -----------------------------------------------------
        scarcity_penalty = 0.0

        if self.inventory < 0.15 * self.initial_inventory:

            scarcity_penalty = (
                (80.0 - price) * 2.0
            )

        # -----------------------------------------------------
        # FINAL REWARD
        # -----------------------------------------------------
        reward = (

            revenue

            - stockout_penalty

            - volatility_penalty

            - overpricing_penalty

            - holding_penalty

            - scarcity_penalty

            + retention_reward
        )

        # -----------------------------------------------------
        # UPDATE
        # -----------------------------------------------------
        self.prev_demand = self.last_demand

        self.last_demand = demand

        self.last_price = price

        self.timestep += 1

        # -----------------------------------------------------
        # DONE
        # -----------------------------------------------------
        done = (

            self.timestep >= self.max_steps

            or self.inventory <= 0
        )

        # -----------------------------------------------------
        # INFO
        # -----------------------------------------------------
        info = {

            "price": float(price),

            "demand": float(demand),

            "sales": float(sales),

            "inventory": float(self.inventory),

            "revenue": float(revenue),

            "reward": float(reward),

            "trust": float(self.customer_trust),

            "regime": self.regime,

            "timestep": int(self.timestep)
        }

        return (
            self._get_state(),
            reward,
            done,
            info
        )