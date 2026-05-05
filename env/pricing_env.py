import numpy as np
import yaml

class DynamicPricingEnv:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['simulation']
            
        self.max_steps = self.config['max_steps']
        self.initial_inventory = self.config['initial_inventory']
        self.base_demand = self.config['base_demand']
        self.elasticity = self.config['price_elasticity']
        self.beta = self.config['memory_beta']
        self.noise = self.config['demand_noise']
        
        self.reset()

    def reset(self):
        self.current_step = 0
        self.inventory = self.initial_inventory
        self.price_history = [20.0, 20.0]
        self.demand_history = [0.0, 0.0]
        self.last_price = 20.0
        return self._get_state()

    def _get_state(self):
        avg_price = np.mean(self.price_history)
        return np.array([
            self.last_price / 100.0,
            self.demand_history[-1] / self.base_demand,
            self.demand_history[-2] / self.base_demand,
            self.inventory / self.initial_inventory,
            self.current_step / self.max_steps
        ], dtype=np.float32)

    def step(self, action_price):
        price = float(np.clip(action_price, 5.0, 100.0))
        
        time_factor = 1.0 + 0.3 * np.sin(2 * np.pi * self.current_step / 30.0)
        current_base = self.base_demand * time_factor
        
        demand_elast = current_base - self.elasticity * price
        
        historical_avg_price = np.mean(self.price_history)
        memory_penalty = np.exp(-self.beta * (price - historical_avg_price)) if price > historical_avg_price else 1.05
        
        expected_demand = max(0.0, demand_elast * memory_penalty)
        actual_demand = max(0.0, expected_demand + np.random.normal(0, self.noise))
        
        actual_sales = min(actual_demand, self.inventory)
        self.inventory = max(0.0, self.inventory - actual_sales)
        
        revenue = actual_sales * price
        
        stockout_penalty = 15.0 * max(0.0, actual_demand - actual_sales)
        overpricing_penalty = 5.0 * max(0.0, price - 2.5 * historical_avg_price)
        volatility_penalty = 2.0 * abs(price - self.last_price)
        customer_retention_reward = 10.0 if price <= historical_avg_price and actual_sales > 5 else 0.0
        
        reward = revenue - stockout_penalty - overpricing_penalty - volatility_penalty + customer_retention_reward
        
        self.price_history.append(price)
        if len(self.price_history) > 5:
            self.price_history.pop(0)
            
        self.demand_history.append(actual_sales)
        if len(self.demand_history) > 5:
            self.demand_history.pop(0)
            
        self.last_price = price
        self.current_step += 1
        done = (self.current_step >= self.max_steps) or (self.inventory <= 0)
        
        info = {
            "revenue": revenue,
            "sales": actual_sales,
            "demand": actual_demand,
            "inventory": self.inventory,
            "price": price
        }
        
        return self._get_state(), reward, done, info
