import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from env.pricing_env import DynamicPricingEnv
from agents.ppo_agent import PPOAgent
from agents.dqn_agent import DQNAgent
from baselines.fixed import FixedPricingBaseline
from baselines.rule_based import RuleBasedPricingBaseline
from baselines.demand_model import DemandModelOptimizationBaseline
from baselines.moving_avg import MovingAveragePricingBaseline
from baselines.bandit import ContextualBanditBaseline
from utils.metrics import calculate_performance_metrics
from utils.plots import generate_evaluation_plots

def run_evaluation_suite():
    env = DynamicPricingEnv()
    
    ppo_agent = PPOAgent()
    if os.path.exists("artifacts/models/ppo_model.pt"):
        ppo_agent.load("artifacts/models/ppo_model.pt")
        
    dqn_agent = DQNAgent()
    if os.path.exists("artifacts/models/dqn_model.pt"):
        dqn_agent.load("artifacts/models/dqn_model.pt")

    models = {
        "PPO": ppo_agent,
        "DQN": dqn_agent,
        "Fixed Pricing": FixedPricingBaseline(),
        "Rule-Based": RuleBasedPricingBaseline(),
        "Demand Optimization": DemandModelOptimizationBaseline(),
        "Moving Average": MovingAveragePricingBaseline(),
        "Contextual Bandit": ContextualBanditBaseline()
    }
    
    global_results = {}
    
    for name, agent in models.items():
        state = env.reset()
        done = False
        
        history = {"revenue": [], "sales": [], "demand": [], "inventory": [], "price": []}
        
        while not done:
            if name == "PPO":
                action, _, _ = agent.select_action(state)
            elif name == "DQN":
                action, _ = agent.select_action(state, eval_mode=True)
            else:
                action = agent.select_action(state)
                
            next_state, reward, done, info = env.step(action)
            
            if name == "Contextual Bandit":
                agent.update(reward)
                
            for key in history.keys():
                history[key].append(info[key])
            state = next_state
            
        global_results[name] = calculate_performance_metrics(history)
        
    with open("artifacts/evaluation_summary.json", "w") as f:
        json.dump(global_results, f, indent=4)
        
    generate_evaluation_plots(global_results)
    
    print("\n" + "="*50 + "\nEVALUATION PERFORMANCE SCOREBOARD\n" + "="*50)
    for model, metrics in global_results.items():
        print(f"\nModel Architecture: {model}")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.2f}")
            
    return global_results

if __name__ == "__main__":
    run_evaluation_suite()
