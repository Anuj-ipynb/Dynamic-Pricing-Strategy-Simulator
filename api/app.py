import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import json
from agents.ppo_agent import PPOAgent
from agents.dqn_agent import DQNAgent
from env.pricing_env import DynamicPricingEnv

app = FastAPI(
    title="Dynamic Pricing Strategy Core Service Engine", 
    description="Production-grade API inference platform deploying Reinforcement Learning execution nodes.",
    version="1.0.0"
)

ppo_model = PPOAgent()
dqn_model = DQNAgent()

if os.path.exists("artifacts/models/ppo_model.pt"):
    ppo_model.load("artifacts/models/ppo_model.pt")
if os.path.exists("artifacts/models/dqn_model.pt"):
    dqn_model.load("artifacts/models/dqn_model.pt")

class PriceRequestSchema(BaseModel):
    state: list
    model: str

class SimulationRequestSchema(BaseModel):
    steps: int
    model: str

@app.post("/price")
def compute_price(payload: PriceRequestSchema):
    if len(payload.state) != 5:
        raise HTTPException(status_code=400, detail="State telemetry payload vector must match shape format [5]")
    
    state_arr = np.array(payload.state, dtype=np.float32)
    
    if payload.model.lower() == "ppo":
        price, _, _ = ppo_model.select_action(state_arr)
    elif payload.model.lower() == "dqn":
        price, _ = dqn_model.select_action(state_arr, eval_mode=True)
    else:
        raise HTTPException(status_code=400, detail="Unsupported model engine target string.")
        
    return {"price": float(price)}

@app.post("/simulate")
def run_simulation(payload: SimulationRequestSchema):
    env = DynamicPricingEnv()
    state = env.reset()
    
    prices, revenues = [], []
    done = False
    current_step = 0
    
    while not done and current_step < payload.steps:
        if payload.model.lower() == "ppo":
            action, _, _ = ppo_model.select_action(state)
        elif payload.model.lower() == "dqn":
            action, _ = dqn_model.select_action(state, eval_mode=True)
        else:
            raise HTTPException(status_code=400, detail="Unknown validation target identifier.")
            
        state, reward, done, info = env.step(action)
        prices.append(info['price'])
        revenues.append(info['revenue'])
        current_step += 1
        
    return {
        "prices": prices,
        "total_revenue": sum(revenues),
        "steps_executed": current_step
    }

@app.get("/metrics")
def fetch_evaluation_metrics():
    summary_path = "artifacts/evaluation_summary.json"
    if not os.path.exists(summary_path):
        raise HTTPException(status_code=404, detail="Inference summary telemetry not compiled yet.")
    with open(summary_path, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
