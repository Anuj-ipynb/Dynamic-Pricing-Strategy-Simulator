import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import json

from agents.ppo_agent import PPOAgent
from agents.dqn_agent import DQNAgent
from env.pricing_env import DynamicPricingEnv


# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI(
    title="Dynamic Pricing Strategy Core Service Engine",
    description="RL-powered pricing API",
    version="1.0.0"
)

# ---------------------------
# CORS (REQUIRED FOR FRONTEND)
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Load Models SAFELY
# ---------------------------
ppo_model = PPOAgent()
dqn_model = DQNAgent()

PPO_LOADED = False
DQN_LOADED = False

if os.path.exists("artifacts/models/ppo_model.pt"):
    try:
        ppo_model.load("artifacts/models/ppo_model.pt")
        PPO_LOADED = True
        print("✅ PPO model loaded")
    except Exception as e:
        print(f"❌ PPO load failed: {e}")

if os.path.exists("artifacts/models/dqn_model.pt"):
    try:
        dqn_model.load("artifacts/models/dqn_model.pt")
        DQN_LOADED = True
        print("✅ DQN model loaded")
    except Exception as e:
        print(f"⚠️ DQN load skipped (incompatible): {e}")


# ---------------------------
# Schemas
# ---------------------------
class PriceRequestSchema(BaseModel):
    state: list
    model: str


class SimulationRequestSchema(BaseModel):
    steps: int
    model: str


# ---------------------------
# Price Endpoint
# ---------------------------
@app.post("/price")
def compute_price(payload: PriceRequestSchema):

    if len(payload.state) != 5:
        raise HTTPException(
            status_code=400,
            detail="State must be length 5"
        )

    state = np.array(payload.state, dtype=np.float32)

    model_name = payload.model.lower()

    if model_name == "ppo":
        if not PPO_LOADED:
            raise HTTPException(status_code=500, detail="PPO model not loaded")
        price, _, _ = ppo_model.select_action(state)

    elif model_name == "dqn":
        if not DQN_LOADED:
            raise HTTPException(status_code=500, detail="DQN model not available")
        price, _ = dqn_model.select_action(state, eval_mode=True)

    else:
        raise HTTPException(status_code=400, detail="Invalid model")

    return {"price": float(price)}


# ---------------------------
# Simulation Endpoint
# ---------------------------
@app.post("/simulate")
def run_simulation(payload: SimulationRequestSchema):

    env = DynamicPricingEnv()
    state = env.reset()

    prices = []
    revenues = []

    done = False
    step = 0

    model_name = payload.model.lower()

    while not done and step < payload.steps:

        if model_name == "ppo":
            if not PPO_LOADED:
                raise HTTPException(status_code=500, detail="PPO not loaded")
            action, _, _ = ppo_model.select_action(state)

        elif model_name == "dqn":
            if not DQN_LOADED:
                raise HTTPException(status_code=500, detail="DQN not available")
            action, _ = dqn_model.select_action(state, eval_mode=True)

        else:
            raise HTTPException(status_code=400, detail="Invalid model")

        state, reward, done, info = env.step(action)

        prices.append(info["price"])
        revenues.append(info["revenue"])

        step += 1

    return {
        "prices": prices,
        "total_revenue": float(sum(revenues)),
        "steps_executed": step
    }


# ---------------------------
# Metrics Endpoint
# ---------------------------
@app.get("/metrics")
def fetch_metrics():

    path = "artifacts/evaluation_summary.json"

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Run evaluation first")

    with open(path, "r") as f:
        return json.load(f)


# ---------------------------
# Run (optional)
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)