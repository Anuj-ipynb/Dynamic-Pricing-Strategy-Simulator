import sys
import os
import json
import glob
from datetime import datetime

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
import pandas as pd

from agents.ppo_agent import PPOAgent
from agents.dqn_agent import DQNAgent
from env.pricing_env import DynamicPricingEnv


# =========================================================
# FASTAPI APP
# =========================================================
app = FastAPI(
    title="Dynamic Pricing Strategy Core Service Engine",
    description=(
        "Production-grade RL analytics and pricing API "
        "for dynamic pricing optimization"
    ),
    version="2.0.0"
)

# =========================================================
# CORS
# =========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# LOAD MODELS
# =========================================================
ppo_model = PPOAgent()
dqn_model = DQNAgent()

PPO_LOADED = False
DQN_LOADED = False

PPO_MODEL_PATH = "artifacts/models/ppo_model.pt"
DQN_MODEL_PATH = "artifacts/models/dqn_model.pt"

if os.path.exists(PPO_MODEL_PATH):

    try:

        ppo_model.load(PPO_MODEL_PATH)

        PPO_LOADED = True

        print("✓ PPO model loaded")

    except Exception as e:

        print(f"✗ PPO loading failed: {e}")

if os.path.exists(DQN_MODEL_PATH):

    try:

        dqn_model.load(DQN_MODEL_PATH)

        DQN_LOADED = True

        print("✓ DQN model loaded")

    except Exception as e:

        print(f"✗ DQN loading failed: {e}")


# =========================================================
# REQUEST SCHEMAS
# =========================================================
class PriceRequestSchema(BaseModel):

    state: list
    model: str


class SimulationRequestSchema(BaseModel):

    steps: int
    model: str


# =========================================================
# HELPERS
# =========================================================
def latest_file(directory, pattern):

    files = glob.glob(
        os.path.join(directory, pattern)
    )

    if not files:
        return None

    return max(files, key=os.path.getctime)


# =========================================================
# ROOT
# =========================================================
@app.get("/")
def root():

    return {
        "service": "Dynamic Pricing RL Platform",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


# =========================================================
# HEALTH
# =========================================================
@app.get("/health")
def health_check():

    return {

        "status": "healthy",

        "models": {

            "ppo_loaded": PPO_LOADED,
            "dqn_loaded": DQN_LOADED
        },

        "artifacts": {

            "ppo_exists": os.path.exists(
                PPO_MODEL_PATH
            ),

            "dqn_exists": os.path.exists(
                DQN_MODEL_PATH
            ),

            "evaluation_exists": os.path.exists(
                "artifacts/evaluation_summary.json"
            )
        },

        "timestamp": datetime.now().isoformat()
    }


# =========================================================
# PRICE INFERENCE
# =========================================================
@app.post("/price")
def compute_price(payload: PriceRequestSchema):

    if len(payload.state) != 5:

        raise HTTPException(
            status_code=400,
            detail="State vector must have length 5"
        )

    state = np.array(
        payload.state,
        dtype=np.float32
    )

    model_name = payload.model.lower()

    # -----------------------------------------------------
    # PPO
    # -----------------------------------------------------
    if model_name == "ppo":

        if not PPO_LOADED:

            raise HTTPException(
                status_code=500,
                detail="PPO model not loaded"
            )

        action, _, _ = ppo_model.select_action(
            state
        )

        action = float(
            np.clip(action, 5.0, 100.0)
        )

        return {
            "model": "PPO",
            "price": action
        }

    # -----------------------------------------------------
    # DQN
    # -----------------------------------------------------
    elif model_name == "dqn":

        if not DQN_LOADED:

            raise HTTPException(
                status_code=500,
                detail="DQN model not loaded"
            )

        action, _ = dqn_model.select_action(
            state,
            eval_mode=True
        )

        return {
            "model": "DQN",
            "price": float(action)
        }

    # -----------------------------------------------------
    # INVALID
    # -----------------------------------------------------
    else:

        raise HTTPException(
            status_code=400,
            detail="Invalid model requested"
        )


# =========================================================
# LIVE SIMULATION
# =========================================================
@app.post("/simulate")
def run_simulation(payload: SimulationRequestSchema):

    env = DynamicPricingEnv()

    state = env.reset()

    done = False

    step = 0

    model_name = payload.model.lower()

    # =====================================================
    # TRACES
    # =====================================================
    pricing_trace = []

    revenue_trace = []

    inventory_trace = []

    trust_trace = []

    reward_trace = []

    # =====================================================
    # LOOP
    # =====================================================
    while not done and step < payload.steps:

        # -------------------------------------------------
        # PPO
        # -------------------------------------------------
        if model_name == "ppo":

            if not PPO_LOADED:

                raise HTTPException(
                    status_code=500,
                    detail="PPO not loaded"
                )

            action, _, _ = ppo_model.select_action(
                state
            )

        # -------------------------------------------------
        # DQN
        # -------------------------------------------------
        elif model_name == "dqn":

            if not DQN_LOADED:

                raise HTTPException(
                    status_code=500,
                    detail="DQN not available"
                )

            action, _ = dqn_model.select_action(
                state,
                eval_mode=True
            )

        else:

            raise HTTPException(
                status_code=400,
                detail="Invalid model"
            )

        # -------------------------------------------------
        # ENV STEP
        # -------------------------------------------------
        state, reward, done, info = env.step(action)

        # -------------------------------------------------
        # STORE
        # -------------------------------------------------
        pricing_trace.append(
            float(info["price"])
        )

        revenue_trace.append(
            float(info["revenue"])
        )

        inventory_trace.append(
            float(info["inventory"])
        )

        trust_trace.append(
            float(info["trust"])
        )

        reward_trace.append(
            float(reward)
        )

        step += 1

    # =====================================================
    # METRICS
    # =====================================================
    total_revenue = float(
        np.sum(revenue_trace)
    )

    total_reward = float(
        np.sum(reward_trace)
    )

    avg_price = float(
        np.mean(pricing_trace)
    )

    price_volatility = float(
        np.std(pricing_trace)
    )

    # =====================================================
    # RESPONSE
    # =====================================================
    return {

        "total_revenue": total_revenue,

        "total_reward": total_reward,

        "avg_price": avg_price,

        "price_volatility": price_volatility,

        "pricing_trace": pricing_trace,

        "revenue_trace": revenue_trace,

        "inventory_trace": inventory_trace,

        "trust_trace": trust_trace,

        "steps_executed": step
    }

# =========================================================
# GLOBAL METRICS
# =========================================================
@app.get("/metrics")
def fetch_metrics():

    metrics_path = (
        "artifacts/evaluation_summary.json"
    )

    if not os.path.exists(metrics_path):

        raise HTTPException(
            status_code=404,
            detail="Evaluation results not found"
        )

    with open(metrics_path, "r") as f:

        metrics = json.load(f)

    return metrics


# =========================================================
# TRAINING HISTORY
# =========================================================
@app.get("/training-history")
def fetch_training_history():

    history_file = latest_file(
        "artifacts/history",
        "*_training_history.json"
    )

    if history_file is None:

        raise HTTPException(
            status_code=404,
            detail="Training history not found"
        )

    with open(history_file, "r") as f:

        history = json.load(f)

    return history


# =========================================================
# MODEL METADATA
# =========================================================
@app.get("/model-info")
def fetch_model_info():

    metadata_file = latest_file(
        "artifacts/metadata",
        "*_metadata.json"
    )

    if metadata_file is None:

        raise HTTPException(
            status_code=404,
            detail="Metadata unavailable"
        )

    with open(metadata_file, "r") as f:

        metadata = json.load(f)

    return metadata


# =========================================================
# COMPARISON ANALYTICS
# =========================================================
@app.get("/compare")
def compare_models():

    metrics_path = (
        "artifacts/evaluation_summary.json"
    )

    if not os.path.exists(metrics_path):

        raise HTTPException(
            status_code=404,
            detail="Evaluation results unavailable"
        )

    with open(metrics_path, "r") as f:

        metrics = json.load(f)

    comparison = []

    for model_name, values in metrics.items():

        comparison.append({

            "model": model_name,

            "revenue": values.get(
                "Total Revenue",
                0.0
            ),

            "volatility": values.get(
                "Price Volatility (StdDev)",
                0.0
            ),

            "inventory_utilization": values.get(
                "Inventory Utilization (%)",
                0.0
            ),

            "demand_fulfillment": values.get(
                "Demand Fulfillment Rate (%)",
                0.0
            ),

            "policy_smoothness": values.get(
                "Policy Smoothness",
                0.0
            )
        })

    comparison = sorted(
        comparison,
        key=lambda x: x["revenue"],
        reverse=True
    )

    return {
        "rankings": comparison
    }


# =========================================================
# EXPERIMENT LOGS
# =========================================================
@app.get("/experiment-log")
def fetch_experiment_logs():

    csv_path = "experiments/results.csv"

    if not os.path.exists(csv_path):

        raise HTTPException(
            status_code=404,
            detail="Experiment log unavailable"
        )

    df = pd.read_csv(csv_path)

    latest_rows = (
    df.sort_values("episode")
    .tail(50)
)

# Replace NaN/inf values
    latest_rows = latest_rows.replace(
        [np.nan, np.inf, -np.inf],
    None
    )

    return latest_rows.to_dict(
    orient="records"
    )


# =========================================================
# SYSTEM TELEMETRY
# =========================================================
@app.get("/telemetry")
def telemetry():

    telemetry_payload = {

        "system_status": "operational",

        "models": {

            "ppo": {
                "loaded": PPO_LOADED
            },

            "dqn": {
                "loaded": DQN_LOADED
            }
        },

        "artifacts": {

            "models_directory": os.path.exists(
                "artifacts/models"
            ),

            "plots_directory": os.path.exists(
                "artifacts/plots"
            ),

            "history_directory": os.path.exists(
                "artifacts/history"
            )
        },

        "timestamp": datetime.now().isoformat()
    }

    return telemetry_payload


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )