# Dynamic Pricing Strategy Simulator Using Reinforcement Learning

A production-oriented reinforcement learning framework for optimizing dynamic pricing decisions in non-stationary marketplace environments. The platform compares deep RL approaches against multiple analytical and rule-based baselines while exposing full telemetry, experiment tracking, monitoring APIs, and operational analytics dashboards.

---

# Core Features

- PPO continuous-control pricing optimization
- DQN discrete pricing baseline
- Non-stationary market simulation
- Customer trust and memory dynamics
- Inventory-aware reward shaping
- Experiment tracking and telemetry
- FastAPI monitoring and inference APIs
- React analytics dashboard
- PPO vs DQN convergence visualization
- Multi-baseline evaluation framework
- Reproducible training configuration
- Artifact and metadata versioning

---

# Environment Design

The marketplace environment is intentionally designed to challenge static optimization techniques.

The simulator models:

- cyclical demand shifts,
- customer trust degradation,
- inventory depletion,
- stochastic demand noise,
- volatility penalties,
- and long-horizon pricing tradeoffs.

The environment behaves as a partially non-stationary sequential decision system.

---

# State Representation

The environment state vector is:

```text
[
    normalized_price,
    previous_demand,
    older_demand,
    normalized_inventory,
    normalized_timestep
]
````

This formulation allows agents to:

* estimate short-term demand momentum,
* reason about inventory depletion,
* adapt to temporal market regimes,
* and learn long-horizon pricing strategies.

---

# Reward Function

The reward function balances:

* revenue maximization,
* customer retention,
* inventory sustainability,
* and pricing stability.

R_t = Revenue_t - \lambda_1 Stockout_t - \lambda_2 Overpricing_t - \lambda_3 Volatility_t + \lambda_4 Retention_t

Where:

* revenue rewards profitable transactions,
* stockout penalties discourage inventory exhaustion,
* volatility penalties stabilize pricing,
* retention incentives preserve customer trust.

---

# Project Structure

```text
dynamic-pricing-rl/
│
├── agents/
│   ├── ppo_agent.py
│   └── dqn_agent.py
│
├── api/
│   └── app.py
│
├── artifacts/
│   ├── history/
│   ├── metadata/
│   ├── models/
│   └── plots/
│
├── baselines/
│   ├── bandit.py
│   ├── demand_model.py
│   ├── fixed.py
│   ├── moving_avg.py
│   └── rule_based.py
│
├── configs/
│   └── config.yaml
│
├── env/
│   └── pricing_env.py
│
├── experiments/
│   └── results.csv
│
├── frontend/
│
├── training/
│   ├── evaluate.py
│   └── train.py
│
├── utils/
│   ├── metrics.py
│   └── plots.py
│
└── requirements.txt
```

---

# Installation

## Backend Setup

```bash
pip install -r requirements.txt
```

---

# Training

Train PPO and DQN agents:

```bash
python training/train.py
```

Artifacts generated:

* trained checkpoints,
* experiment logs,
* telemetry history,
* metadata snapshots.

---

# Evaluation

Run the evaluation suite:

```bash
python training/evaluate.py
```

Outputs:

* PPO vs DQN comparison,
* baseline benchmarking,
* inventory analytics,
* convergence plots,
* operational telemetry.

---

# FastAPI Backend

Launch the backend service:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

---

# Frontend Dashboard

```bash
cd frontend
npm install
npm run dev
```

---

# Monitoring & Telemetry APIs

| Endpoint            | Description                       |
| ------------------- | --------------------------------- |
| `/health`           | System health monitoring          |
| `/price`            | Real-time RL pricing inference    |
| `/simulate`         | Multi-step environment simulation |
| `/metrics`          | Evaluation metrics                |
| `/compare`          | PPO vs baseline analytics         |
| `/training-history` | Reward convergence telemetry      |
| `/model-info`       | Metadata and configuration        |
| `/telemetry`        | Runtime operational telemetry     |
| `/experiment-log`   | Historical experiment logs        |

---

# Experiment Tracking

The platform automatically tracks:

* rewards,
* revenues,
* volatility,
* inventory utilization,
* model metadata,
* hyperparameters,
* convergence telemetry,
* timestamps,
* experiment identifiers.

All telemetry is stored locally for reproducibility and analytics.

---

# Reproducibility

The system enforces deterministic execution through:

* NumPy seeding,
* PyTorch seeding,
* CUDA deterministic settings,
* centralized YAML configuration.

---

# RL Baselines

The project evaluates RL against:

* Fixed Pricing
* Rule-Based Pricing
* Demand Optimization
* Moving Average Pricing
* Contextual Bandit Pricing

This demonstrates the advantage of RL under:

* non-stationary demand,
* long-term inventory reasoning,
* and adaptive customer behavior.

---

# Dashboard Features

The frontend dashboard provides:

* PPO vs DQN convergence analytics,
* inventory trajectory visualization,
* customer trust monitoring,
* pricing trajectory analysis,
* RL telemetry panels,
* operational KPI tracking,
* live simulation execution.

---

# Deployment Philosophy

The platform is designed as an operational RL analytics system rather than a standalone academic notebook.

The architecture separates:

* training,
* inference,
* monitoring,
* telemetry,
* evaluation,
* and visualization.

This structure aligns with modern RL deployment workflows used in production AI systems.

---

# Sustainability & SDG Alignment

This project aligns with:

* SDG 9 (Industry, Innovation and Infrastructure)
* SDG 12 (Responsible Consumption and Production)

by:

* improving inventory efficiency,
* reducing wasteful stockouts,
* stabilizing pricing volatility,
* and enabling data-driven operational optimization.

```
```
