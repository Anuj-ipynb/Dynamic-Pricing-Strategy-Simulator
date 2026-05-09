<div align="center">

```
██████╗ ██╗   ██╗███╗   ██╗ █████╗ ███╗   ███╗██╗ ██████╗
██╔══██╗╚██╗ ██╔╝████╗  ██║██╔══██╗████╗ ████║██║██╔════╝
██║  ██║ ╚████╔╝ ██╔██╗ ██║███████║██╔████╔██║██║██║     
██║  ██║  ╚██╔╝  ██║╚██╗██║██╔══██║██║╚██╔╝██║██║██║     
██████╔╝   ██║   ██║ ╚████║██║  ██║██║ ╚═╝ ██║██║╚██████╗
╚═════╝    ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝ ╚═════╝

██████╗ ██████╗ ██╗ ██████╗██╗███╗   ██╗ ██████╗
██╔══██╗██╔══██╗██║██╔════╝██║████╗  ██║██╔════╝
██████╔╝██████╔╝██║██║     ██║██╔██╗ ██║██║  ███╗
██╔═══╝ ██╔══██╗██║██║     ██║██║╚██╗██║██║   ██║
██║     ██║  ██║██║╚██████╗██║██║ ╚████║╚██████╔╝
╚═╝     ╚═╝  ╚═╝╚═╝ ╚═════╝╚═╝╚═╝  ╚═══╝ ╚═════╝
```

### Deep Reinforcement Learning · Non-Stationary Market Optimization · Production MLOps

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![SDG 9](https://img.shields.io/badge/SDG-9%20Innovation-F36D25?style=flat-square)](https://sdgs.un.org/goals/goal9)
[![SDG 12](https://img.shields.io/badge/SDG-12%20Responsible%20Consumption-CF8D2A?style=flat-square)](https://sdgs.un.org/goals/goal12)

</div>

---

## What Is This?

**Dynamic Pricing RL** is a production-oriented reinforcement learning framework for optimizing pricing decisions in non-stationary marketplace environments. Static pricing rules fail when demand shifts, customer trust erodes, and inventory depletes — this platform uses deep RL to adapt in real time.

The system trains **PPO** (continuous control) and **DQN** (discrete action) agents against a richly simulated market, benchmarks them against five analytical baselines, and exposes full telemetry, experiment tracking, monitoring APIs, and a live React analytics dashboard.

> Built as an operational RL analytics system — not an academic notebook.  
> Training, inference, monitoring, telemetry, evaluation, and visualization are fully separated.

---

## Why Reinforcement Learning?

Classical pricing strategies fail under:

| Challenge | Classical Approach | RL Approach |
|---|---|---|
| Cyclical demand shifts | Recalibrate manually | Adapts online |
| Customer trust degradation | Ignored | Modeled in reward |
| Inventory depletion | Threshold rules | Inventory-aware reward shaping |
| Stochastic demand noise | Averaged out | Learned via exploration |
| Long-horizon tradeoffs | Myopic optimization | Discounted return maximization |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Dynamic Pricing RL Platform                        │
├──────────────┬─────────────────┬──────────────────┬────────────────────┤
│  ENVIRONMENT │     AGENTS      │    BASELINES     │    OPERATIONS      │
│              │                 │                  │                    │
│ pricing_env  │  PPO Agent      │  Fixed Pricing   │  FastAPI Backend   │
│              │  (continuous)   │  Rule-Based      │  9 REST endpoints  │
│ Non-stationary│               │  Demand Model    │                    │
│ market sim   │  DQN Agent      │  Moving Average  │  React Dashboard   │
│              │  (discrete)     │  Bandit          │  Live analytics    │
│ State: 5-dim │                 │                  │                    │
│ Reward: R_t  │  Shared config  │  Benchmarking    │  Experiment Logs   │
│ with 4 terms │  via YAML       │  suite           │  Artifact store    │
└──────────────┴─────────────────┴──────────────────┴────────────────────┘
         │               │                │                  │
         └───────────────┴────────────────┴──────────────────┘
                              training/train.py
                              training/evaluate.py
```

---

## Environment Design

The marketplace environment is a **partially non-stationary sequential decision system** — intentionally designed to defeat static optimization.

### State Vector (5-dimensional)

```
s_t = [
    normalized_price,       ← current price relative to bounds
    previous_demand,        ← demand at t-1 (momentum signal)
    older_demand,           ← demand at t-2 (trend signal)
    normalized_inventory,   ← fraction of stock remaining
    normalized_timestep     ← temporal market regime signal
]
```

This lets agents estimate demand momentum, reason about depletion risk, detect temporal market regimes, and learn long-horizon tradeoffs.

### Market Dynamics Simulated

- Cyclical demand shifts (seasonal/periodic patterns)
- Customer trust degradation from aggressive pricing
- Inventory depletion with stockout risk
- Stochastic demand noise (ε ~ N(0, σ))
- Volatility penalties for erratic pricing

---

## Reward Function

$$R_t = \text{Revenue}_t - \lambda_1 \cdot \text{Stockout}_t - \lambda_2 \cdot \text{Overpricing}_t - \lambda_3 \cdot \text{Volatility}_t + \lambda_4 \cdot \text{Retention}_t$$

| Term | Role |
|---|---|
| Revenue | Reward profitable transactions |
| Stockout penalty (λ₁) | Discourage inventory exhaustion |
| Overpricing penalty (λ₂) | Prevent trust destruction |
| Volatility penalty (λ₃) | Stabilize pricing trajectories |
| Retention incentive (λ₄) | Preserve long-run customer base |

---

## Agents

### PPO — Proximal Policy Optimization
- Continuous action space (price as a real-valued output)
- Actor-Critic architecture with clipped surrogate objective
- Best suited for smooth, high-dimensional pricing manifolds
- Gradient clipping + entropy regularization for stability

### DQN — Deep Q-Network
- Discrete action space (bucketed price levels)
- Experience replay + target network stabilization
- ε-greedy exploration with decay schedule
- Baseline comparator against PPO continuous control

---

## Baselines

The RL agents are benchmarked against five analytical strategies:

| Baseline | Strategy |
|---|---|
| **Fixed Pricing** | Constant price throughout horizon |
| **Rule-Based** | Heuristic price rules (demand thresholds) |
| **Demand Optimization** | Model-based price from demand curve |
| **Moving Average** | Price tracks rolling demand average |
| **Contextual Bandit** | One-step exploration, no memory |

This demonstrates RL's advantage under non-stationary demand, long-term inventory reasoning, and adaptive customer behavior — where reactive and model-free baselines degrade.

---

## Project Structure

```
dynamic-pricing-rl/
│
├── 🤖 agents/
│   ├── ppo_agent.py              ← PPO continuous-control agent
│   └── dqn_agent.py              ← DQN discrete-action agent
│
├── 🌐 api/
│   └── app.py                    ← FastAPI backend (9 endpoints)
│
├── 📦 artifacts/
│   ├── history/                  ← Training telemetry logs
│   ├── metadata/                 ← Hyperparameters + run info
│   ├── models/                   ← Saved agent checkpoints
│   └── plots/                    ← Generated evaluation figures
│
├── 📊 baselines/
│   ├── bandit.py                 ← Contextual bandit strategy
│   ├── demand_model.py           ← Demand-curve optimizer
│   ├── fixed.py                  ← Fixed price baseline
│   ├── moving_avg.py             ← Moving average baseline
│   └── rule_based.py             ← Heuristic rule baseline
│
├── ⚙️ configs/
│   └── config.yaml               ← Centralized experiment config
│
├── 🏪 env/
│   └── pricing_env.py            ← Non-stationary market simulator
│
├── 🧪 experiments/
│   └── results.csv               ← Tracked experiment outcomes
│
├── 🖥️ frontend/                  ← React analytics dashboard
│
├── 🏋️ training/
│   ├── train.py                  ← PPO + DQN training loop
│   └── evaluate.py               ← Full evaluation + benchmarking
│
├── 🔧 utils/
│   ├── metrics.py                ← Evaluation metric utilities
│   └── plots.py                  ← Visualization helpers
│
└── 📋 requirements.txt
```

---

## Quickstart

### 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2 — Configure the Experiment

Edit `configs/config.yaml` to set hyperparameters, seeds, episode length, reward weights (λ₁–λ₄), and agent-specific settings. All training and evaluation scripts read from this single file.

---

### 3 — Train Agents

```bash
python training/train.py
```

This trains both PPO and DQN agents. Artifacts generated:

- Trained model checkpoints → `artifacts/models/`
- Reward convergence telemetry → `artifacts/history/`
- Hyperparameter metadata → `artifacts/metadata/`
- Experiment log entry → `experiments/results.csv`

---

### 4 — Evaluate & Benchmark

```bash
python training/evaluate.py
```

Outputs:

- PPO vs DQN comparison
- All 5 baseline benchmarks
- Inventory trajectory analytics
- Customer trust monitoring
- Convergence plots → `artifacts/plots/`
- Operational telemetry summary

---

### 5 — Launch FastAPI Backend

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

API available at **http://localhost:8000** · Swagger docs at **http://localhost:8000/docs**

---

### 6 — Launch React Dashboard

```bash
cd frontend
npm install
npm run dev
```

Dashboard available at **http://localhost:5173**

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health monitoring |
| `POST` | `/price` | Real-time RL pricing inference |
| `POST` | `/simulate` | Multi-step environment simulation |
| `GET` | `/metrics` | Evaluation metrics snapshot |
| `GET` | `/compare` | PPO vs baseline analytics |
| `GET` | `/training-history` | Reward convergence telemetry |
| `GET` | `/model-info` | Metadata and configuration |
| `GET` | `/telemetry` | Runtime operational telemetry |
| `GET` | `/experiment-log` | Historical experiment logs |

---

## Dashboard Features

The React frontend provides live analytics across six panels:

```
┌─────────────────────┬─────────────────────┐
│  PPO vs DQN         │  Inventory           │
│  Convergence        │  Trajectory          │
│  (reward curves)    │  (depletion risk)    │
├─────────────────────┼─────────────────────┤
│  Customer Trust     │  Pricing             │
│  Monitoring         │  Trajectory          │
│  (retention score)  │  (stability view)    │
├─────────────────────┼─────────────────────┤
│  RL Telemetry       │  Live Simulation     │
│  (KPI panels)       │  (run on demand)     │
└─────────────────────┴─────────────────────┘
```

---

## Experiment Tracking

Every training run automatically records:

| Category | Fields Tracked |
|---|---|
| Performance | Reward, revenue, volatility score |
| Inventory | Utilization rate, stockout events |
| Agent | Hyperparameters, architecture config |
| Convergence | Reward curve, episode-level telemetry |
| Run metadata | Timestamp, experiment ID, seed |

All telemetry is stored locally in `artifacts/` and `experiments/results.csv` for full reproducibility — no external tracking server required.

---

## Reproducibility

The system enforces deterministic execution end-to-end:

```python
np.random.seed(config.seed)           # NumPy global seed
torch.manual_seed(config.seed)        # PyTorch CPU seed
torch.cuda.manual_seed(config.seed)   # PyTorch CUDA seed
torch.backends.cudnn.deterministic = True  # CUDA determinism
```

Every hyperparameter, reward weight, and environment parameter lives in `configs/config.yaml` — one file controls the entire experiment.

---

## SDG Alignment

| Goal | Connection |
|---|---|
| **SDG 9** — Industry, Innovation & Infrastructure | Production RL deployment architecture for smart industrial pricing |
| **SDG 12** — Responsible Consumption & Production | Reduces wasteful stockouts, curbs overpricing, stabilizes supply chains |

---

## Deployment Philosophy

This platform deliberately separates concerns across independently deployable layers:

```
Training  ──►  Artifact Store  ──►  FastAPI Inference
                    │
                    ▼
             Telemetry Store  ──►  React Dashboard
                    │
                    ▼
             Experiment Log   ──►  Reproducibility Audit
```

This structure mirrors production RL deployment workflows used in real AI systems — not a monolithic notebook, but a composable operational stack.

---

<div align="center">

**Dynamic Pricing Strategy Simulator**  
Deep Reinforcement Learning · PPO & DQN · Non-Stationary Markets

*Adapt. Optimize. Deploy.*

</div>