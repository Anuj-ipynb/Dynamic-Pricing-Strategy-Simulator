# Dynamic Pricing Strategy Simulator Using Reinforcement Learning

A self-contained production framework for optimizing continuous dynamic pricing models using standard deep-policy methods.

## Setup Instructions
1. Install Dependencies: `pip install -r requirements.txt`
2. Run Training Pipeline: `python training/train.py`
3. Evaluate Models & Create Plots: `python training/evaluate.py`
4. Start FastAPI Application: `uvicorn api.app:app --host 0.0.0.0 --port 8000`
5. Launch Streamlit UI Dashboard: `streamlit run frontend/app.py`
