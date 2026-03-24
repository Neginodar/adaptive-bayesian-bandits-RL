# Bayesian Reinforcement Learning — Uncertainty-Driven Exploration

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
python experiments/compare.py
```

## Structure
```
bayesian_rl/
├── agents/
│   ├── bayesian_agent.py   # Thompson Sampling + PyMC MCMC
│   └── dqn_baseline.py     # Vanilla DQN (PyTorch, CPU)
├── models/
│   └── reward_pgm.py       # Hierarchical Bayesian PGM (PyMC)
├── experiments/
│   └── compare.py          # Main runner + plots + CSV export
├── results/                # Auto-created: comparison.png, metrics.csv
└── requirements.txt
```

## No GPU Required
All computation runs on CPU. PyMC uses NUTS (CPU-optimised).
DQN uses a small 2-layer MLP — trains in seconds on CPU.

## Output
- `results/comparison.png` — reward curves + posterior entropy decay
- `results/metrics.csv`    — per-episode metrics for all agents
