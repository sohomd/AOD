# Attention-on-Demand (AoD)
## Surprise-Driven Selective Computation in Reinforcement Learning

---

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Train AoD on MemoryS7
python train.py --model aod --env MemoryS7 --seed 0

# Train all baselines
python train.py --model gru --env MemoryS7 --seed 0
python train.py --model transformer --env MemoryS7 --seed 0
python train.py --model mlp --env MemoryS7 --seed 0
python train.py --model fixed_mixture --env MemoryS7 --seed 0

# Run all experiments (all models, envs, seeds)
bash run_experiments.sh

# Generate paper figures
python analyze.py
```

## File Structure

```
aod/
├── train.py          # Main training script
├── models.py         # All architectures (MLP, GRU, Transformer, AoD)
├── ppo.py            # PPO trainer with AoD losses
├── env_utils.py      # MiniGrid environments + corruption
├── logger.py         # CSV logging
├── analyze.py        # Paper figure generation
├── run_experiments.sh# Full experiment sweep
└── requirements.txt
```

## Models

| Model | Description |
|-------|-------------|
| `mlp` | Memoryless baseline |
| `gru` | Recurrent baseline |
| `transformer` | Full-attention baseline |
| `fixed_mixture` | Fixed ω=0.5 hybrid |
| `aod` | Attention-on-Demand (ours) |

## Environments

| Environment | Description |
|-------------|-------------|
| `MemoryS7` | Memory task requiring long-range recall |
| `DoorKey` | Key-door task with partial observability |
| `DynamicObstacles` | Navigation with dynamic obstacles |
| `MemoryS7Corrupt` | MemoryS7 with observation dropout |

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lambda_compute` | 0.01 | Attention sparsity penalty |
| `--alpha_pred` | 0.1 | Prediction loss weight |
| `--beta_gate` | 0.01 | Gate entropy regularization |
| `--gate_threshold` | 0.5 | Hard gating threshold at inference |
| `--hidden_dim` | 128 | Hidden dimension (all models) |
| `--seq_len` | 16 | History length for attention |
