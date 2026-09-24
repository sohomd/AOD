# Attention-on-Demand (AoD)
## Surprise-Driven Selective Computation in Reinforcement Learning
![Architecture Overview](https://raw.githubusercontent.com/sohomd/AOD/asset/Arch.png)
---

## Setup

```bash
pip install -r requirements.txt
```

> **Required:** `gymnasium == 1.3.0`. The trainer relies on the gymnasium
> **≥ 1.0 vector-env autoreset API** (`SAME_STEP` autoreset and the
> `infos["final_obs"]` / `final_observation` key) to bootstrap value targets on
> time-limit truncations. On older gymnasium (< 1.0) the truncation handling in
> `ppo.py` will break. Pin the version if unsure:
> ```bash
> pip install "gymnasium==1.3.0"
> ```

Optional environment families:

```bash
pip install minigrid     # MiniGrid memory / navigation tasks
pip install popgym       # POPGym partially observable benchmarks
# T-Maze is bundled locally (tmaze_env.py) — no extra install needed
```

> **POPGym action spaces — important.** The policy uses a **Categorical
> (discrete) action head**, so only POPGym environments with a `Discrete`
> action space are supported. Environments with `Box`/continuous action spaces
> will crash. Verify before launching a long run:
> ```bash
> python -c "import popgym, gymnasium as gym; e=gym.make('popgym-RepeatPreviousEasy-v0'); print('obs:', e.observation_space, '| act:', e.action_space)"
> ```
> If `act` prints `Discrete(...)` you are good; if it prints `Box(...)`, skip
> that env. Known-compatible discrete POPGym tasks:
> `popgym-RepeatPreviousEasy-v0`, `popgym-RepeatPreviousMedium-v0`,
> `popgym-CountRecallEasy-v0`, `popgym-AutoencodeEasy-v0`.

## Quick Start

```bash
# Train AoD on a memory task (use a PARTIALLY observable size: S9/S11/S13)
python train.py --model aod --env MemoryS9 --seed 0 \
    --total_steps 2000000 --n_envs 64 --n_steps 512 --env_batch_size 16 \
    --lr 7e-4 --anneal_lr --ent_coef 0.05

# Train all baselines (same settings)
python train.py --model gru           --env MemoryS9 --seed 0 --total_steps 2000000 --n_envs 64 --n_steps 512 --env_batch_size 16 --lr 7e-4 --anneal_lr --ent_coef 0.05
python train.py --model transformer   --env MemoryS9 --seed 0 --total_steps 2000000 --n_envs 64 --n_steps 512 --env_batch_size 16 --lr 7e-4 --anneal_lr --ent_coef 0.05
python train.py --model mlp           --env MemoryS9 --seed 0 --total_steps 2000000 --n_envs 64 --n_steps 512 --env_batch_size 16 --lr 7e-4 --anneal_lr --ent_coef 0.05
python train.py --model fixed_mixture --env MemoryS9 --seed 0 --total_steps 2000000 --n_envs 64 --n_steps 512 --env_batch_size 16 --lr 7e-4 --anneal_lr --ent_coef 0.05

# T-Maze (cleanest "attention only at the junction" testbed)
python train.py --model aod --env TMaze10 --seed 0 \
    --total_steps 1000000 --n_envs 64 --n_steps 128 --env_batch_size 16 \
    --lr 7e-4 --anneal_lr --ent_coef 0.05

# POPGym (flat-vector partially observable benchmark; discrete-action envs only)
python train.py --model aod --env popgym-RepeatPreviousEasy-v0 --seed 0 \
    --total_steps 2000000 --n_envs 64 --n_steps 128 --env_batch_size 16 \
    --lr 7e-4 --anneal_lr --ent_coef 0.05

# Run all experiments (all models, envs, seeds)
bash run_experiments.sh

# Generate paper figures
python analyze.py
```

> **Note on `MemoryS7`:** MemoryS7 is *fully observable* (7×7 grid == 7×7 agent view),
> so it does **not** require memory and cannot distinguish recurrent/attention models
> from a memoryless MLP. Use **MemoryS9 / S11 / S13** (grid larger than the view) for
> any memory claim.

## File Structure

```
aod/
├── train.py          # Main training script
├── models.py         # All architectures (MLP, GRU, Transformer, AoD)
├── ppo.py            # PPO trainer with AoD losses (BPTT + truncation bootstrap)
├── env_utils.py      # MiniGrid + POPGym + T-Maze environments + corruption
├── tmaze_env.py      # Local T-Maze implementation (flat vector obs)
├── logger.py         # CSV logging
├── analyze.py        # Paper figure generation
├── run_experiments.sh# Full experiment sweep
└── requirements.txt
```
<p align="center">
  <img src="https://raw.githubusercontent.com/sohomd/AOD/asset/training1.png" width="32%" alt="MemoryS7 Training" />
  <img src="https://raw.githubusercontent.com/sohomd/AOD/asset/training2.png" width="32%" alt="DoorKey Training" />
  <img src="https://raw.githubusercontent.com/sohomd/AOD/asset/training3.png" width="32%" alt="DynamicObstacles Training" />
</p>

## Models

| Model | Description |
|-------|-------------|
| `mlp` | Memoryless baseline |
| `gru` | Recurrent baseline |
| `transformer` | Full-attention baseline |
| `fixed_mixture` | Fixed ω=0.5 hybrid |
| `aod` | Attention-on-Demand (ours) |

## Environments

The encoder is auto-selected by observation rank: 3D image obs → CNN/one-hot
encoder (MiniGrid); 1D vector obs → MLP encoder (POPGym, T-Maze).

| Environment | Type | Obs | Memory required? |
|-------------|------|-----|------------------|
| `MemoryS7` | MiniGrid | image | ❌ (fully observable) |
| `MemoryS9` | MiniGrid | image | ✅ |
| `MemoryS11` | MiniGrid | image | ✅ |
| `MemoryS13` | MiniGrid | image | ✅ (hardest) |
| `DoorKey` | MiniGrid | image | partial |
| `DynamicObstacles` | MiniGrid | image | ❌ (reactive) |
| `TMaze10` / `TMaze20` / `TMaze50` | T-Maze (local) | vector | ✅ (cue at t=0, recall at junction) |
| `popgym-*` | POPGym | vector | ✅ (varies by task; discrete-action only) |

Observation corruption (stochastic zeroing) is available via `--corruption_p`.

## Key Hyperparameters

| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `--total_steps` | 2000000 | ≥2M (memory tasks) | Total environment steps |
| `--n_envs` | 8 | 64 | Parallel environments |
| `--n_steps` | 128 | 512 (long episodes) | Rollout length per env |
| `--env_batch_size` | None | 16 | Envs per PPO minibatch (preserves sequences) |
| `--lr` | 3e-4 | 7e-4 | Learning rate |
| `--anneal_lr` | off | on | Linear LR annealing |
| `--ent_coef` | 0.01 | 0.05 | Entropy bonus (raise for exploration) |
| `--corruption_p` | 0.0 | — | Per-step observation dropout probability (stochastic zeroing) |
| `--lambda_compute` | 0.01 | — | Attention sparsity penalty (AoD) |
| `--alpha_pred` | 0.1 | — | Prediction loss weight (AoD) |
| `--beta_gate` | 0.01 | — | Gate entropy regularization (AoD) |
| `--hidden_dim` | 128 | — | Hidden dimension (all models) |
| `--seq_len` | 16 | — | History length for attention |

## Implementation Notes

- **Observation encoding.** MiniGrid observations are categorical triples
  `[object, color, state]` per cell and are **one-hot encoded** (not fed as raw
  integers) before the MLP encoder. POPGym / T-Maze use a flat-vector encoder.
- **Recurrent PPO.** Hidden state is carried through each rollout sequence with
  backpropagation-through-time (BPTT); the exact rollout-start hidden state is
  stored and reused during optimization so PPO ratios are consistent.
- **Truncation handling.** Time-limit truncations are bootstrapped with
  `γ·V(final_obs)` rather than treated as true terminals; only genuine
  terminations zero the value target in GAE. This requires the gymnasium
  ≥ 1.0 vector-env autoreset API (see Setup).
- **Efficiency metric.** AoD reports average gate activation `E[ω_t]` as a proxy
  for attention reliance. Both branches are evaluated every step (soft gating);
  the gate controls representational weighting, not executed computation.
