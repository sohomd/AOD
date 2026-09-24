"""
save_trajectory.py — Record a trajectory from a trained AoD agent.
Saves timesteps, surprise, and omega arrays for Figure 5.

Usage:
    python save_trajectory.py \
        --checkpoint results/aod_MemoryS7_seed0/model_final.pt
"""

import argparse
import os

import numpy as np
import torch

from env_utils import make_env
from models import make_model


def main():
    parser = argparse.ArgumentParser(
        description="Record trajectory data from a trained AoD agent."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--env", type=str, default="MemoryS7")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--n_episodes", type=int, default=10,
                        help="Number of episodes to record")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = make_env(args.env, n_envs=1, seed=args.seed)
    obs_shape = envs.observation_space.shape
    n_actions = envs.action_space.n

    model = make_model(
        "aod", obs_shape, n_actions,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        seq_len=args.seq_len,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_timesteps = []
    all_surprises = []
    all_omegas = []
    global_t = 0

    obs, _ = envs.reset()
    obs = torch.as_tensor(obs, dtype=torch.float32)
    hidden = model.init_hidden(1, device)
    episodes_done = 0

    print(f"Recording {args.n_episodes} episodes from {args.checkpoint}...")

    while episodes_done < args.n_episodes:
        with torch.no_grad():
            logits, value, hidden, info = model(obs.to(device), hidden)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

        all_timesteps.append(global_t)
        all_surprises.append(info["surprise"].item())
        all_omegas.append(info["omega"].item())
        global_t += 1

        next_obs, reward, terminated, truncated, _ = envs.step(
            action.cpu().numpy()
        )
        done = (terminated | truncated).any()

        if done:
            episodes_done += 1
            hidden = model.init_hidden(1, device)

        obs = torch.as_tensor(next_obs, dtype=torch.float32)

    envs.close()

    save_dir = os.path.dirname(args.checkpoint)
    save_path = os.path.join(save_dir, "trajectory.npz")
    np.savez(
        save_path,
        timesteps=np.array(all_timesteps),
        surprise=np.array(all_surprises),
        omega=np.array(all_omegas),
    )
    print(f"Saved {len(all_timesteps)} timesteps to {save_path}")
    print(f"  Episodes: {episodes_done}")
    print(f"  Mean omega: {np.mean(all_omegas):.3f}")
    print(f"  Mean surprise: {np.mean(all_surprises):.4f}")


if __name__ == "__main__":
    main()