"""
save_trajectory.py — Record trajectories from a trained AoD agent.
Saves timestep, surprise, and omega arrays for Figure 5.

Usage:
    python save_trajectory.py \
        --checkpoint /home/san/AOD/results/aod_MemoryS7_seed0/model_final.pt
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
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument("--env", type=str, default="MemoryS7")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=10,
        help="Number of episodes to record",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint does not exist:\n{args.checkpoint}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # --------------------------------------------------
    # Environment
    # --------------------------------------------------
    # make_env returns a vector environment even when n_envs=1.
    envs = make_env(
        args.env,
        n_envs=1,
        seed=args.seed,
    )

    # IMPORTANT: use the spaces of ONE environment.
    obs_shape = envs.single_observation_space.shape
    n_actions = envs.single_action_space.n

    print(f"Device: {device}")
    print(f"Environment: {args.env}")
    print(f"Observation shape: {obs_shape}")
    print(f"Actions: {n_actions}")

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = make_model(
        "aod",
        obs_shape,
        n_actions,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        seq_len=args.seq_len,
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # --------------------------------------------------
    # Storage
    # --------------------------------------------------
    all_timesteps = []
    all_surprises = []
    all_omegas = []
    all_episode_ids = []

    global_t = 0
    episodes_done = 0

    # --------------------------------------------------
    # Initial state
    # --------------------------------------------------
    obs, _ = envs.reset(seed=args.seed)

    obs = torch.as_tensor(
        obs,
        dtype=torch.float32,
        device=device,
    )

    hidden = model.init_hidden(1, device)

    print(
        f"\nRecording {args.n_episodes} episodes "
        f"from {args.checkpoint}..."
    )

    # --------------------------------------------------
    # Rollout
    # --------------------------------------------------
    while episodes_done < args.n_episodes:

        with torch.no_grad():
            logits, value, hidden, info = model(
                obs,
                hidden,
            )

            # Deterministic evaluation trajectory.
            action = torch.argmax(
                logits,
                dim=-1,
            )

        # Save AoD quantities for this decision step.
        all_timesteps.append(global_t)
        all_surprises.append(
            float(info["surprise"].item())
        )
        all_omegas.append(
            float(info["omega"].item())
        )
        all_episode_ids.append(episodes_done)

        global_t += 1

        # --------------------------------------------------
        # Environment step
        # --------------------------------------------------
        next_obs, reward, terminated, truncated, infos = envs.step(
            action.cpu().numpy()
        )

        done = bool(
            np.asarray(terminated | truncated).reshape(-1)[0]
        )

        # SAME_STEP vector autoreset returns the next episode's
        # initial observation when the current episode ends.
        if done:
            episodes_done += 1

            print(
                f"  Episode {episodes_done}/"
                f"{args.n_episodes} complete "
                f"at timestep {global_t}"
            )

            # Prevent recurrent/attention history from leaking
            # across episode boundaries.
            hidden = model.init_hidden(
                1,
                device,
            )

        obs = torch.as_tensor(
            next_obs,
            dtype=torch.float32,
            device=device,
        )

    envs.close()

    # --------------------------------------------------
    # Save trajectory
    # --------------------------------------------------
    save_dir = os.path.dirname(
        os.path.abspath(args.checkpoint)
    )

    save_path = os.path.join(
        save_dir,
        "trajectory.npz",
    )

    np.savez(
        save_path,
        timesteps=np.asarray(
            all_timesteps,
            dtype=np.int64,
        ),
        episode=np.asarray(
            all_episode_ids,
            dtype=np.int64,
        ),
        surprise=np.asarray(
            all_surprises,
            dtype=np.float32,
        ),
        omega=np.asarray(
            all_omegas,
            dtype=np.float32,
        ),
    )

    omegas = np.asarray(all_omegas)
    surprises = np.asarray(all_surprises)

    print("\nTrajectory recording complete.")
    print(
        f"Saved {len(all_timesteps)} timesteps to:"
    )
    print(save_path)

    print(f"\nEpisodes: {episodes_done}")
    print(
        f"Mean omega: {np.mean(omegas):.3f}"
    )
    print(
        f"Attention-reliance reduction: "
        f"{100.0 * (1.0 - np.mean(omegas)):.1f}%"
    )
    print(
        f"Mean surprise: {np.mean(surprises):.4f}"
    )


if __name__ == "__main__":
    main()