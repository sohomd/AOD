"""
Attention-on-Demand (AoD) — Main Training Script

Usage:
    python train.py --model aod --env MemoryS7 --seed 0
    python train.py --model gru --env MemoryS7 --seed 0
    python train.py --model transformer --env MemoryS7 --seed 0
    python train.py --model mlp --env MemoryS7 --seed 0

    # Corruption experiment
    python train.py --model aod --env MemoryS7 --seed 0 --corruption_p 0.3

    # Lambda sweep
    python train.py --model aod --env MemoryS7 --seed 0 --lambda_compute 0.05
"""

import argparse
import os
import random
import time

import numpy as np
import torch

from env_utils import make_env, ENV_MAP
from models import make_model
from ppo import PPOTrainer
from logger import Logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train AoD and baseline policies on MiniGrid environments."
    )

    # Model
    parser.add_argument(
        "--model", type=str, default="aod",
        choices=["aod", "gru", "transformer", "mlp", "fixed_mixture"],
        help="Policy architecture",
    )

    # Environment
    parser.add_argument(
        "--env", type=str, default="MemoryS7",
        choices=sorted(ENV_MAP.keys()),
        help="MiniGrid environment",
    )
    parser.add_argument(
        "--corruption_p", type=float, default=0.0,
        help="Observation dropout probability (0=none)",
    )

    # Training
    parser.add_argument("--total_steps", type=int, default=2_000_000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument(
        "--n_steps", type=int, default=128,
        help="Steps per rollout per env",
    )
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Timestep-equivalent batch size (actual minibatch is "
             "batch_size // n_steps environments × n_steps timesteps)",
    )
    parser.add_argument(
        "--env_batch_size", type=int, default=None,
        help="Override: number of environments per minibatch "
             "(takes precedence over --batch_size)",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--anneal_lr", action="store_true",
        help="Linearly anneal learning rate to 0 over training",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    # AoD-specific
    parser.add_argument(
        "--lambda_compute", type=float, default=0.01,
        help="Compute regularization coefficient",
    )
    parser.add_argument(
        "--alpha_pred", type=float, default=0.1,
        help="Prediction loss coefficient",
    )
    parser.add_argument(
        "--beta_gate", type=float, default=0.01,
        help="Gate entropy coefficient",
    )

    # Architecture
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument(
        "--seq_len", type=int, default=16,
        help="History length for attention",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument(
        "--exp_name", type=str, default="",
        help="Override experiment name (default: auto-generated)",
    )
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--resume", type=str, default="",
        help="Path to checkpoint to resume from",
    )

    return parser.parse_args()


def make_exp_name(args):
    """
    Generate an experiment name that encodes all non-default settings
    relevant to distinguishing runs.

    Naming convention (matches analyze.py expectations):
        {model}_{env}_seed{seed}
        {model}_lambda{lambda_compute}_{env}_seed{seed}
        {model}_{env}_corrupt{corruption_p}_seed{seed}
    """
    parts = [args.model]

    # Include lambda_compute if non-default (for Pareto sweep)
    if args.lambda_compute != 0.01:
        parts.append(f"lambda{args.lambda_compute:g}")

    parts.append(args.env)

    # Include corruption if active
    if args.corruption_p > 0.0:
        parts.append(f"corrupt{args.corruption_p:.1f}")

    parts.append(f"seed{args.seed}")
    return "_".join(parts)


def set_seed(seed, deterministic=True):
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value.
        deterministic: If True, also set cuDNN to deterministic mode.
            This may reduce performance but ensures reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    start_time = time.time()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Seed
    set_seed(args.seed)

    # Experiment name
    if not args.exp_name:
        args.exp_name = make_exp_name(args)
    save_dir = os.path.join(args.results_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # Use context managers for proper resource cleanup
    with Logger(save_dir) as logger:
        logger.log_args(args)

        # Environments
        envs = make_env(
            env_name=args.env,
            n_envs=args.n_envs,
            seed=args.seed,
            corruption_p=args.corruption_p,
        )

        try:
            obs_shape = envs.single_observation_space.shape
            n_actions = envs.single_action_space.n
            print(f"Obs shape: {obs_shape} | Actions: {n_actions}")

            # Model
            model = make_model(
                model_type=args.model,
                obs_shape=obs_shape,
                n_actions=n_actions,
                hidden_dim=args.hidden_dim,
                n_heads=args.n_heads,
                seq_len=args.seq_len,
            ).to(device)

            n_params = sum(p.numel() for p in model.parameters())
            print(f"Model: {args.model} | Params: {n_params:,}")

            # Trainer
            trainer = PPOTrainer(
                model=model,
                envs=envs,
                device=device,
                args=args,
                logger=logger,
                save_dir=save_dir,
            )

            # Resume from checkpoint if specified
            if args.resume:
                trainer.load(args.resume)

            # Train
            print(f"\nStarting training: {args.exp_name}")
            print(f"Total steps: {args.total_steps:,}")
            print(f"Updates: {trainer.total_updates:,} "
                  f"({args.n_steps} steps × {args.n_envs} envs × "
                  f"{args.n_epochs} epochs)")
            print()

            trainer.train()

        finally:
            envs.close()

    # Summary
    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining complete: {args.exp_name}")
    print(f"Duration: {hours}h {minutes}m {seconds}s")
    print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    main()