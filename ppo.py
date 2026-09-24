"""
ppo.py — PPO Trainer with AoD-specific loss terms

Optimized objective:
    L = L_PPO + alpha * L_pred + lambda * L_compute - beta * H_gate

Notes:
- We preserve temporal structure during PPO updates by batching full rollout
  sequences across subsets of environments.
- This is important for GRU / Transformer / AoD models, which depend on hidden
  state and should not be trained on randomly shuffled single timesteps.
- Hidden states are detached between timesteps during PPO updates (no TBPTT).
  This is standard for PPO with recurrent policies.
- Value loss uses unclipped MSE (not clipped value loss). This is a deliberate
  choice: clipped value loss can cause underestimation in some settings, and
  unclipped MSE is used in several strong PPO baselines (e.g., CleanRL).
- args.batch_size is interpreted in timestep-equivalent units. The actual
  optimization minibatch is a contiguous sequence of n_steps timesteps for
  (batch_size // n_steps) environments. Use args.env_batch_size to override
  the number of environments per minibatch directly.
"""

import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class RolloutBuffer:
    """Stores transitions for one PPO update."""

    def __init__(self, n_steps, n_envs, obs_shape, hidden_dim, device,
                 store_aod_fields=True):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_shape = obs_shape
        self.hidden_dim = hidden_dim
        self.device = device
        self.store_aod_fields = store_aod_fields

        self.obs = torch.zeros(n_steps, n_envs, *obs_shape, dtype=torch.float32)
        self.actions = torch.zeros(n_steps, n_envs, dtype=torch.long)
        self.rewards = torch.zeros(n_steps, n_envs, dtype=torch.float32)
        self.dones = torch.zeros(n_steps, n_envs, dtype=torch.float32)
        self.values = torch.zeros(n_steps, n_envs, dtype=torch.float32)
        self.log_probs = torch.zeros(n_steps, n_envs, dtype=torch.float32)
        self.advantages = torch.zeros(n_steps, n_envs, dtype=torch.float32)
        self.returns = torch.zeros(n_steps, n_envs, dtype=torch.float32)

        # AoD-specific fields (only allocated when needed)
        if store_aod_fields:
            self.omegas = torch.zeros(n_steps, n_envs, dtype=torch.float32)
            self.surprises = torch.zeros(n_steps, n_envs, dtype=torch.float32)
            self.x_t = torch.zeros(n_steps, n_envs, hidden_dim, dtype=torch.float32)
            self.o_hat = torch.zeros(n_steps, n_envs, hidden_dim, dtype=torch.float32)
        else:
            self.omegas = None
            self.surprises = None
            self.x_t = None
            self.o_hat = None

        self.ptr = 0

    def add(
        self,
        obs,
        action,
        reward,
        done,
        value,
        log_prob,
        omega=None,
        surprise=None,
        x_t=None,
        o_hat=None,
    ):
        self.obs[self.ptr] = torch.as_tensor(obs, dtype=torch.float32)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.long)
        self.rewards[self.ptr] = torch.as_tensor(reward, dtype=torch.float32)
        self.dones[self.ptr] = torch.as_tensor(done, dtype=torch.float32)
        self.values[self.ptr] = value.detach().cpu()
        self.log_probs[self.ptr] = log_prob.detach().cpu()

        if self.store_aod_fields:
            if omega is not None:
                self.omegas[self.ptr] = omega.detach().cpu()
            if surprise is not None:
                self.surprises[self.ptr] = surprise.detach().cpu()
            if x_t is not None:
                self.x_t[self.ptr] = x_t.detach().cpu()
            if o_hat is not None:
                self.o_hat[self.ptr] = o_hat.detach().cpu()

        self.ptr += 1

    def compute_returns_and_advantages(self, last_value, last_done, gamma, gae_lambda):
        last_value = last_value.detach().cpu()
        last_done = torch.as_tensor(last_done, dtype=torch.float32)

        gae = torch.zeros(self.n_envs, dtype=torch.float32)
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    def normalize_advantages(self):
        """Normalize advantages globally. Call once before PPO epochs."""
        self.advantages = (
            (self.advantages - self.advantages.mean())
            / (self.advantages.std() + 1e-8)
        )

    def get_sequence_batches(self, env_batch_size):
        """
        Yield contiguous rollout sequences for subsets of environments.

        Each yielded batch contains the full temporal sequence (T timesteps)
        for a subset of environments (env_batch_size envs). This preserves
        the sequential structure needed by recurrent and attention models.

        Returns tensors with leading dimensions (T, B,...), where:
            T = n_steps
            B = number of environments in this minibatch
        """
        env_indices = torch.randperm(self.n_envs)

        for start in range(0, self.n_envs, env_batch_size):
            idx = env_indices[start : start + env_batch_size]

            batch = {
                "obs": self.obs[:, idx].to(self.device),
                "actions": self.actions[:, idx].to(self.device),
                "log_probs": self.log_probs[:, idx].to(self.device),
                "advantages": self.advantages[:, idx].to(self.device),
                "returns": self.returns[:, idx].to(self.device),
                "dones": self.dones[:, idx].to(self.device),
            }

            if self.store_aod_fields:
                batch.update({
                    "omegas": self.omegas[:, idx].to(self.device),
                    "surprises": self.surprises[:, idx].to(self.device),
                    "x_t": self.x_t[:, idx].to(self.device),
                    "o_hat": self.o_hat[:, idx].to(self.device),
                })

            yield batch

    def reset(self):
        self.ptr = 0


class PPOTrainer:
    def __init__(self, model, envs, device, args, logger, save_dir):
        self.model = model
        self.envs = envs
        self.device = device
        self.args = args
        self.logger = logger
        self.save_dir = save_dir

        self.optimizer = Adam(
            self._base_model.parameters(), lr=args.lr, eps=1e-5
        )

        self.obs_shape = envs.observation_space.shape
        self.hidden_dim = getattr(
            self._base_model, "hidden_dim", args.hidden_dim
        )

        self.buffer = RolloutBuffer(
            n_steps=args.n_steps,
            n_envs=args.n_envs,
            obs_shape=self.obs_shape,
            hidden_dim=self.hidden_dim,
            device=device,
            store_aod_fields=self._is_aod_model,
        )

        self.n_updates = 0
        self.global_step = 0
        self.total_updates = args.total_steps // (args.n_steps * args.n_envs)

    @property
    def _base_model(self):
        """Unwrap DataParallel or similar wrappers."""
        model = self.model
        if hasattr(model, "module"):
            model = model.module
        return model

    @property
    def _is_aod_model(self):
        """Check if model uses AoD-specific loss terms."""
        return getattr(self._base_model, "model_type", None) in (
            "aod",
            "fixed_mixture",
        )

    def _detach_hidden(self, hidden):
        """Detach hidden state from computation graph."""
        if hidden is None:
            return None
        if isinstance(hidden, tuple):
            return tuple(
                h.detach() if h is not None else None for h in hidden
            )
        return hidden.detach()

    def _reset_hidden_for_done(self, hidden, done, inplace=False):
        """Zero out hidden states for environments that terminated."""
        if hidden is None:
            return None

        done_t = torch.as_tensor(done, dtype=torch.bool, device=self.device)

        if not done_t.any():
            return hidden

        if isinstance(hidden, tuple):
            new_hidden = []
            for h in hidden:
                if h is not None and h.dim() >= 2:
                    if not inplace:
                        h = h.clone()
                    h[done_t] = 0.0
                new_hidden.append(h)
            return tuple(new_hidden)

        if not inplace:
            hidden = hidden.clone()
        hidden[done_t] = 0.0
        return hidden

    def _update_lr(self, update):
        """Linear learning rate annealing (optional, controlled by args.anneal_lr)."""
        if not getattr(self.args, "anneal_lr", False):
            return
        frac = 1.0 - update / self.total_updates
        lr = self.args.lr * frac
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_env_batch_size(self):
        """
        Determine how many environments per optimization minibatch.

        If args.env_batch_size is set, use it directly.
        Otherwise, derive from args.batch_size (interpreted as total
        timestep-equivalent units): env_batch_size = batch_size // n_steps.
        """
        if hasattr(self.args, "env_batch_size") and self.args.env_batch_size is not None:
            return self.args.env_batch_size
        return max(1, self.args.batch_size // self.args.n_steps)

    def train(self):
        args = self.args
        obs, _ = self.envs.reset()
        obs = torch.as_tensor(obs, dtype=torch.float32)
        hidden = self._base_model.init_hidden(args.n_envs, self.device)

        episode_returns = np.zeros(args.n_envs, dtype=np.float32)
        completed_returns = deque(maxlen=100)

        start_time = time.time()
        done = np.zeros(args.n_envs, dtype=np.float32)

        for update in range(self.total_updates):
            self._update_lr(update)
            self.model.eval()
            self.buffer.reset()

            # ── Rollout phase ──
            for _ in range(args.n_steps):
                with torch.no_grad():
                    obs_dev = obs.to(self.device)
                    logits, value, hidden, info = self.model(obs_dev, hidden)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                next_obs, reward, terminated, truncated, _ = self.envs.step(
                    action.cpu().numpy()
                )
                done = (terminated | truncated).astype(np.float32)

                # Extract AoD info (None for non-AoD models)
                omega = info.get("omega")
                surprise = info.get("surprise")
                x_t = info.get("x_t")
                o_hat = info.get("o_hat")

                self.buffer.add(
                    obs=obs.numpy(),
                    action=action.cpu().numpy(),
                    reward=reward,
                    done=done,
                    value=value,
                    log_prob=log_prob,
                    omega=omega,
                    surprise=surprise,
                    x_t=x_t,
                    o_hat=o_hat,
                )

                episode_returns += reward
                for i, d in enumerate(done):
                    if d:
                        completed_returns.append(float(episode_returns[i]))
                        episode_returns[i] = 0.0

                obs = torch.as_tensor(next_obs, dtype=torch.float32)
                # inplace=True is safe under no_grad
                hidden = self._reset_hidden_for_done(hidden, done, inplace=True)

            # ── Bootstrap final value ──
            with torch.no_grad():
                _, last_value, _, _ = self.model(obs.to(self.device), hidden)

            self.buffer.compute_returns_and_advantages(
                last_value=last_value,
                last_done=done,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )

            # ── PPO update phase ──
            self.model.train()
            self.buffer.normalize_advantages()
            metrics = self._update_sequence_batches()

            self.n_updates += 1
            self.global_step += args.n_steps * args.n_envs

            # ── Logging ──
            if update % args.log_interval == 0:
                fps = self.global_step / max(time.time() - start_time, 1e-8)
                mean_return = (
                    np.mean(completed_returns) if completed_returns else 0.0
                )

                log_data = {
                    "update": update,
                    "global_step": self.global_step,
                    "mean_return": mean_return,
                    "fps": fps,
                    **metrics,
                }

                if self._is_aod_model and self.buffer.omegas is not None:
                    log_data["mean_omega"] = self.buffer.omegas.mean().item()
                    log_data["mean_surprise"] = self.buffer.surprises.mean().item()

                self.logger.log(log_data)

                omega_str = ""
                surprise_str = ""
                if self._is_aod_model and self.buffer.omegas is not None:
                    omega_str = f" | ω {log_data['mean_omega']:.3f}"
                    surprise_str = f" | Surprise {log_data['mean_surprise']:.4f}"

                print(
                    f"Update {update:5d}/{self.total_updates} | "
                    f"Steps {self.global_step:>10,} | "
                    f"Return {mean_return:6.2f}"
                    f"{omega_str}{surprise_str}"
                    f" | FPS {fps:5.0f}"
                )

            if update > 0 and update % args.save_interval == 0:
                self._save(update)

        self._save("final")

    def _update_sequence_batches(self):
        args = self.args
        all_metrics = []

        env_batch_size = self._get_env_batch_size()

        for _ in range(args.n_epochs):
            for batch in self.buffer.get_sequence_batches(env_batch_size):
                obs_seq = batch["obs"]                  # (T, B, H, W, C)
                actions_seq = batch["actions"]          # (T, B)
                old_log_probs_seq = batch["log_probs"]  # (T, B)
                advantages_seq = batch["advantages"]    # (T, B)
                returns_seq = batch["returns"]          # (T, B)
                dones_seq = batch["dones"]              # (T, B)

                T = obs_seq.shape[0]
                batch_envs = obs_seq.shape[1]
                hidden = self._base_model.init_hidden(
                    batch_envs, self.device
                )

                l_rl_acc = 0.0
                l_v_acc = 0.0
                l_ent_acc = 0.0
                l_compute_acc = 0.0
                gate_entropy_acc = 0.0

                # For prediction loss: collect predictions and targets
                # across the sequence to compute the one-step-offset loss.
                predictions = []  # o_hat[t] predicts x_{t+1}
                targets = []      # x_t[t] is the target for o_hat[t-1]

                for t in range(T):
                    # Reset hidden for environments that terminated at the
                    # previous timestep, so the model sees a clean state
                    # at the start of a new episode.
                    if t > 0:
                        hidden = self._reset_hidden_for_done(
                            hidden,
                            dones_seq[t - 1].detach().cpu().numpy(),
                        )

                    logits, values, hidden, info = self.model(
                        obs_seq[t], hidden
                    )

                    # Detach hidden state: no TBPTT (standard PPO)
                    hidden = self._detach_hidden(hidden)

                    dist = torch.distributions.Categorical(logits=logits)
                    log_probs = dist.log_prob(actions_seq[t])
                    entropy = dist.entropy()

                    # ── PPO clipped surrogate ──
                    ratio = torch.exp(log_probs - old_log_probs_seq[t])
                    surr1 = ratio * advantages_seq[t]
                    surr2 = (
                        torch.clamp(
                            ratio, 1 - args.clip_eps, 1 + args.clip_eps
                        )
                        * advantages_seq[t]
                    )
                    l_rl = -torch.min(surr1, surr2).mean()

                    # Value loss: unclipped MSE. See module docstring for
                    # rationale. Clipped value loss is an alternative but
                    # can cause underestimation in some settings.
                    l_v = F.mse_loss(values, returns_seq[t])

                    l_ent = entropy.mean()

                    l_rl_acc = l_rl_acc + l_rl
                    l_v_acc = l_v_acc + l_v
                    l_ent_acc = l_ent_acc + l_ent

                    # ── AoD-specific terms ──
                    if self._is_aod_model and "o_hat" in info:
                        predictions.append(info["o_hat"])
                        targets.append(info["x_t"].detach())

                        omega_t = info["omega"].clamp(1e-8, 1.0 - 1e-8)
                        l_compute = omega_t.mean()

                        gate_entropy = -(
                            omega_t * torch.log(omega_t)
                            + (1.0 - omega_t) * torch.log(1.0 - omega_t)
                        ).mean()

                        l_compute_acc = l_compute_acc + l_compute
                        gate_entropy_acc = gate_entropy_acc + gate_entropy

                # ── Average losses over time ──
                l_rl_acc = l_rl_acc / T
                l_v_acc = l_v_acc / T
                l_ent_acc = l_ent_acc / T

                l_ppo = (
                    l_rl_acc
                    + args.vf_coef * l_v_acc
                    - args.ent_coef * l_ent_acc
                )

                if self._is_aod_model and predictions:
                    l_compute_acc = l_compute_acc / T
                    gate_entropy_acc = gate_entropy_acc / T

                    # Prediction loss with one-step offset:
                    # o_hat[t] should predict x_{t+1}.
                    # Mask out episode boundaries: if dones_seq[t] = 1,
                    # then x_{t+1} belongs to a new episode and should
                    # not be used as a target for o_hat[t].
                    if len(predictions) > 1:
                        preds = torch.stack(predictions[:-1])  # (T-1, B, D)
                        tgts = torch.stack(targets[1:])        # (T-1, B, D)
                        valid = 1.0 - dones_seq[:-1]           # (T-1, B)

                        per_elem = F.mse_loss(
                            preds, tgts, reduction="none"
                        )                                      # (T-1, B, D)
                        per_pair = per_elem.mean(dim=-1)       # (T-1, B)
                        denom = valid.sum().clamp_min(1.0)
                        l_pred = (per_pair * valid).sum() / denom
                    else:
                        l_pred = torch.tensor(0.0, device=self.device)

                    loss = (
                        l_ppo
                        + args.alpha_pred * l_pred
                        + args.lambda_compute * l_compute_acc
                        - args.beta_gate * gate_entropy_acc
                    )

                    all_metrics.append({
                        "loss": loss.item(),
                        "l_rl": l_rl_acc.item(),
                        "l_v": l_v_acc.item(),
                        "l_pred": l_pred.item(),
                        "l_compute": l_compute_acc.item(),
                        "gate_entropy": gate_entropy_acc.item(),
                    })
                else:
                    loss = l_ppo
                    all_metrics.append({
                        "loss": loss.item(),
                        "l_rl": l_rl_acc.item(),
                        "l_v": l_v_acc.item(),
                    })

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self._base_model.parameters(), args.max_grad_norm
                )
                self.optimizer.step()

        # Average metrics across all mini-batches and epochs
        if not all_metrics:
            return {}
        avg = {}
        for k in all_metrics[0]:
            avg[k] = float(np.mean([m[k] for m in all_metrics]))
        return avg

    def _save(self, tag):
        path = os.path.join(self.save_dir, f"model_{tag}.pt")
        torch.save(
            {
                "model_state": self._base_model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "n_updates": self.n_updates,
            },
            path,
        )

    def load(self, path):
        """Resume training from a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self._base_model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.global_step = checkpoint["global_step"]
        self.n_updates = checkpoint["n_updates"]
        print(f"Resumed from {path} (step {self.global_step:,})")