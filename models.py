"""
models.py — All policy architectures for AoD paper

Models:
    - MLPPolicy          : memoryless baseline
    - GRUPolicy          : recurrent baseline
    - TransformerPolicy  : full-attention baseline
    - FixedMixturePolicy : fixed ω=0.5 hybrid baseline
    - AoDPolicy          : Attention-on-Demand (ours)

Weight initialization:
    All modules use PyTorch defaults (Kaiming uniform for Linear, Xavier-like
    for Conv2d). The gate network's final Sigmoid receives near-zero input at
    init, producing ω ≈ 0.5 (equal recurrent/attention mix), which is a
    reasonable starting point.

History buffer note:
    TransformerPolicy and AoDPolicy maintain a history buffer of encoded
    observations. During PPO updates, these encodings were computed under
    the rollout-phase parameters and become stale as the policy updates.
    This is a standard tradeoff for transformer-based RL policies and is
    shared by prior work (e.g., GTrXL).
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# Shared observation encoder
# ─────────────────────────────────────────────

class ObsEncoder(nn.Module):
    """
    Encodes MiniGrid observations (H, W, C) into a hidden vector.

    Note:
        MiniGrid default observations are symbolic integer encodings, not RGB
        images. We therefore do NOT divide by 255 by default. Set normalize_by
        to 255.0 if using an RGB observation wrapper.
    """

    def __init__(self, obs_shape, hidden_dim, normalize_by=None):
        super().__init__()
        h, w, c = obs_shape
        self.normalize_by = normalize_by

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            cnn_out = self.cnn(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(cnn_out, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        """
        Args:
            obs: (B, H, W, C) or (B, T, H, W, C)
        Returns:
            (B, D) or (B*T, D)
        """
        if obs.dim() == 4:
            x = obs.permute(0, 3, 1, 2).float()
        elif obs.dim() == 5:
            x = obs.permute(0, 1, 4, 2, 3).float()
            b, t = x.shape[:2]
            x = x.reshape(b * t, *x.shape[2:])
        else:
            raise ValueError(f"Unexpected observation shape: {tuple(obs.shape)}")

        if self.normalize_by is not None and self.normalize_by != 0:
            x = x / self.normalize_by

        x = self.cnn(x)
        x = self.fc(x)
        return x


# ─────────────────────────────────────────────
# Shared policy / value heads
# ─────────────────────────────────────────────

class PolicyHead(nn.Module):
    def __init__(self, hidden_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, z):
        return self.net(z)


class ValueHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


# ─────────────────────────────────────────────
# 1. MLP baseline
# ─────────────────────────────────────────────

class MLPPolicy(nn.Module):
    def __init__(self, obs_shape, n_actions, hidden_dim=128, **kwargs):
        super().__init__()
        self.encoder = ObsEncoder(obs_shape, hidden_dim)
        self.policy_head = PolicyHead(hidden_dim, n_actions)
        self.value_head = ValueHead(hidden_dim)
        self.hidden_dim = hidden_dim
        self.model_type = "mlp"

    def forward(self, obs, hidden=None):
        z = self.encoder(obs)
        logits = self.policy_head(z)
        value = self.value_head(z)
        return logits, value, None, {}

    def init_hidden(self, batch_size, device):
        return None


# ─────────────────────────────────────────────
# 2. GRU baseline
# ─────────────────────────────────────────────

class GRUPolicy(nn.Module):
    def __init__(self, obs_shape, n_actions, hidden_dim=128, **kwargs):
        super().__init__()
        self.encoder = ObsEncoder(obs_shape, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.policy_head = PolicyHead(hidden_dim, n_actions)
        self.value_head = ValueHead(hidden_dim)
        self.hidden_dim = hidden_dim
        self.model_type = "gru"

    def forward(self, obs, hidden=None):
        x = self.encoder(obs)
        if hidden is None:
            hidden = self.init_hidden(x.shape[0], x.device)
        m = self.gru(x, hidden)
        logits = self.policy_head(m)
        value = self.value_head(m)
        return logits, value, m, {}

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)


# ─────────────────────────────────────────────
# 3. Transformer baseline
# ─────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, seq_len):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"n_heads ({n_heads})"
            )

        self.attn = nn.MultiheadAttention(
            hidden_dim,
            n_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # True = blocked. Upper triangular mask blocks future positions.
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool(),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D)
        """
        t = x.shape[1]
        mask = self.causal_mask[:t, :t]
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class TransformerPolicy(nn.Module):
    def __init__(self, obs_shape, n_actions, hidden_dim=128,
                 n_heads=4, seq_len=16, **kwargs):
        super().__init__()
        self.encoder = ObsEncoder(obs_shape, hidden_dim)
        self.pos_embed = nn.Embedding(seq_len, hidden_dim)
        self.transformer = CausalSelfAttention(hidden_dim, n_heads, seq_len)
        self.policy_head = PolicyHead(hidden_dim, n_actions)
        self.value_head = ValueHead(hidden_dim)
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.model_type = "transformer"

    def forward(self, obs, hidden=None):
        x = self.encoder(obs)  # (B, D)

        if hidden is None:
            hidden = self.init_hidden(x.shape[0], x.device)

        # Shift history buffer and append new encoding.
        # Note: roll returns a new tensor; the original hidden is not mutated.
        hidden = hidden.roll(shifts=-1, dims=1)
        hidden[:, -1, :] = x

        t = hidden.shape[1]
        pos = torch.arange(t, device=x.device)
        h = hidden + self.pos_embed(pos).unsqueeze(0)

        h = self.transformer(h)
        z = h[:, -1, :]  # use final position's output

        logits = self.policy_head(z)
        value = self.value_head(z)
        return logits, value, hidden, {}

    def init_hidden(self, batch_size, device):
        return torch.zeros(
            batch_size, self.seq_len, self.hidden_dim, device=device,
        )


# ─────────────────────────────────────────────
# AoD components
# ─────────────────────────────────────────────

class SurpriseSignal(nn.Module):
    """
    Surprise signal based on squared prediction error:
        s_t = ||o_t - ô_t||_2^2

    Design choice:
        We treat o_hat_prev as a detached signal across time. The prediction
        head is trained through the explicit prediction loss in the trainer,
        while surprise serves as a stable control signal for gating rather
        than a cross-timestep gradient path.

    Running statistics:
        Surprise values are normalized using exponential moving average
        statistics (like BatchNorm). The first batch initializes the running
        stats directly to avoid a slow warmup period.
    """

    def __init__(self, hidden_dim, obs_dim, ema_momentum=0.01):
        super().__init__()
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_var", torch.ones(1))
        self.register_buffer("_initialized", torch.tensor(False))
        self.momentum = ema_momentum

    def forward(self, m_t, o_t, o_hat_prev):
        """
        Args:
            m_t:        (B, D) recurrent state
            o_t:        (B, D) encoded current observation
            o_hat_prev: (B, D) predicted current observation from previous step

        Returns:
            s_t_norm:   (B, 1) normalized squared error
            o_hat_next: (B, D) next-step prediction
            s_t_raw:    (B,)   raw squared error for logging
        """
        # Predict next encoded observation
        o_hat_next = self.pred_head(m_t)

        # Squared L2 prediction error (detached: no gradient through o_hat_prev)
        diff = o_t - o_hat_prev.detach()
        s_t = (diff * diff).sum(dim=-1, keepdim=True)  # (B, 1)
        s_t_raw = s_t.squeeze(-1).detach()

        # Update running statistics for normalization
        if self.training:
            with torch.no_grad():
                batch_mean = s_t.mean()
                batch_var = s_t.var(unbiased=False)

                if not self._initialized:
                    # First batch: initialize directly for fast warmup
                    self.running_mean.copy_(batch_mean)
                    self.running_var.copy_(batch_var.clamp_min(1e-8))
                    self._initialized.fill_(True)
                else:
                    self.running_mean.mul_(1 - self.momentum).add_(
                        self.momentum * batch_mean
                    )
                    self.running_var.mul_(1 - self.momentum).add_(
                        self.momentum * batch_var
                    )

        s_t_norm = (s_t - self.running_mean) / (
            torch.sqrt(self.running_var) + 1e-8
        )
        return s_t_norm, o_hat_next, s_t_raw


class GatingMechanism(nn.Module):
    """
    ω_t = g_φ(o_t, m_t, s̃_t),  ω_t ∈ [0, 1]
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, o_enc, m_t, s_norm):
        """
        Args:
            o_enc:  (B, D) encoded observation
            m_t:    (B, D) recurrent state
            s_norm: (B, 1) normalized surprise
        Returns:
            (B, 1) gate value in [0, 1]
        """
        gate_input = torch.cat([o_enc, m_t, s_norm], dim=-1)
        return self.gate(gate_input)


# ─────────────────────────────────────────────
# 4. AoD policy
# ─────────────────────────────────────────────

class AoDPolicy(nn.Module):
    """
    Attention-on-Demand Policy.

    1. x_t      = Enc(o_t)
    2. m_t      = GRU(x_t, m_{t-1})
    3. s_t      = ||x_t - x̂_t||²        (surprise)
    4. z_attn   = Attention(x_1, ..., x_t)
    5. ω_t      = gate(x_t, m_t, s̃_t)
    6. z_t      = ω_t · z_attn + (1 - ω_t) · z_rec
    7. π, V     = heads(z_t)
    """

    def __init__(self, obs_shape, n_actions, hidden_dim=128,
                 n_heads=4, seq_len=16, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.model_type = "aod"

        # Shared encoder
        self.encoder = ObsEncoder(obs_shape, hidden_dim)

        # Recurrent branch
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.rec_proj = nn.Linear(hidden_dim, hidden_dim)

        # Surprise / prediction
        self.surprise = SurpriseSignal(hidden_dim, hidden_dim)

        # Attention branch
        self.pos_embed = nn.Embedding(seq_len, hidden_dim)
        self.attention = CausalSelfAttention(hidden_dim, n_heads, seq_len)
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim)

        # Gate
        self.gate = GatingMechanism(hidden_dim)

        # Output heads
        self.policy_head = PolicyHead(hidden_dim, n_actions)
        self.value_head = ValueHead(hidden_dim)

    def _compute_attention_branch(self, hist_buf):
        """Run causal self-attention over the history buffer."""
        t = hist_buf.shape[1]
        pos = torch.arange(t, device=hist_buf.device)
        h = hist_buf + self.pos_embed(pos).unsqueeze(0)
        h = self.attention(h)
        z_attn = self.attn_proj(h[:, -1, :])
        return z_attn

    def _compute_gate(self, x_t, m_t, s_norm, batch_size, device):
        """Compute gating value ω_t. Override in subclasses for ablations."""
        return self.gate(x_t, m_t, s_norm)  # (B, 1)

    def forward(self, obs, hidden=None):
        """
        Args:
            obs:    (B, H, W, C)
            hidden: tuple (m_prev, hist_buf, o_hat_prev) or None

        Returns:
            logits:     (B, n_actions) action logits
            value:      (B,) state value estimate
            new_hidden: tuple (m_t, hist_buf, o_hat_next)
            info:       dict with omega, surprise, o_hat, x_t
        """
        b = obs.shape[0]
        device = obs.device

        if hidden is None:
            hidden = self.init_hidden(b, device)

        m_prev, hist_buf, o_hat_prev = hidden

        # 1. Encode observation
        x_t = self.encoder(obs)  # (B, D)

        # 2. Recurrent branch
        m_t = self.gru(x_t, m_prev)
        z_rec = self.rec_proj(m_t)

        # 3. Surprise signal
        s_norm, o_hat_next, s_raw = self.surprise(m_t, x_t, o_hat_prev)

        # 4. Update attention history and compute attention branch
        hist_buf = hist_buf.roll(shifts=-1, dims=1)
        hist_buf[:, -1, :] = x_t
        z_attn = self._compute_attention_branch(hist_buf)

        # 5. Gate
        omega_t = self._compute_gate(x_t, m_t, s_norm, b, device)  # (B, 1)

        # 6. Fuse
        z_t = omega_t * z_attn + (1.0 - omega_t) * z_rec

        # 7. Policy / value heads
        logits = self.policy_head(z_t)
        value = self.value_head(z_t)

        new_hidden = (m_t, hist_buf, o_hat_next)
        info = {
            "omega": omega_t.squeeze(-1).detach(),  # (B,)
            "surprise": s_raw,                      # (B,)
            "o_hat": o_hat_next,                    # (B, D)
            "x_t": x_t,                             # (B, D)
        }
        return logits, value, new_hidden, info

    def init_hidden(self, batch_size, device):
        m = torch.zeros(batch_size, self.hidden_dim, device=device)
        buf = torch.zeros(
            batch_size, self.seq_len, self.hidden_dim, device=device,
        )
        o_hat = torch.zeros(batch_size, self.hidden_dim, device=device)
        return (m, buf, o_hat)


# ─────────────────────────────────────────────
# 5. Fixed mixture baseline
# ─────────────────────────────────────────────

class FixedMixturePolicy(AoDPolicy):
    """
    AoD variant with fixed ω=0.5 — ablation baseline.

    Inherits all of AoDPolicy but overrides only the gate computation.
    The learned gate network still exists (inherited) but is unused during
    the forward pass. The surprise signal is still computed for comparable
    logging and analysis.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = "fixed_mixture"

    def _compute_gate(self, x_t, m_t, s_norm, batch_size, device):
        """Fixed gate: always returns 0.5."""
        return torch.full((batch_size, 1), 0.5, device=device)


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

_MODEL_REGISTRY = {
    "mlp": MLPPolicy,
    "gru": GRUPolicy,
    "transformer": TransformerPolicy,
    "aod": AoDPolicy,
    "fixed_mixture": FixedMixturePolicy,
}

# Models that require attention-specific hyperparameters
_ATTENTION_MODELS = {"transformer", "aod", "fixed_mixture"}


def make_model(model_type, obs_shape, n_actions, hidden_dim=128,
               n_heads=4, seq_len=16):
    """
    Construct a policy model by name.

    Args:
        model_type: one of "mlp", "gru", "transformer", "aod", "fixed_mixture"
        obs_shape:  (H, W, C) observation shape
        n_actions:  number of discrete actions
        hidden_dim: hidden layer dimensionality (all models)
        n_heads:    attention heads (transformer, aod, fixed_mixture only)
        seq_len:    history length (transformer, aod, fixed_mixture only)

    Returns:
        nn.Module with forward(obs, hidden) -> (logits, value, new_hidden, info)
    """
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_type!r}. "
            f"Available: {sorted(_MODEL_REGISTRY.keys())}"
        )

    common = dict(
        obs_shape=obs_shape,
        n_actions=n_actions,
        hidden_dim=hidden_dim,
    )

    if model_type in _ATTENTION_MODELS:
        common["n_heads"] = n_heads
        common["seq_len"] = seq_len

    return _MODEL_REGISTRY[model_type](**common)