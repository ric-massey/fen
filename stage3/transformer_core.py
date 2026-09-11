"""
Transformer core for Stage 3 — a transformer with a body.

Drop-in replacement for the GRU in world_model.py, matching its interface so the
training and measurement code is unchanged.

What this is. At each cycle the organism embeds its current situation
(observation + action) as a token, appends it to a rolling context, attends
causally over that context, and emits a discrete code from the learned codebook.
That is a transformer generating one token at a time with a cached context — the
same loop an LLM runs — except the vocabulary was not given to it. The codebook
grew out of prediction pressure on a body that can die.

Two practical notes:

  Training is FASTER than the GRU version, not slower. A GRU forward over T steps
  is T sequential operations; causal attention does the whole sequence in one
  parallel pass. The recurrence that cost us wall-clock is exactly what attention
  removes.

  Context is a window, not unbounded. `context` must be at least the training
  sequence length or the online step() and the parallel forward() disagree about
  what the model can see, and the measurements stop meaning the same thing.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from vq import VQ


class Block(nn.Module):
    """Pre-norm causal self-attention block."""

    def __init__(self, d: int, heads: int, mlp_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_mult * d), nn.GELU(), nn.Linear(mlp_mult * d, d),
        )

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TransformerWorldModel(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128,
                 code_dim: int = 64, n_codes: int = 128, mlp: int = 256,
                 layers: int = 3, heads: int = 4, context: int = 64):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.hidden = hidden
        self.context = context

        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim + act_dim, mlp), nn.SiLU(), nn.Linear(mlp, hidden),
        )
        self.pos = nn.Parameter(torch.zeros(1, context, hidden))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(hidden, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(hidden)

        self.to_code = nn.Sequential(nn.Linear(hidden, mlp), nn.SiLU(),
                                     nn.Linear(mlp, code_dim))
        self.vq = VQ(n_codes=n_codes, dim=code_dim)

        # Prediction sees only the code and the action, never the raw
        # observation — otherwise the model routes around the bottleneck and
        # the symbols mean nothing.
        self.predict = nn.Sequential(
            nn.Linear(code_dim + act_dim, mlp), nn.SiLU(),
            nn.Linear(mlp, mlp), nn.SiLU(), nn.Linear(mlp, obs_dim),
        )

        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))
        self.register_buffer("delta_mean", torch.zeros(obs_dim))
        self.register_buffer("delta_std", torch.ones(obs_dim))

    # ── interface parity with WorldModel ─────────────────────────────────────

    def fit_normalisation(self, obs, delta) -> None:
        self.obs_mean.copy_(torch.as_tensor(obs.mean(0), dtype=torch.float32))
        self.obs_std.copy_(torch.as_tensor(obs.std(0) + 1e-6, dtype=torch.float32))
        self.delta_mean.copy_(torch.as_tensor(delta.mean(0), dtype=torch.float32))
        self.delta_std.copy_(torch.as_tensor(delta.std(0) + 1e-6, dtype=torch.float32))

    def init_state(self, batch: int, device="cpu") -> torch.Tensor:
        """State is the rolling context window of embedded tokens."""
        return torch.zeros(batch, 0, self.hidden, device=device)

    def _causal_mask(self, n: int, device) -> torch.Tensor:
        return torch.triu(torch.full((n, n), float("-inf"), device=device), diagonal=1)

    def _embed(self, obs, act):
        o = (obs - self.obs_mean) / self.obs_std
        return self.input_proj(torch.cat([o, act], dim=-1))

    def _trunk(self, x):
        n = x.shape[1]
        x = x + self.pos[:, :n]
        mask = self._causal_mask(n, x.device)
        for b in self.blocks:
            x = b(x, mask)
        return self.ln_f(x)

    def step(self, obs, act, h):
        """One cycle. h is the context window; returns it grown by one token."""
        tok = self._embed(obs, act).unsqueeze(1)
        win = torch.cat([h, tok], dim=1)[:, -self.context:]
        out = self._trunk(win)[:, -1]
        z = self.to_code(out)
        code, idx, vq_loss = self.vq(z)
        return code, idx, win, vq_loss

    def forward(self, obs_seq, act_seq, h=None, burn_in: int = 0):
        """Whole sequence in one parallel pass — the attention advantage."""
        B, T, _ = obs_seq.shape
        assert T <= self.context, f"sequence {T} exceeds context {self.context}"
        x = self._embed(obs_seq, act_seq)
        out = self._trunk(x)

        z = self.to_code(out[:, burn_in:])
        code, idx, vq_loss = self.vq(z)
        pred = self.predict(torch.cat([code, act_seq[:, burn_in:]], dim=-1))
        return pred, idx, vq_loss, x[:, -self.context:]

    def loss(self, obs_seq, act_seq, delta_seq, burn_in: int = 0):
        pred, idx, vq_loss, _ = self.forward(obs_seq, act_seq, burn_in=burn_in)
        target = (delta_seq[:, burn_in:] - self.delta_mean) / self.delta_std
        pred_loss = nn.functional.huber_loss(pred, target)
        return pred_loss + vq_loss, {"pred": float(pred_loss), "vq": float(vq_loss)}, idx


class NoAttention(TransformerWorldModel):
    """Ablation: identical capacity, no access to context.

    Each position attends only to itself, so the model has the same parameters
    and the same bottleneck but cannot use history. The control for every claim
    about what attention buys.
    """

    def _causal_mask(self, n: int, device):
        m = torch.full((n, n), float("-inf"), device=device)
        return m.fill_diagonal_(0.0)
