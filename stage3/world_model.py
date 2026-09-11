"""
Recurrent state + discrete bottleneck: the Stage 3 core.

    (obs, action) -> GRU -> h -> encoder -> z -> VQ -> code -> predict next obs

Two things at once, and they are the same structure:

  Persistent internal state. The GRU carries information across time, so the
  system can represent things that unfold over many steps. Stage 2's unsolved
  failure was managing accumulated health damage — a variable that moves over
  ~1000 steps and depends on the history of your regulation, not your current
  state. A feedforward policy cannot represent "I have been running hot for a
  while and I am paying for it." This is the capacity that fixes that, and if
  Stage 3 does not fix it the explanation was wrong.

  Learned symbols. The bottleneck forces continuous experience into a discrete
  vocabulary the organism develops for itself.

The bottleneck sits on the path to prediction, which is what makes the codes
mean anything: a code has to carry enough about the current situation to predict
what happens next, so it is forced to encode real world structure rather than
arbitrary partitions.

Trained self-supervised, deliberately. Prediction error is dense and free, and
it decouples "do the symbols work" from "does recurrent RL train" — two hard
problems that should not be debugged at the same time.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from vq import VQ


class WorldModel(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128,
                 code_dim: int = 64, n_codes: int = 128, mlp: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden

        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim + act_dim, mlp), nn.SiLU(),
            nn.Linear(mlp, hidden),
        )
        self.gru = nn.GRUCell(hidden, hidden)
        self.to_code = nn.Sequential(nn.Linear(hidden, mlp), nn.SiLU(),
                                     nn.Linear(mlp, code_dim))
        self.vq = VQ(n_codes=n_codes, dim=code_dim)

        # Prediction reads ONLY the code plus the current action. Denying it the
        # raw observation is what forces the code to carry the state — otherwise
        # the bottleneck is decorative and the model routes around it.
        self.predict = nn.Sequential(
            nn.Linear(code_dim + act_dim, mlp), nn.SiLU(),
            nn.Linear(mlp, mlp), nn.SiLU(),
            nn.Linear(mlp, obs_dim),
        )

        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))
        self.register_buffer("delta_mean", torch.zeros(obs_dim))
        self.register_buffer("delta_std", torch.ones(obs_dim))

    def fit_normalisation(self, obs, delta) -> None:
        self.obs_mean.copy_(torch.as_tensor(obs.mean(0), dtype=torch.float32))
        self.obs_std.copy_(torch.as_tensor(obs.std(0) + 1e-6, dtype=torch.float32))
        self.delta_mean.copy_(torch.as_tensor(delta.mean(0), dtype=torch.float32))
        self.delta_std.copy_(torch.as_tensor(delta.std(0) + 1e-6, dtype=torch.float32))

    def init_state(self, batch: int, device="cpu") -> torch.Tensor:
        return torch.zeros(batch, self.hidden, device=device)

    def step(self, obs, act, h):
        """One timestep. Returns (code, indices, next_h, vq_loss)."""
        o = (obs - self.obs_mean) / self.obs_std
        x = self.input_proj(torch.cat([o, act], dim=-1))
        h = self.gru(x, h)
        z = self.to_code(h)
        code, idx, vq_loss = self.vq(z)
        return code, idx, h, vq_loss

    def forward(self, obs_seq, act_seq, h=None, burn_in: int = 0):
        """Roll a sequence. obs_seq/act_seq: (B, T, dim).

        burn_in steps rebuild the recurrent state without contributing loss —
        the standard fix for replaying stored sequences whose hidden state was
        not saved. Losses computed over a state that started from zeros are
        systematically wrong for the early steps.
        """
        B, T, _ = obs_seq.shape
        if h is None:
            h = self.init_state(B, obs_seq.device)

        preds, idxs, vq_losses = [], [], []
        for t in range(T):
            code, idx, h, vq_loss = self.step(obs_seq[:, t], act_seq[:, t], h)
            if t >= burn_in:
                preds.append(self.predict(torch.cat([code, act_seq[:, t]], dim=-1)))
                idxs.append(idx)
                vq_losses.append(vq_loss)
        return (torch.stack(preds, 1), torch.stack(idxs, 1),
                torch.stack(vq_losses).mean(), h)

    def loss(self, obs_seq, act_seq, delta_seq, burn_in: int = 0):
        pred, idx, vq_loss, _ = self.forward(obs_seq, act_seq, burn_in=burn_in)
        target = (delta_seq[:, burn_in:] - self.delta_mean) / self.delta_std
        pred_loss = nn.functional.huber_loss(pred, target)
        return pred_loss + vq_loss, {"pred": float(pred_loss), "vq": float(vq_loss)}, idx


class NoRecurrence(WorldModel):
    """Ablation: identical capacity, no memory across steps.

    The control for every claim Stage 3 makes about recurrence. If this matches
    the recurrent model, the GRU is decoration.
    """

    def step(self, obs, act, h):
        o = (obs - self.obs_mean) / self.obs_std
        x = self.input_proj(torch.cat([o, act], dim=-1))
        h = self.gru(x, torch.zeros_like(h))   # state reset every step
        z = self.to_code(h)
        code, idx, vq_loss = self.vq(z)
        return code, idx, h, vq_loss
