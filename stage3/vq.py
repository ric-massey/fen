"""
Vector-quantised bottleneck — the organism's own symbols.

Continuous internal state is forced through a learned discrete codebook. The
system develops its own vocabulary of distinctions, grounded in what it actually
processes rather than in anything we named.

This is also the global workspace. Rich parallel activity compressed through a
narrow channel, with the winner broadcast to every downstream consumer. Baars'
bottleneck and a VQ codebook are the same equation approached from two
directions — build one and you get the other.

Codebook collapse is the standard failure: everything maps to two or three codes
and the vocabulary is dead. Three mitigations here, all necessary in practice:

  EMA updates      codebook entries track a running mean of what maps to them,
                   rather than being pushed around by gradients
  dead-code revival unused entries are reseeded from live encoder outputs, so a
                   code that loses its cluster gets another chance
  commitment loss  keeps encoder outputs from drifting away from the codebook
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQ(nn.Module):
    def __init__(self, n_codes: int = 128, dim: int = 64, decay: float = 0.99,
                 commitment: float = 0.25, eps: float = 1e-5,
                 revive_threshold: float = 1.0):
        super().__init__()
        self.n_codes = n_codes
        self.dim = dim
        self.decay = decay
        self.commitment = commitment
        self.eps = eps
        self.revive_threshold = revive_threshold

        embed = torch.randn(n_codes, dim) * 0.5
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_codes))
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("usage", torch.zeros(n_codes))

    def _distances(self, z: torch.Tensor) -> torch.Tensor:
        # ||z - e||^2 expanded, avoiding an explicit outer difference
        return (z.pow(2).sum(-1, keepdim=True)
                - 2 * z @ self.embed.t()
                + self.embed.pow(2).sum(-1))

    def forward(self, z: torch.Tensor):
        """z: (..., dim). Returns quantised z, indices, loss, stats."""
        shape = z.shape
        flat = z.reshape(-1, self.dim)

        idx = self._distances(flat).argmin(dim=-1)
        quantised = self.embed[idx].view(shape)

        if self.training:
            self._ema_update(flat, idx)

        # Commitment: pull the encoder toward the code it selected. The codebook
        # itself is updated by EMA, not by this gradient.
        commit_loss = F.mse_loss(z, quantised.detach())
        loss = self.commitment * commit_loss

        # Straight-through: forward uses the code, backward passes through to z.
        quantised = z + (quantised - z).detach()

        return quantised, idx.view(shape[:-1]), loss

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, idx: torch.Tensor) -> None:
        onehot = F.one_hot(idx, self.n_codes).type(flat.dtype)
        counts = onehot.sum(0)
        summed = onehot.t() @ flat

        self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(summed, alpha=1 - self.decay)
        self.usage.mul_(0.99).add_(counts, alpha=0.01)

        n = self.cluster_size.sum()
        normalised = ((self.cluster_size + self.eps) / (n + self.n_codes * self.eps)) * n
        self.embed.copy_(self.embed_avg / normalised.unsqueeze(1))

        self._revive(flat)

    @torch.no_grad()
    def _revive(self, flat: torch.Tensor) -> None:
        """Reseed codes that have lost their cluster from live encoder outputs."""
        dead = self.cluster_size < self.revive_threshold
        n_dead = int(dead.sum())
        if n_dead == 0 or flat.shape[0] == 0:
            return
        pick = torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)
        self.embed[dead] = flat[pick]
        self.embed_avg[dead] = flat[pick]
        self.cluster_size[dead] = 1.0

    @torch.no_grad()
    def stats(self) -> dict:
        """Utilisation is the first thing to check — a collapsed codebook has
        high accuracy and no vocabulary."""
        p = self.usage / (self.usage.sum() + 1e-9)
        live = int((self.cluster_size >= self.revive_threshold).sum())
        entropy = float(-(p * (p + 1e-12).log()).sum())
        return {
            "live_codes": live,
            "n_codes": self.n_codes,
            "usage_entropy": entropy,
            "max_entropy": float(torch.log(torch.tensor(float(self.n_codes)))),
            "perplexity": float(torch.exp(torch.tensor(entropy))),
        }
