"""
Forward model, efference-copy surprise, frame stacking, and the buffer.

The model predicts the sensory consequence of its own action as a DISTRIBUTION,
not a point:

    f(history, action) -> mu, logvar

Surprise is the negative log likelihood of what actually happened under that
distribution. This matters for two reasons:

  1. It is calibrated. A raw residual is an uncalibrated distance — large in
     regions where the dynamics are inherently hard, small where they are easy,
     regardless of whether anything external happened.

  2. It separates "the world acted on me" from "my self-model is wrong." A point
     model conflates them: a creature with a bad self-model would conclude it was
     constantly being pushed. Predicting variance lets the system ask the real
     question — is this larger than I *should* expect here?
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn

LOGVAR_MIN, LOGVAR_MAX = -8.0, 4.0


class FrameStack:
    """Rolling history of (obs, action) pairs.

    Cheap test of the Stage 3 hypothesis: if temporal context lowers the
    detection threshold, recurrence will help. A single-step model cannot
    distinguish a small sustained external force from ordinary dynamics,
    because within one transition they look identical.
    """

    def __init__(self, k: int, obs_dim: int, act_dim: int):
        self.k = k
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_hist: deque = deque(maxlen=k)
        self.act_hist: deque = deque(maxlen=k - 1) if k > 1 else deque(maxlen=1)
        self.feat_dim = k * obs_dim + (k - 1) * act_dim

    def reset(self, obs: np.ndarray) -> None:
        self.obs_hist.clear()
        self.act_hist.clear()
        for _ in range(self.k):
            self.obs_hist.append(obs.copy())
        for _ in range(max(self.k - 1, 0)):
            self.act_hist.append(np.zeros(self.act_dim, dtype=np.float32))

    def feature(self) -> np.ndarray:
        parts = list(self.obs_hist)
        if self.k > 1:
            parts += list(self.act_hist)
        return np.concatenate(parts).astype(np.float32)

    def push(self, obs: np.ndarray, act: np.ndarray) -> None:
        if self.k > 1:
            self.act_hist.append(act.astype(np.float32))
        self.obs_hist.append(obs.copy())


class ForwardModel(nn.Module):
    def __init__(self, feat_dim: int, act_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.feat_dim = feat_dim
        self.act_dim = act_dim
        self.out_dim = out_dim
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim + act_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden, out_dim)
        self.logvar_head = nn.Linear(hidden, out_dim)

        self.register_buffer("feat_mean", torch.zeros(feat_dim))
        self.register_buffer("feat_std", torch.ones(feat_dim))
        self.register_buffer("delta_mean", torch.zeros(out_dim))
        self.register_buffer("delta_std", torch.ones(out_dim))

    def fit_normalisation(self, feat: np.ndarray, delta: np.ndarray) -> None:
        self.feat_mean.copy_(torch.tensor(feat.mean(0), dtype=torch.float32))
        self.feat_std.copy_(torch.tensor(feat.std(0) + 1e-6, dtype=torch.float32))
        self.delta_mean.copy_(torch.tensor(delta.mean(0), dtype=torch.float32))
        self.delta_std.copy_(torch.tensor(delta.std(0) + 1e-6, dtype=torch.float32))

    def forward(self, feat: torch.Tensor, act: torch.Tensor):
        f = (feat - self.feat_mean) / self.feat_std
        h = self.trunk(torch.cat([f, act], dim=-1))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(LOGVAR_MIN, LOGVAR_MAX)
        return mu, logvar

    def _target(self, delta: torch.Tensor) -> torch.Tensor:
        return (delta - self.delta_mean) / self.delta_std

    def nll(self, feat, act, delta, reduce: bool = True):
        """Gaussian negative log likelihood, per sample or averaged."""
        mu, logvar = self.forward(feat, act)
        target = self._target(delta)
        per_dim = 0.5 * (((target - mu) ** 2) / logvar.exp() + logvar)
        per_sample = per_dim.mean(dim=-1)
        return per_sample.mean() if reduce else per_sample

    def huber(self, feat, act, delta, delta_h: float = 1.0):
        """Point loss, used as a warmup before variance learning is enabled.

        Heteroscedastic models trained from scratch often collapse to predicting
        huge variance everywhere, which lowers the loss without predicting
        anything. Warming up on a point loss avoids that.
        """
        mu, _ = self.forward(feat, act)
        return nn.functional.huber_loss(mu, self._target(delta), delta=delta_h)

    @torch.no_grad()
    def surprise(self, feat, act, delta) -> torch.Tensor:
        """Per-sample calibrated surprise. This is the efference-copy signal."""
        return self.nll(feat, act, delta, reduce=False)

    @torch.no_grad()
    def point_residual(self, feat, act, delta) -> torch.Tensor:
        """Uncalibrated residual, kept for comparison against the old metric."""
        mu, _ = self.forward(feat, act)
        return (mu - self._target(delta)).pow(2).mean(dim=-1).sqrt()


class Buffer:
    """Rolling transition buffer with a held-out split.

    No episodes — a sliding window over one life. A fixed fraction of
    transitions is reserved at insert time and never sampled for training, so
    the in-loop metric is not scored on data the model fitted.
    """

    def __init__(self, capacity: int, feat_dim: int, act_dim: int, out_dim: int,
                 holdout: float = 0.15, seed: int = 0):
        self.capacity = capacity
        self.feat = np.zeros((capacity, feat_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.delta = np.zeros((capacity, out_dim), dtype=np.float32)
        self.perturbed = np.zeros(capacity, dtype=bool)
        self.ramp = np.zeros(capacity, dtype=bool)
        self.is_eval = np.zeros(capacity, dtype=bool)
        self.holdout = holdout
        self.rng = np.random.default_rng(seed)
        self.n = 0
        self.i = 0

    def add(self, feat, act, delta, perturbed, ramp) -> None:
        j = self.i
        self.feat[j] = feat
        self.act[j] = act
        self.delta[j] = delta
        self.perturbed[j] = perturbed
        self.ramp[j] = ramp
        self.is_eval[j] = self.rng.random() < self.holdout
        self.i = (self.i + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def sample(self, batch: int, rng: np.random.Generator, clean_only: bool = False):
        """Training sample. Excludes held-out transitions.

        clean_only=False is the honest default: a real organism has no label
        telling it which events were externally caused, so it cannot filter them
        out. Perturbations are handled by the loss (Huber warmup, then NLL,
        which learns to widen its variance where things are unpredictable).

        clean_only=True uses the privileged label and is an upper-bound
        reference only. Never report it as the result.
        """
        mask = ~self.is_eval[: self.n]
        if clean_only:
            mask &= ~self.perturbed[: self.n]
        pool = np.flatnonzero(mask)
        if len(pool) == 0:
            return None
        idx = rng.choice(pool, size=min(batch, len(pool)), replace=False)
        return self.feat[idx], self.act[idx], self.delta[idx]

    def eval_split(self):
        m = self.is_eval[: self.n]
        return (self.feat[: self.n][m], self.act[: self.n][m], self.delta[: self.n][m],
                self.perturbed[: self.n][m], self.ramp[: self.n][m])

    def all(self):
        return (self.feat[: self.n], self.act[: self.n], self.delta[: self.n],
                self.perturbed[: self.n], self.ramp[: self.n])
