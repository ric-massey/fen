"""
The transformer as the organism, not as an observer of one.

Until now the transformer was a world model: trained separately, frozen, watching
while a small MLP did the acting. Handing that MLP the transformer's features
changed nothing, which is unsurprising — a spectator with a good vocabulary is
still a spectator.

Here the loop is:

    sense -> append to context -> attend over your own history -> emit a symbol
          -> act -> repeat

which is a transformer generating one token at a time with a cached context, the
same loop an LLM runs. The difference is that each token is a situation it named
for itself, and each output moves a body that can starve.

Why now and not earlier. Until the world gained weather drift, a roaming hazard
and per-life layouts, it was static within a life: whatever you learned on your
first circuit stayed true forever, so context bought nothing behaviourally. There
was nothing for memory to be *for*. There is now.

Design notes:

  The action head reads the QUANTISED code, not the continuous pre-bottleneck
  state. That is a deliberate constraint. If the policy could bypass the codebook
  the symbols would become decorative — the model would route around them and
  the vocabulary would stop meaning anything. Forcing action through the discrete
  channel is what keeps the symbols load-bearing.

  Prediction is kept as an auxiliary loss. It is dense and free where the reward
  is sparse and flat, and it is what made the codebook carve the world in the
  first place. Dropping it and training on reward alone reliably collapses the
  codebook.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_core import TransformerWorldModel

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


class TransformerActor(nn.Module):
    """Transformer core + an action head reading only the emitted symbol."""

    def __init__(self, obs_dim, act_dim, act_low, act_high, hidden=128,
                 code_dim=64, n_codes=128, context=64, layers=3, heads=4):
        super().__init__()
        self.core = TransformerWorldModel(obs_dim, act_dim, hidden=hidden,
                                          code_dim=code_dim, n_codes=n_codes,
                                          layers=layers, heads=heads, context=context)
        self.head = nn.Sequential(
            nn.Linear(code_dim, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 2 * act_dim),
        )
        self.act_dim = act_dim
        self.register_buffer("scale", torch.tensor((act_high - act_low) / 2.0))
        self.register_buffer("bias", torch.tensor((act_high + act_low) / 2.0))

    def _dist(self, code):
        mu, log_std = self.head(code).chunk(2, dim=-1)
        return mu, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def act_from_code(self, code, deterministic=False, with_logprob=True):
        mu, log_std = self._dist(code)
        dist = torch.distributions.Normal(mu, log_std.exp())
        u = mu if deterministic else dist.rsample()
        a = torch.tanh(u)
        logp = None
        if with_logprob:
            logp = dist.log_prob(u).sum(-1)
            logp -= (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return a * self.scale + self.bias, logp

    def step(self, obs, prev_act, h, deterministic=False):
        """One cycle of the organism. Returns action, symbol index, new context."""
        code, idx, h, vq_loss = self.core.step(obs, prev_act, h)
        a, _ = self.act_from_code(code, deterministic=deterministic, with_logprob=False)
        return a, idx, h, vq_loss

    def sequence(self, obs_seq, act_seq, burn_in=0):
        """Parallel pass over a stored sequence — the attention advantage.

        Returns codes and the auxiliary prediction loss. Burn-in steps rebuild
        context without contributing loss; a state started from zeros is
        systematically wrong for the first steps.
        """
        x = self.core._embed(obs_seq, act_seq)
        out = self.core._trunk(x)
        z = self.core.to_code(out[:, burn_in:])
        code, idx, vq_loss = self.core.vq(z)
        pred = self.core.predict(torch.cat([code, act_seq[:, burn_in:]], dim=-1))
        return code, idx, pred, vq_loss


class SequenceReplay:
    """Stores whole trajectories so context can be rebuilt at training time.

    A flat transition buffer cannot train a context model: sampling isolated
    (s, a, r, s') tuples throws away the history the attention needs. Sequences
    are sampled as contiguous chunks that never straddle a death, since a chunk
    spanning a reset teaches that the world discontinuously jumps — a fact about
    the harness, not the world.
    """

    def __init__(self, capacity, obs_dim, act_dim, seed=0):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.prev = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.delta = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dead = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.n, self.i = capacity, 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, prev_act, act, rew, nxt, dead):
        j = self.i
        self.obs[j], self.prev[j], self.act[j] = obs, prev_act, act
        self.rew[j], self.delta[j], self.dead[j] = rew, nxt - obs, float(dead)
        self.i = (self.i + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def sample(self, batch, length):
        starts = []
        tries = 0
        while len(starts) < batch and tries < batch * 40:
            tries += 1
            s = int(self.rng.integers(0, max(self.n - length - 1, 1)))
            if not self.dead[s: s + length - 1].any():
                starts.append(s)
        if not starts:
            return None
        idx = np.array([np.arange(s, s + length) for s in starts])
        t = lambda a: torch.as_tensor(a[idx])
        return (t(self.obs), t(self.prev), t(self.act), t(self.rew),
                t(self.delta), t(self.dead))
