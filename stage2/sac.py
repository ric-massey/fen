"""
Soft Actor-Critic.

Chosen because Fen has no episodes (FEN.md, "There are no episodes"): SAC is
off-policy, handles continuous actions, and never needs a fresh on-policy
rollout or a clean reset. Entropy regularisation also gives exploration for
free, which matters when there is no task reward to shape behaviour.

One detail that is easy to get wrong here: death and timeout are different.
Death is a true terminal — there is no future value to bootstrap. Hitting the
step limit is a truncation of a life that would have continued, so its value
must still be bootstrapped. Conflating them teaches the agent that the world
ends arbitrarily.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def mlp(sizes, act=nn.SiLU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
        elif out_act is not None:
            layers.append(out_act())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_low, act_high, hidden=256):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, 2 * act_dim])
        self.act_dim = act_dim
        self.register_buffer("scale", torch.tensor((act_high - act_low) / 2.0))
        self.register_buffer("bias", torch.tensor((act_high + act_low) / 2.0))

    def forward(self, obs, deterministic=False, with_logprob=True):
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)

        u = mu if deterministic else dist.rsample()
        a = torch.tanh(u)

        logp = None
        if with_logprob:
            logp = dist.log_prob(u).sum(-1)
            # tanh change of variables
            logp -= (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)

        return a * self.scale + self.bias, logp


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, hidden, hidden, 1])
        self.q2 = mlp([obs_dim + act_dim, hidden, hidden, 1])

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class Replay:
    """Flat rolling buffer. No episode structure — one continuous life."""

    def __init__(self, capacity, obs_dim, act_dim, seed=0):
        self.o = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros((capacity, act_dim), dtype=np.float32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.o2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dead = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.n, self.i = capacity, 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, o, a, r, o2, dead):
        j = self.i
        self.o[j], self.a[j], self.r[j], self.o2[j], self.dead[j] = o, a, r, o2, float(dead)
        self.i = (self.i + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def sample(self, batch):
        idx = self.rng.integers(0, self.n, size=batch)
        return (torch.as_tensor(self.o[idx]), torch.as_tensor(self.a[idx]),
                torch.as_tensor(self.r[idx]), torch.as_tensor(self.o2[idx]),
                torch.as_tensor(self.dead[idx]))


class SAC:
    def __init__(self, obs_dim, act_dim, act_low, act_high, lr=3e-4, gamma=0.999,
                 tau=0.005, buffer=400_000, seed=0, target_entropy=None):
        torch.manual_seed(seed)
        self.actor = Actor(obs_dim, act_dim, act_low, act_high)
        self.critic = Critic(obs_dim, act_dim)
        self.critic_targ = Critic(obs_dim, act_dim)
        self.critic_targ.load_state_dict(self.critic.state_dict())
        for p in self.critic_targ.parameters():
            p.requires_grad_(False)

        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic entropy tuning; target is the usual -dim(A) heuristic.
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        # The usual -dim(A) heuristic comes from episodic benchmarks where
        # exploration is free. Here movement costs energy and energy is the
        # dominant cause of death, so a default target entropy pays the agent
        # far more to be random than the world charges it for the consequences.
        self.target_entropy = (-float(act_dim) if target_entropy is None
                               else float(target_entropy))

        self.gamma, self.tau = gamma, tau
        self.replay = Replay(buffer, obs_dim, act_dim, seed=seed)

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        a, _ = self.actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
                          deterministic=deterministic, with_logprob=False)
        return a.squeeze(0).numpy()

    def update(self, batch=256):
        if self.replay.n < batch * 4:
            return {}
        o, a, r, o2, dead = self.replay.sample(batch)

        with torch.no_grad():
            a2, logp2 = self.actor(o2)
            q1t, q2t = self.critic_targ(o2, a2)
            # `dead` is true termination only; timeouts still bootstrap.
            target = r + self.gamma * (1 - dead) * (torch.min(q1t, q2t) - self.alpha * logp2)

        q1, q2 = self.critic(o, a)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        for p in self.critic.parameters():
            p.requires_grad_(False)
        pi, logp = self.actor(o)
        q1p, q2p = self.critic(o, pi)
        pi_loss = (self.alpha * logp - torch.min(q1p, q2p)).mean()
        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_targ.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)

        return {"q_loss": float(q_loss), "pi_loss": float(pi_loss),
                "alpha": float(self.alpha), "entropy": float(-logp.mean())}
