"""
Motor babbling: smoothed, temporally correlated torque.

White noise produces jitter that averages out and teaches almost nothing about
body dynamics. An Ornstein-Uhlenbeck process produces sustained pushes that
actually move the arm through its configuration space, which is what the forward
model needs to see (FEN.md, Bootstrapping).
"""
from __future__ import annotations

import numpy as np


class Babbler:
    def __init__(self, dim: int, seed: int = 0, theta: float = 0.12, sigma: float = 0.55,
                 scale: float = 2.0):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.theta = theta      # mean reversion — lower means longer correlation
        self.sigma = sigma      # noise magnitude per step
        self.scale = scale      # output torque scale
        self.x = np.zeros(dim)

    def __call__(self) -> np.ndarray:
        self.x += -self.theta * self.x + self.sigma * self.rng.normal(size=self.dim)
        return np.tanh(self.x) * self.scale

    def autocorrelation_halflife(self) -> float:
        """Steps for a deviation to decay by half — sanity check on smoothness."""
        return float(np.log(2) / self.theta)
