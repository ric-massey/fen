"""
Homeostatic drives and foggy interoception.

Two drives, deliberately in spatial conflict:

  energy       depletes with time and with effort; restored only at food sites,
               which sit mostly in the warm half of the arena
  temperature  drifts toward local ambient, which is warm at +x and cool at -x;
               the setpoint is neutral

So eating costs thermal comfort and staying comfortable costs energy. Neither
drive can be satisfied without spending the other. That conflict is the whole
design — it is what forces regulation rather than a single fixed behaviour.

There is no task reward anywhere. Reward is reduction in total drive deviation,
and every goal the agent appears to have has to come out of that.

TRUE state is recorded for measurement and never shown to the agent. What the
agent receives is `sense()` — the same state after noise, cross-channel bleed,
lag, and dropout. A creature with a clean readout of its own condition is not
modelling anything like a creature (FEN.md, design principle 4).
"""
from __future__ import annotations

from collections import deque

import numpy as np

# Attention-gating (a channel resolving only when attended, and attending
# perturbing it) is deliberately deferred: it needs an attend action, which
# belongs with Stage 5's control hierarchy. Noted rather than silently dropped.


class Drives:
    def __init__(
        self,
        seed: int = 0,
        dt: float = 0.025,
        energy_decay: float = 0.0003,     # per second at rest - a large reserve.
                                         # Starving should take a long time; danger
                                         # should not.
        effort_cost: float = 0.0003,      # per unit torque-second
        temp_rate: float = 0.35,          # how fast body temp tracks ambient
        fog: float = 1.0,                 # global fog scale; 0 disables
        lag_steps: int = 3,
        bleed: float = 0.18,
        dropout_p: float = 0.04,
        noise_scale: float = 0.10,
    ):
        self.rng = np.random.default_rng(seed)
        self.dt = dt
        self.energy_decay = energy_decay
        self.effort_cost = effort_cost
        self.temp_rate = temp_rate

        self.fog = fog
        self.lag_steps = lag_steps
        self.bleed = bleed
        self.dropout_p = dropout_p
        self.noise_scale = noise_scale

        # Setpoints. Energy is one-sided (more is fine); temperature is
        # two-sided (both directions are bad).
        self.energy_set = 1.0
        # Ambient saturates at +/-1.0, so a lethal threshold above that would
        # make temperature a comfort cost with no stakes. 0.92 is reachable:
        # roughly 290 steps of sitting in the far warm zone.
        self.temp_lethal = 0.92
        self.temp_set = 0.0
        self.alive_bonus = 1.0
        self.death_penalty = 5.0
        self.reflex_cost = 0.25

        self.n_drives = 3   # energy, temperature, health
        self.reset()

    # ── state ────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self.energy = 1.0
        self.temp = 0.0
        self._lag_buf: deque = deque(
            [np.zeros(self.n_drives)] * max(self.lag_steps, 1),
            maxlen=max(self.lag_steps, 1),
        )
        self._last_sense = np.zeros(self.n_drives)
        self._noise_level = self.noise_scale
        self.health_sense = 1.0
        self.alive = True

    def true_state(self) -> np.ndarray:
        return np.array([self.energy, self.temp], dtype=np.float32)

    # ── dynamics ─────────────────────────────────────────────────────────────

    def update(self, ambient: float, effort: float, ate: float) -> None:
        """Advance one control step.

        ambient : local ambient temperature in [-1, 1], from position
        effort  : sum |torque| this step
        ate     : energy gained this step
        """
        self.energy -= (self.energy_decay + self.effort_cost * effort) * self.dt * 40.0
        self.energy += ate
        self.energy = float(np.clip(self.energy, 0.0, 1.0))

        # Body temperature relaxes toward ambient — the agent must move to
        # regulate it, there is no internal thermostat.
        self.temp += self.temp_rate * (ambient - self.temp) * self.dt
        self.temp = float(np.clip(self.temp, -1.5, 1.5))

        # Energy and temperature no longer kill directly. Running out of
        # energy is not an event, it is a condition — it damages you, and the
        # damage is what you learn from. health.py owns death.

    # ── perceived deficit and reward ─────────────────────────────────────────

    @staticmethod
    def _compress(d: float, k: float = 0.25) -> float:
        """Weber-Fechner: fine discrimination near baseline, crude at extremes."""
        return float(np.log1p(max(d, 0.0) / k))

    def deficit(self, health: float = 1.0) -> float:
        """Total weighted drive deviation. Lower is better.

        Health is part of what is felt. Being damaged is aversive in itself,
        which is what makes neglect teach something without needing to be fatal.
        """
        e = self._compress(self.energy_set - self.energy)
        t = self._compress(abs(self.temp - self.temp_set))
        h = self._compress(1.0 - health)
        return e + 0.8 * t + 1.2 * h

    def reward(self, prev_deficit: float = 0.0, reflex_fired: bool = False,
               health: float = 1.0) -> float:
        """Negative deficit, offset so that being alive and well is positive.

        NOT the reduction in deficit. A difference-based reward telescopes:
        the undiscounted return over a whole life is just
        (deficit_start - deficit_end), which is independent of how long the
        life was. Survival is then worth exactly nothing and the agent has no
        reason to prefer living, which is precisely what the first two runs
        showed.

        The offset matters too. With a purely negative reward, dying is an
        escape — terminal states earn nothing, which beats accumulating
        negatives forever. `alive_bonus` has to exceed the deficit of a
        healthy state so that staying alive and regulated is strictly better
        than not existing.
        """
        r = self.alive_bonus - self.deficit(health)
        if reflex_fired:
            r -= self.reflex_cost
        if not self.alive:
            r -= self.death_penalty
        return float(r)

    # ── interoception ────────────────────────────────────────────────────────

    def sense(self) -> np.ndarray:
        """What the agent actually receives about its own condition.

        Human interoception is poorly localised, low resolution, confounded
        across channels, laggy, and intermittent. This applies all of that at
        the SENSING layer, so the unconscious machinery does not get a clean
        readout either — only the reporting layer being fogged would be a much
        weaker claim.
        """
        raw = np.array(
            [self.energy_set - self.energy, self.temp - self.temp_set,
             1.0 - self.health_sense],
            dtype=np.float32,
        )

        if self.fog <= 0.0:
            self._last_sense = raw
            return raw

        # Cross-channel bleed: two drives partially merge into an
        # undifferentiated pressure that cannot be decomposed back out.
        b = self.bleed * self.fog
        mixed = (1.0 - b) * raw + b * raw.mean()

        # Signal-proportional noise, with the noise level itself drifting, so
        # the agent cannot learn a fixed correction.
        self._noise_level += 0.05 * (self.noise_scale - self._noise_level)
        self._noise_level += 0.01 * self.rng.normal()
        self._noise_level = float(np.clip(self._noise_level, 0.02, 0.35))
        sigma = self._noise_level * self.fog
        noisy = mixed + self.rng.normal(size=self.n_drives) * (sigma * np.abs(mixed) + 0.02)

        # Lag: interoceptive signals arrive late.
        self._lag_buf.append(noisy)
        lagged = self._lag_buf[0] if self.lag_steps > 0 else noisy

        # Dropout: sometimes you simply cannot tell, and the last impression persists.
        if self.rng.random() < self.dropout_p * self.fog:
            out = self._last_sense
        else:
            out = lagged
            self._last_sense = out

        return np.asarray(out, dtype=np.float32)


class FoodSites:
    """Food that depletes on use and regrows on a cycle.

    Note the cycle is TWO phases: a dormant cooldown after depletion, then
    gradual regrowth. Total restore time is 2*regrow_steps, which is easy to
    forget when budgeting the world's energy supply against the agent's burn
    rate. Supply must comfortably exceed burn or nothing can survive however
    well it behaves.

    The regrowth cycle is the hidden structure of the world: it is learnable but
    not visible from any single observation. It is what memory will later be
    *for* — nothing in Stage 2 can exploit it, and that is fine.
    """

    def __init__(self, positions: np.ndarray, regrow_steps: int = 400,
                 reach: float = 0.55, bite: float = 0.055):
        # reach must exceed the arm's resting tip-to-base offset (~0.47 with
        # zero torque under gravity), or driving the base onto a food site
        # still fails to feed and eating is only ever reachable by accidental
        # flailing. Above that offset, posing the arm extends effective range —
        # so the arm earns its keep without being required for a first bite.
        self.pos = np.asarray(positions, dtype=np.float32)
        self.regrow_steps = regrow_steps
        self.reach = reach
        self.bite = bite
        self.reset()

    def reset(self) -> None:
        self.charge = np.ones(len(self.pos), dtype=np.float32)
        self.cooldown = np.zeros(len(self.pos), dtype=np.int32)

    def step(self, tip_xy: np.ndarray) -> float:
        """Returns energy gained this step."""
        self.cooldown = np.maximum(self.cooldown - 1, 0)
        regrown = (self.cooldown == 0) & (self.charge < 1.0)
        self.charge[regrown] = np.minimum(self.charge[regrown] + 1.0 / self.regrow_steps, 1.0)

        d = np.linalg.norm(self.pos[:, :2] - tip_xy[None, :], axis=1)
        near = (d < self.reach) & (self.charge > 0.05)
        if not near.any():
            return 0.0

        i = int(np.flatnonzero(near)[np.argmin(d[near])])
        gain = min(self.bite, float(self.charge[i]))
        self.charge[i] -= gain
        if self.charge[i] <= 0.05:
            self.cooldown[i] = self.regrow_steps
        return gain
