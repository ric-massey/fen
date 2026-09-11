"""
Danger and unpredictable change.

Two problems this fixes at once.

**Death was a countdown, not an event.** Neglect killed on a fixed schedule, so
every life was a race against a clock the organism could not stop. That is not
how anything dies. Starvation is slow and survivable for a long time; a predator
is not. Here neglect only *damages* — slowly, recoverably — and the fast way to
die is something dangerous. An organism that is regulating badly gets a long time
to work it out. An organism that walks into a hazard does not.

**The world was static within a life**, so memory bought nothing: whatever you
learned on your first circuit stayed true forever. Nothing here can be predicted
from the past — the thermal field drifts, food sites exhaust and reappear
elsewhere, and the hazard roams. A map learned ten minutes ago is wrong now, so
representing the present and updating it is worth something, which it was not
before.

The hazard is deliberately not a scripted predator with a pursuit rule. It drifts
with momentum and a weak attraction, so it is *statistically* dangerous without
being a solvable chase. Predictable danger gets memorised; unpredictable danger
has to be watched.
"""
from __future__ import annotations

import numpy as np


class Hazard:
    """A roaming region that damages badly on contact."""

    # The organism's top speed is ~0.0102 units/step (measured, not assumed).
    # The hazard is capped at roughly 1.1x that: fast enough to be genuinely
    # threatening, slow enough that fleeing works. Danger the organism cannot
    # escape is not danger, it is a timer.

    def __init__(self, seed: int = 0, arena: float = 2.6, radius: float = 0.55,
                 speed: float = 0.0038, attraction: float = 0.035,
                 damage: float = 0.030):
        self.rng = np.random.default_rng(seed)
        self.arena = arena
        self.radius = radius
        self.speed = speed
        # Weak. At 0.10 this became an effective pursuer that finds a
        # stationary organism every time, which contradicts the design above:
        # it should be a thing to keep an eye on, not a chase to solve.
        self.attraction = attraction
        # ~33 steps of contact is fatal from full health. Danger should be
        # an event you react to, not another slow drain.
        self.damage = damage
        self.reset()

    def reset(self) -> None:
        ang = self.rng.uniform(0, 2 * np.pi)
        self.pos = np.array([np.cos(ang), np.sin(ang)], dtype=np.float32) * self.arena * 0.8
        self.vel = self.rng.normal(0, self.speed, 2).astype(np.float32)

    def step(self, target: np.ndarray) -> float:
        """Advance the hazard. Returns health damage dealt this step."""
        # momentum + weak drift toward the organism + noise. Not a pursuit rule:
        # it should be a thing to keep an eye on, not a puzzle with a solution.
        to = target - self.pos
        d = np.linalg.norm(to) + 1e-6
        self.vel += self.attraction * (to / d) * self.speed
        self.vel += self.rng.normal(0, self.speed * 0.9, 2)
        sp = np.linalg.norm(self.vel)
        if sp > self.speed * 3:
            self.vel *= (self.speed * 3) / sp
        self.pos = self.pos + self.vel

        for i in (0, 1):                       # reflect off the arena bounds
            if abs(self.pos[i]) > self.arena:
                self.pos[i] = np.sign(self.pos[i]) * self.arena
                self.vel[i] *= -1.0

        return self.damage if d < self.radius else 0.0

    def sense(self, base: np.ndarray) -> np.ndarray:
        """Egocentric direction to the hazard and a proximity signal.

        Exteroception. Whether to flee still depends on how damaged it already
        is, which is internal — same split as the thermal gradient.
        """
        to = self.pos - base
        d = float(np.linalg.norm(to))
        if d < 1e-6:
            return np.zeros(3, dtype=np.float32)
        return np.array([to[0] / d, to[1] / d, float(np.exp(-d / 1.2))],
                        dtype=np.float32)


class Weather:
    """Slow, unpredictable drift of the thermal field within a single life.

    The axis is not fixed for the duration any more. What was the cool side an
    hour ago may not be now, so a learned spatial rule decays and the organism
    has to keep sensing.
    """

    def __init__(self, seed: int = 0, drift: float = 0.0015):
        self.rng = np.random.default_rng(seed)
        self.drift = drift
        self.reset()

    def reset(self) -> None:
        self.angle = float(self.rng.uniform(0, 2 * np.pi))
        self.rate = float(self.rng.normal(0, self.drift))

    def step(self) -> np.ndarray:
        # the drift rate itself wanders, so the field is not even predictably
        # rotating — it can stall, reverse, or swing
        self.rate += self.rng.normal(0, self.drift * 0.35)
        self.rate = float(np.clip(self.rate, -self.drift * 4, self.drift * 4))
        self.angle += self.rate
        return np.array([np.cos(self.angle), np.sin(self.angle)], dtype=np.float32)
