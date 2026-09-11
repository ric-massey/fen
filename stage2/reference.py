"""
The canonical reference policy for frozen world v1.

One definition, imported everywhere. There were four near-identical copies of
this scattered across collect/export/watch/tests, and every time the observation
layout moved I patched some of them and missed others — which silently changed
what the "reference" meant between measurements.

This is deliberately simple and hand-written. Its job is not to be good; it is to
answer the only question that matters before training anything:

    is this world solvable at all?

If the reference cannot survive, a learned policy failing tells you nothing,
because you cannot distinguish "the agent did not learn" from "there is nothing
to learn". That check should run before every training pipeline, and for a long
time it did not.

Priority order is reflexive, not optimal: danger first, then thermal, then food.
An organism that finishes its meal while something is eating it does not have
descendants.
"""
from __future__ import annotations

import numpy as np

from env import OBS_LAYOUT

SMELL = OBS_LAYOUT["smell"]
GRAD = OBS_LAYOUT["thermal_gradient"]
HAZ = OBS_LAYOUT["hazard"]
AROUSAL = OBS_LAYOUT["arousal"]


def competent(obs: np.ndarray, energy: float, temp: float,
              flee: float = 0.85, tcap: float = 0.45, hungry: float = 0.9):
    """Hand-written survival: flee danger, regulate temperature, then eat."""
    a = np.zeros(6, dtype=np.float32)
    smell = obs[SMELL[0]:SMELL[0] + 2]
    grad = obs[GRAD[0]:GRAD[1]]
    hz = obs[HAZ[0]:HAZ[1]]

    if hz[2] > flee:
        a[0], a[1] = -1.2 * hz[0], -1.2 * hz[1]
    elif abs(temp) > tcap:
        a[0], a[1] = -np.sign(temp) * 1.2 * grad[0], -np.sign(temp) * 1.2 * grad[1]
    elif energy < hungry:
        a[0] = np.clip(smell[0] * 1.2, -1.2, 1.2)
        a[1] = np.clip(smell[1] * 1.2, -1.2, 1.2)
    return a


def reckless(obs, energy, temp, **kw):
    """Identical but ignores danger. Control for whether the hazard matters."""
    return competent(obs, energy, temp, flee=99.0, **kw)


def solvable(env_factory, trials: int = 5, cap: int = 30_000,
             threshold_cycles: float = 5.0) -> dict:
    """Gate: can the reference survive this world? Run before training anything."""
    lives, causes = [], []
    for k in range(trials):
        env = env_factory()
        o = env.reset()
        e, t, c = 1.0, 0.0, 0
        while True:
            o, r, d, info = env.step(competent(o, e, t))
            e, t, c = info["energy"], info["temp"], c + 1
            if info["dead"] or c > cap:
                break
        lives.append(c)
        causes.append(info.get("death_cause") or "survived")
    mean_cycles = float(np.mean(lives)) / 1000
    return {"lifespans": lives, "causes": causes,
            "mean_cycles": mean_cycles,
            "median_cycles": float(np.median(lives)) / 1000,
            "passes": mean_cycles >= threshold_cycles}
