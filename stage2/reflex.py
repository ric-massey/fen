"""
Innate reflexes — the lowest layer of hierarchical control, moved forward from
Stage 5.

Why here and not there. Stage 2 was asking SAC to discover survival from tabula
rasa: it died every ~600 steps, so its replay buffer contained almost nothing
but short lives ending in death, and it had no examples of surviving to learn
from. A scripted policy clears 30,000 steps easily, so the task was never the
problem — the bootstrapping was.

No animal learns to withdraw from damage by dying a few hundred times. That
behaviour is innate, and learning refines it afterwards. FEN.md already says
reflexes are "fixed, unlearned, lowest latency... runs below cognition"; it just
put them at Stage 5, which turns out to be the wrong order. Reflexes are what
keep an organism alive *while* the slow layer works things out, so they have to
exist before there is a slow layer worth training.

Two design commitments:

  Reflexes read TRUE interoceptive state, not the fogged signal. Spinal reflexes
  use raw afferents and bypass conscious perception entirely; the fog in
  drives.py models interoceptive *awareness*, which reflexes are meant to be
  below. This is the same split as Orrin's two-readers design — the fast path
  reads raw values, the deliberative path reads a rendered estimate.

  Reflexes are AVERSIVE. Firing one costs reward (drives.reflex_cost). A free
  safety net gets exploited: with costless reflexes the policy learned to park
  permanently inside the emergency envelope, running its drives down to the
  triggers and letting the innate layer do 38% of its acting. Pain and panic
  are unpleasant precisely so an organism avoids needing them; a reflex that
  costs nothing prevents the learning it exists to enable.

  Reflexes fire only in emergencies. Triggers sit well outside normal operating
  range (thermal at 0.72 against a lethal 0.92; energy at 0.12 of 1.0), so the
  learned policy owns most of the state space. A reflex that fires constantly
  is not a reflex, it is the policy, and there would be nothing left to learn.
"""
from __future__ import annotations

import numpy as np


class Reflex:
    def __init__(self, temp_trigger: float = 0.72, energy_trigger: float = 0.12,
                 base_speed: float = 1.2):
        self.temp_trigger = temp_trigger
        self.energy_trigger = energy_trigger
        self.base_speed = base_speed
        self.fired_temp = 0
        self.fired_energy = 0

    def reset_counts(self) -> None:
        self.fired_temp = 0
        self.fired_energy = 0

    def __call__(self, action: np.ndarray, energy: float, temp: float,
                 smell: np.ndarray) -> tuple[np.ndarray, str | None]:
        """Override the policy's base motion in an emergency.

        Returns (action, which_reflex_fired). Thermal takes priority: at the
        extremes it kills faster than starvation, and there is no point walking
        to food while cooking.

        Arm torque is zeroed whenever a reflex fires. Stopping non-essential
        effort under threat is itself a survival response, and effort is the
        dominant term in the energy budget.
        """
        a = action.copy()

        if abs(temp) > self.temp_trigger:
            a[0] = -np.sign(temp) * self.base_speed
            a[1] = 0.0
            a[2:] = 0.0
            self.fired_temp += 1
            return a, "temp"

        if energy < self.energy_trigger and np.linalg.norm(smell[:2]) > 1e-6:
            a[0] = np.clip(smell[0] * self.base_speed, -self.base_speed, self.base_speed)
            a[1] = np.clip(smell[1] * self.base_speed, -self.base_speed, self.base_speed)
            a[2:] = 0.0
            self.fired_energy += 1
            return a, "energy"

        return a, None
