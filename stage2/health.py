"""
Health: lasting consequence for neglect, so death does not have to be the teacher.

The mistake this fixes. Stage 2 was training by letting the agent die hundreds
of times and treating death as the learning signal. Nothing learns that way. You
do not get 700 attempts, and a terminal state instructs nobody — you are not
there afterwards to update on it.

What actually teaches an organism is the *approach* to death: hunger, cold,
sickness, injury. All of those are survivable, all of them are felt continuously,
and all of them are carried. Death is just what happens if you ignore them long
enough. So the aversive gradient should live in the states before death, and
death itself should be rare and almost incidental.

Health is that gradient. Sustained neglect damages it. Damage persists and heals
slowly, so a bad decision has a consequence the agent lives with rather than one
that wipes the episode. And damage is *felt*, not merely counted:

  - metabolism rises: being damaged costs more energy to sustain
  - movement slows: a damaged body is less capable
  - interoception fogs: being ill makes you worse at reading your own state

That last one matters most for what Fen is for. Sickness degrading self-knowledge
is a real feature of being an organism, and it means the agent's access to itself
is not constant — it varies with condition, and it is worst exactly when accurate
self-assessment would matter most.

Every mechanism here is optional (see the enable flags) so the contribution of
each can be ablated rather than assumed.
"""
from __future__ import annotations

import numpy as np


class Health:
    def __init__(
        self,
        dt: float = 0.025,
        # what counts as neglect
        energy_stress: float = 0.35,     # below this, energy deficit damages
        temp_stress: float = 0.82,       # beyond this, thermal load damages. High
                                         # on purpose: being warm should be unpleasant
                                         # (it costs reward via the deficit) long before
                                         # it is injurious. Only the extreme edge harms.
        damage_rate: float = 0.06,       # per unit stress per second - deliberately
                                         # slow: neglect should give you a long time
                                         # to work it out, not run a countdown
        heal_rate: float = 0.10,         # recovery per second when unstressed
        heal_energy_gate: float = 0.55,  # healing needs spare energy
        heal_cost: float = 0.25,         # healing burns energy
        # consequences of being damaged (each independently switchable)
        effect_metabolism: bool = True,
        effect_speed: bool = True,
        effect_fog: bool = True,
        metabolism_penalty: float = 1.8,  # burn multiplier at zero health
        speed_penalty: float = 0.55,      # speed multiplier at zero health
        fog_penalty: float = 2.2,         # fog multiplier at zero health
    ):
        self.dt = dt
        self.energy_stress = energy_stress
        self.temp_stress = temp_stress
        self.damage_rate = damage_rate
        self.heal_rate = heal_rate
        self.heal_energy_gate = heal_energy_gate
        self.heal_cost = heal_cost

        self.effect_metabolism = effect_metabolism
        self.effect_speed = effect_speed
        self.effect_fog = effect_fog
        self.metabolism_penalty = metabolism_penalty
        self.speed_penalty = speed_penalty
        self.fog_penalty = fog_penalty

        self.reset()

    def reset(self) -> None:
        self.health = 1.0
        self.damage_taken = 0.0
        self.healed = 0.0

    # ── dynamics ─────────────────────────────────────────────────────────────

    def stress(self, energy: float, temp: float) -> float:
        """How badly the organism is currently failing to look after itself."""
        e = max(0.0, self.energy_stress - energy) / max(self.energy_stress, 1e-6)
        t = max(0.0, abs(temp) - self.temp_stress) / max(1.0 - self.temp_stress, 1e-6)
        return float(e + t)

    def injure(self, amount: float) -> None:
        """Acute damage from danger. Fast, unlike neglect."""
        self.health = float(np.clip(self.health - amount, 0.0, 1.0))
        self.damage_taken += amount

    def update(self, energy: float, temp: float) -> float:
        """Advance health. Returns energy consumed by healing this step."""
        s = self.stress(energy, temp)
        spent = 0.0

        if s > 0.0:
            self.health -= self.damage_rate * s * self.dt
            self.damage_taken += self.damage_rate * s * self.dt
        elif energy > self.heal_energy_gate and self.health < 1.0:
            # Recovery is not free. An organism repairing itself is spending.
            gain = self.heal_rate * self.dt
            self.health += gain
            self.healed += gain
            spent = gain * self.heal_cost

        self.health = float(np.clip(self.health, 0.0, 1.0))
        return spent

    # ── consequences ─────────────────────────────────────────────────────────

    def metabolism_multiplier(self) -> float:
        if not self.effect_metabolism:
            return 1.0
        return 1.0 + (self.metabolism_penalty - 1.0) * (1.0 - self.health)

    def speed_multiplier(self) -> float:
        if not self.effect_speed:
            return 1.0
        return 1.0 - (1.0 - self.speed_penalty) * (1.0 - self.health)

    def fog_multiplier(self) -> float:
        if not self.effect_fog:
            return 1.0
        return 1.0 + (self.fog_penalty - 1.0) * (1.0 - self.health)

    @property
    def dead(self) -> bool:
        return self.health <= 0.0
