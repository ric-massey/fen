"""
Fear as a bodily state, not a signal.

The hazard was already in the observation vector and the trained policy ignored
it — deaths clustered near the hazard while the action barely changed with its
proximity. Adding more information about danger would not have fixed that,
because the problem was not information.

Fear in an animal is not a message saying "flee". It is a mode the whole body
enters, which changes how everything else works:

  fast, before recognition   the subcortical path fires before the cortex knows
                             what it saw — you flinch, then find out why
  metabolic mobilisation     costly, which is why it must be gated rather than
                             left on
  freezing before flight     stillness is the default response in most animals
  attentional narrowing      perception collapses onto the threat and DEGRADES
                             everything else
  persistence                arousal outlasts the stimulus; you stay jumpy
  strong encoding            one exposure is enough to be remembered

Three of those are implemented here, and the third is the interesting one.

`arousal` rises fast on threat proximity and on injury, decays slowly, and is
FELT — it joins the interoceptive channels, so the organism must read its own
fear the same way it reads hunger. Being afraid costs energy. And while afraid,
interoception of the slow drives degrades while the threat channel sharpens:
frightened organisms stop noticing they are hungry.

That last effect is the point. Fear is not extra information, it is a
reallocation — and it means self-knowledge is not constant but varies with
state, which is what this whole project is ultimately about.
"""
from __future__ import annotations

import numpy as np


class Fear:
    def __init__(
        self,
        dt: float = 0.025,
        rise: float = 5.0,          # how fast arousal climbs with threat proximity
        decay: float = 0.35,        # per second — slow, so it outlasts the threat
        injury_spike: float = 8.0,  # arousal per unit damage taken
        metabolic_cost: float = 1.6,   # energy burn multiplier at full arousal
        narrowing: float = 2.5,        # fog multiplier on slow drives when afraid
        threat_gain: float = 2.0,      # sharpening of the threat channel when afraid
    ):
        self.dt = dt
        self.rise = rise
        self.decay = decay
        self.injury_spike = injury_spike
        self.metabolic_cost = metabolic_cost
        self.narrowing = narrowing
        self.threat_gain = threat_gain
        self.reset()

    def reset(self) -> None:
        self.arousal = 0.0
        self.peak = 0.0

    def update(self, threat_proximity: float, injury: float = 0.0) -> None:
        """threat_proximity in [0,1] (1 = on top of you); injury = damage taken."""
        # Rise is driven by proximity directly rather than by contact, so the
        # response is anticipatory: it builds as the thing approaches, which is
        # what makes fear useful rather than merely a report of being hurt.
        drive = self.rise * max(threat_proximity, 0.0) ** 2
        self.arousal += (drive - self.decay * self.arousal) * self.dt
        if injury > 0.0:
            self.arousal += self.injury_spike * injury
        self.arousal = float(np.clip(self.arousal, 0.0, 1.0))
        self.peak = max(self.peak, self.arousal)

    # ── consequences ─────────────────────────────────────────────────────────

    def metabolism_multiplier(self) -> float:
        """Being afraid is expensive. Chronic fear starves you."""
        return 1.0 + (self.metabolic_cost - 1.0) * self.arousal

    def drive_fog_multiplier(self) -> float:
        """Attentional narrowing: hunger and temperature get harder to read."""
        return 1.0 + (self.narrowing - 1.0) * self.arousal

    def threat_sharpening(self) -> float:
        """...while the threat itself becomes clearer."""
        return 1.0 + (self.threat_gain - 1.0) * self.arousal

    @property
    def felt(self) -> float:
        """Fear is interoceptive. The organism has to read its own arousal."""
        return self.arousal
