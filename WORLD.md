# Death, and the world

The two things still unresolved. They are one problem: death is where an organism
returns what it borrowed.

Status: design. Not built.

---

# Part 1 — Death

## What is wrong now

Death is an episode reset. The organism dies, the loop calls `reset()`, the
learner keeps its weights and carries on. That is not death, it is a
discontinuity in a training run — and it has caused every confusion in this
project so far, because two incompatible frames have been running at once:

- **the RL frame**, where death is a terminal state that supplies a negative
  reward and deaths are data
- **the life frame**, where death is the end and there is one of them

Sliding between these produced the nonsense of "it dies every 600 steps" for a
system that was really 400 organisms sharing one brain.

## Resolution

**Death is terminal for the individual. Always. Including in development.**

The development exemption applies to restarting *the whole simulation*, never to
restoring an individual inside a running one. There is no checkpoint-reload of an
organism, no rollback, no second attempt. If it dies, that one is over and the
world continues without it.

**Death is not a learning signal.** Nothing updates from the event. This was the
error that took longest to see: an organism cannot learn from dying, because it
is not there afterwards. What teaches is the *approach* — hunger, injury, fear,
the felt decline of health — all of which are survivable and all of which are
already built. Death instructs the lineage, never the individual.

**Death's only function is selection.** It decides whether you reproduced. That
reframes it correctly: death is not a punishment to avoid, it is a **deadline**.
An organism that reproduces and then dies has lost nothing. One that dies first
ends its line. The significance of death is entirely a matter of its timing
relative to reproduction, and nothing else.

**Dying is felt; death is not.** Health decline is interoceptive and degrades
self-knowledge as it worsens. The organism experiences deteriorating, badly and
with increasing confusion. It never experiences the end.

**The corpse returns its energy.** Whatever it was still carrying goes back into
the substrate where it fell. This is not decoration — it is what closes the loop
in Part 2, and it means dying is a contribution to the world rather than a
deletion from it.

## What this forbids

- restoring a dead organism, for any reason, at any time
- running two copies of the same organism
- treating a saved organism as anything but a record of what happened

Backups exist for studying a lineage after the fact. They are never resurrection.

---

# Part 2 — The world

## What is wrong now

**Energy comes from nowhere.** Food sites regrow on a timer, at a rate I chose.
That single fact makes almost everything downstream arbitrary:

- carrying capacity is a parameter I set rather than something the world implies
- population size cannot be a real result
- competition is not over anything genuinely scarce
- "different organisms get energy differently" has nothing to differ about

A world where food appears by fiat is a game level. To take a page from Earth,
the page is this: **there is a fixed energy influx, and everything else is
consequence.**

## Resolution: conserve energy

**Light.** A spatial field delivering energy into the world at a fixed rate per
unit area per step. This is the only source. It is the one number that sets the
scale of everything alive.

**Producers.** Static, rooted. They accumulate the light falling on them, up to a
cap, and hold it as biomass. Today's "food sites" become these, but stop being
magic: a producer in shade grows slowly, and one that is grazed flat takes real
time to rebuild from real input.

**Consumers.** Organisms. They take energy from producers, or from other
consumers, at a transfer efficiency well below 1. Nature runs about 10%; that is
probably too brutal here, but it must be *lossy*, because lossy transfer is what
makes trophic levels expensive and therefore meaningful.

**Return.** Corpses, and metabolic waste, deposit energy back into the substrate
where they are, which feeds producers there. Nothing is destroyed. Every joule
that entered as light eventually leaves as heat, having passed through however
many organisms it could.

**The invariant that makes this real:** total energy in the system should be
accountable at every step — influx minus dissipation, distributed across
producers, consumers, and substrate. If it does not balance, the model is wrong,
and that is a check no other part of this project currently has.

## What this buys

**Carrying capacity emerges.** How many organisms the world supports stops being
a parameter and becomes a consequence of light. Change the influx and the
population finds a new level on its own. That is a real result rather than a
setting.

**Competition becomes genuine.** Energy eaten by one is not available to another.
Two organisms in the same patch are actually in each other's way.

**Trophic structure gives niches for free.** "Each type gets energy differently"
becomes: producers, grazers, predators, scavengers. Predation needs no designed
predator — it is one consumer with a different diet.

**The world becomes modifiable in a meaningful way.** Not "move the food" but
change the light field — a bright patch and a dark one, a gradient, a day/night
cycle, a seasonal swing. Every one of those is a different selective regime, and
they are the same three numbers.

**And it gives failure a mechanism.** A lineage that over-grazes its producers
starves the next generation. That is not a rule anyone wrote; it falls out of
conservation.

## Specification, not hard-coding

The world should be a declared object — light field, terrain, producer
parameters, hazard behaviour, transfer efficiencies — so that "give it any world
and see what it does" is editing a specification rather than editing code. That
is a precondition for the world-distribution training the adaptation question
needs.

---

## Open questions, honestly

- **Timescales.** Producers must regrow slowly enough to be worth competing over
  and fast enough that lives are not all famine. That ratio sets everything and
  will need measuring, not guessing.
- **How lossy should transfer be?** Real ecosystems support few trophic levels
  because 10% compounds fast. Too generous and predation is free; too harsh and
  nothing above a grazer can exist.
- **Does light need to be spatially structured from the start?** A uniform field
  is simpler and may already produce competition. Structure can come later.
- **Death and reproduction ordering.** If reproduction costs energy, an organism
  can reproduce itself to death — which is real (semelparity) and should probably
  be allowed rather than prevented.
