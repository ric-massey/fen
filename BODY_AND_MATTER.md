# Body and matter

Design for the eleven internal deaths, and for a world made of stuff rather than
of numbers.

Status: design. Not built.

---

## The organising asymmetry

Everything here turns on one distinction:

**External threats are perceivable. Internal failures are not.**

You can see a predator. You cannot see a tumour. You feel hunger sharply and
early; you feel senescence not at all until it has already taken most of what it
is going to take. An organism has sense organs pointed outward and almost none
pointed inward, and the inward ones are indirect — you infer that something is
wrong from feeling wrong, without access to what.

So in Fen: **the organism can feel that it is declining, and cannot find out
why.** That is not a limitation of the implementation, it is the design. It is
also the most direct thing this project has produced on the question it started
from — a system with states it genuinely cannot introspect, arising from the
structure of being a body rather than from an experimenter withholding
information.

Perceivability is therefore a property of every mechanism below, declared
explicitly.

---

# Part 1 — The eleven deaths

| # | cause | perceivable? | mechanism |
|---|---|---|---|
| 1 | starvation | **yes**, early and sharply | energy reserve to zero |
| 2 | homeostatic collapse | **yes** | temperature outside survivable band |
| 3 | injury | **yes**, acutely | external damage exceeding integrity |
| 4 | senescence | **no** — only its consequences | wear accumulates; repair is < 100% efficient |
| 5 | allostatic load | **no** | regulating hard adds wear directly |
| 6 | organ failure | **partially** — you feel the deficit, not the cause | subsystems have separate condition |
| 7 | replication error | **no** | repair introduces copying errors that compound |
| 8 | autoimmune | **no** | repair mistargets and damages healthy tissue |
| 9 | infection | **late** — you feel ill before you know why | another organism consuming you from inside |
| 10 | inherited defect | **no** | a bad parameter you were born with |
| 11 | reproductive cost | **yes** — it is a choice | reproduction spends body mass |

## 4. Senescence — the one that matters most

**Without aging, a competent organism never dies.** In a safe patch with adequate
light it lives forever, has no reason to reproduce on any schedule, never vacates
its niche, and selection has nothing to act on. **The lineage stops evolving.**

A world with danger but no aging produces immortals wherever it happens to be
safe. Aging is the mechanism that forces generational turnover.

**Mechanism.** A `wear` variable that rises with metabolic throughput — living
costs, and the cost is cumulative. Repair reduces wear but at efficiency < 1, so
a residue always remains. Nothing here is a clock: an organism that lives hard
ages fast, one that idles ages slowly, and neither escapes.

**Not perceivable.** No interoceptive channel reports wear. Its *consequences*
are felt — capacities decline, everything costs more — so the organism
experiences getting worse without access to why.

## 5. Allostatic load

The cost of regulating too hard for too long. You do not die of the stressor, you
die of your response to it.

**Mechanism.** Sustained arousal and sustained regulatory effort add directly to
wear, over and above their energy cost. Fear already costs energy; this makes
chronic fear also *damaging*.

That turns the fear system into a genuine trade-off rather than a free benefit,
and it means a permanently vigilant strategy is lethal on a long timescale while
looking optimal on a short one.

## 6. Organ failure

**Mechanism.** Condition is not one number. Separate variables for locomotion,
sensing, metabolism, and repair, each accumulating damage unevenly. Below
threshold a subsystem loses function: a damaged sensor fogs, damaged locomotion
slows, damaged metabolism costs more, **damaged repair cannot fix any of the
others** — the failure mode that cascades.

**Partially perceivable.** You feel the deficit, not its source. A fogging sensor
feels like a confusing world, not like a broken sensor.

## 7. Replication error

Repair means rebuilding, rebuilding means copying, and copying is imperfect.

**Mechanism.** Each repair operation carries a small error probability. Errors
accumulate in somatic condition and compound — an error in the repair machinery
makes subsequent repairs worse. Past a threshold, tissue consumes resources
without contributing, which is what a tumour is metabolically.

This is the same mechanism as genetic mutation seen at a different scale. Do not
implement cancer and mutation separately; implement imperfect copying once.

## 8. Autoimmune

**Mechanism.** Repair has a targeting error rate: occasionally it damages healthy
tissue instead of repairing damaged tissue. The rate rises with repair *activity*,
so a heavily damaged organism repairing hard is more likely to hurt itself.

A second cascade: damage → more repair → more mistargeting → more damage.

## 9. Infection

**Mechanism.** A pathogen load that grows on the host's own resources, spreads on
contact between organisms, and is suppressed by an immune response that costs
energy and adds wear.

Ideally pathogens are *organisms in the ecology* rather than a variable — a type
whose energy source is other consumers. Then infection is predation at a
different scale and needs no separate machinery.

**Late perceivability.** Illness is felt well before its cause could be known.

## 11. Reproductive cost

Reproduction spends body mass and energy. Enough of it kills — which is real
(salmon, octopus) and should be **permitted rather than prevented**. Whether to
spend yourself on offspring is exactly the kind of decision selection should
settle.

---

# Part 2 — Matter

## Why energy-as-a-number is not enough

Currently energy is a scalar that appears from a timer. Even with a light influx,
one number cannot express:

- being short of *one specific thing* while surrounded by plenty of everything else
- waste from one organism being food for another
- bodies being made of anything
- novel substances existing at all

Real ecosystems are limited by particular elements — nitrogen, phosphorus — not
by "energy" in the abstract. A pond with abundant light and no phosphorus is a
dead pond.

## Six elements

Not the real ones. Naming them after real elements would imply claims about
chemistry that this does not make. Six letters, each with a role:

| | role | abundance |
|---|---|---|
| **A** | structural backbone — most of any body | very high |
| **B** | pairs with A, forms most bonds | very high |
| **C** | energy carrier — high-energy bonds | moderate |
| **D** | catalytic — needed in traces for repair and metabolism | low |
| **E** | limiting nutrient — required for growth and reproduction | scarce |
| **F** | inert diluent — participates in nothing, occupies space | high |

**E is the design's phosphorus.** Scarce, required to build a body, and therefore
the thing populations actually compete over once light is plentiful. F exists so
that "abundant" and "useful" are not the same word.

## Compounds

A compound is a **multiset of elements**, size 1 to 4. From six elements that is
209 possible compounds — combinatorially rich, computationally trivial, and open
enough that substances can exist which nobody designed.

Each compound has a **formation energy**. A reaction rearranges elements between
compounds and releases or consumes the difference. Elements are conserved
exactly; energy is conserved exactly.

## Metabolism as chemistry

- **Producers** use light to drive *uphill* reactions, building high-energy
  compounds from low-energy ones. Photosynthesis, structurally.
- **Consumers** run reactions *downhill*, capturing part of the released energy
  and excreting the products.
- **Decomposers** break complex compounds back toward simple ones, returning
  elements to the substrate.

**Metabolic pathways are heritable.** Which reactions an organism can catalyse is
in its genome. That is what makes "each type gets energy differently" a real
difference rather than a label — a lineage that can metabolise a compound nobody
else can has found a niche, and one that loses the pathway for its own food
starves surrounded by it.

## What this buys

**Waste closes the loop.** One organism's excretion is a compound with energy
still in it. Something that can metabolise it has a food source that exists only
because the first organism is there.

**Scarcity becomes specific.** You can starve for E in a world drenched in light.
Population limits stop being about energy in general.

**Bodies are made of something.** An organism is a quantity of compounds. Growth
acquires them, reproduction spends them, and a corpse returns exactly what it was
holding to the ground where it fell.

**Two conservation invariants to check every step** — elements and energy. If
either fails to balance, the model is wrong, and that is a correctness check
nothing else in this project currently has.

---

## Open questions

- **How many reactions should exist?** All thermodynamically valid ones is
  probably too many to search. Some restricted set, possibly itself evolvable.
- **Should pathways be discoverable within a life**, or only inherited? Within-life
  discovery is the more interesting answer and much harder.
- **Compound size cap.** Four keeps it tractable; larger allows more structure and
  a much bigger space.
- **Does F earn its place?** An inert diluent is realistic but may add nothing but
  bookkeeping.
- **Timescales again.** Producer growth against consumer demand against lifespan
  sets everything, and will need measuring rather than choosing.
