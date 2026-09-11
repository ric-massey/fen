# The environment

Design for a world made of materials with properties, rather than a plane with
food on it.

Status: design. Not built.

---

## The principle

**Give materials properties. Never give them uses.**

If the design says "rock can be used as shelter," then shelter is a feature I
built and an organism sheltering proves nothing. If rock simply *blocks heat and
blocks movement*, then hiding behind a rock to cool down is something an organism
has to discover — and discovering it is evidence about what the organism can do.

This is the same rule as everywhere else in Fen: build the pressure, not the
capability. Applied to the environment it means the world is a **physics of
materials**, and every affordance is emergent.

The test for whether a property is specified correctly: could you enumerate, in
advance, all the things it makes possible? If yes, it is too specific. Hardness
should be a number that determines what can scratch what — not a list of tools.

---

## Material properties

Every material carries the same small vector. Nothing is special-cased.

| property | what it determines |
|---|---|
| **hardness** | what can break what. The ordering matters more than the values |
| **density** | sinks or floats; how much effort to move |
| **thermal conductivity** | transmits or blocks heat |
| **thermal capacity** | how much heat it absorbs before changing temperature |
| **opacity** | blocks sensing — sight, smell, whatever the sensors are |
| **permeability** | blocks or permits movement through it |
| **friction** | grip, drag, how things slide |
| **cohesion** | holds together, or crumbles when disturbed |
| **fracture product** | what it becomes when broken, and at what size |

Nine numbers. A material is a point in that space, and there is no list of
material *types* — rock, sand and ice are just regions of it.

---

## Terrain

The world is a field of material, not a plane. Minimum useful structure:

- **Solid** — high cohesion, low permeability. Blocks movement, heat and sensing.
- **Loose** — low cohesion. Can be displaced, dug, piled.
- **Fluid** — flows, seeks level, high thermal capacity, carries things and
  drowns things.
- **Open** — nothing. Where organisms usually are.

Plus **elevation**, because it makes fluid flow somewhere, gives shelter a shape,
and makes some places visible from others.

---

## What should emerge, and must not be built

None of these are implemented. All of them are consequences of the properties
above, and every one is a measurable claim about whether an organism found it.

**Thermal shelter.** Get behind or under something with low thermal conductivity
and the thermal field stops reaching you. Falls out of conductivity and
permeability. Nobody writes "shelter."

**Concealment.** Opacity blocks sensing in both directions. Something opaque
between you and a hazard means it cannot sense you — and you cannot sense it.

**Water as refuge and as risk.** High thermal capacity makes fluid a thermal
buffer, which is a real strategy in heat. It also drowns things. Same material,
two consequences, neither one written down.

**Digging.** Low-cohesion material can be displaced. A displaced hollow is
shelter. That is two properties, not a burrowing behaviour.

**Carrying.** Density determines the cost of moving something. An organism that
can move a material at all can move it *somewhere useful*, which is the whole
basis of construction.

---

## Hardness gives you technology for free

This is the part worth building the rest for.

**You can break X with Y if Y is harder than X.**

One rule, and it produces a technology hierarchy nobody designed. Soft materials
are workable with anything. Harder ones need a tool. The hardest material in the
world can work everything and is worked by nothing.

That is the stone age, and then bronze, and then steel — not because anyone wrote
a tech tree but because **hardness is an ordering, and orderings have levels.**

It gets better with `fracture product`. Break something hard and you get
fragments — and the fragments inherit the hardness. A fragment of a hard material
is a thing that can now work materials the organism could not work before. That
is a tool, and it exists because breaking things produces sharp pieces.

So the sequence — *find hard thing → break it → use fragment to work a softer
thing you could not work before* — requires no design at all. It requires an
organism that can pick things up.

**What this predicts, and how you would know:** the hardness distribution of
materials an organism can affect should increase over its life, and over a
lineage. If later generations work materials earlier ones could not, that is
cumulative technology, and it is measurable without interpreting anything.

---

## Why this matters beyond being richer

Three reasons, in ascending order of importance.

**It gives behaviour somewhere to go.** Right now the entire action space is
"move, and hold your arm still." An organism with nothing to manipulate has a
ceiling on how interesting it can become that no amount of intelligence passes.

**It makes memory finally worth having.** Every attempt to show that context helps
has failed, because the world was reactive — gradients told you everything and
nothing needed remembering. A world of materials is not reactive. *Where the hard
rock was*, *which hollow was cool*, *what happened when I broke that* — none of
these are in the current observation and all of them persist. This is the first
version of the world where a transformer's context could carry something that
matters.

**It is what culture would be about.** Teaching, in the inheritance design, means
a juvenile watching an adult. Watching an adult forage teaches very little —
follow the gradient, which the juvenile can already do. Watching an adult *break
a rock and use the fragment* teaches something that would take a long time to
find alone. Cumulative culture needs something worth transmitting, and technique
is that thing.

---

## Open questions

- **Resolution.** A grid is simple and makes digging trivial to represent; a
  continuous field is more physical and much harder. Grid first, probably.
- **Does material need to be conserved?** Consistent with the energy conservation
  elsewhere, and it prevents infinite mining, but adds bookkeeping.
- **How is manipulation actuated?** The arm exists and currently does nothing. It
  should be the manipulator — which finally gives it a reason to exist beyond
  proprioception.
- **Should materials have a chemistry underneath**, connecting to `BODY_AND_MATTER.md`,
  or stay separate? Unified is more elegant and considerably more work.
- **How much should be visible?** If material properties are directly observable,
  discovery is trivial. They probably should not be: hardness should be learned
  by trying to break something, which makes the knowledge worth teaching.
