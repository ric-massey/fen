# Inheritance — two channels

Design for the part of Fen that makes a lineage learn more than any organism can.

Status: design. Not built.

---

## The problem this solves

Lives are short. Measured, in the frozen v1 world: a competent hand-written
policy survives 5–14 hunger cycles. Nothing meaningful accumulates inside one
life, and every experiment so far has run into that ceiling.

A lineage is not short. If each organism starts from what the last one worked
out, the thing that learns stops being the individual and becomes the line.

That is the actual lesson from Earth, and it is not the genes. Two inheritance
channels exist in biology: genetic — slow, blind, across generations — and
cultural — fast, directed, transmitted by contact. Humans are unusual almost
entirely because of the second. Nobody rediscovers fire.

## The failure mode to avoid

**If offspring inherit the parent's weights, this is training with extra steps.**

A lineage of N organisms each living M steps is arithmetically identical to one
training run of N×M steps with periodic resets. Calling the resets "death"
adds ceremony, not science. Any version of this that copies a policy forward is
a training loop wearing a costume.

Three properties make it genuinely different, and all three are required:

- **Selection** — not everyone reproduces; the world decides which lines continue
- **Variation** — offspring differ, so the lineage explores rather than descending
  a single gradient
- **Lossy transmission** — what passes forward is degraded, partial, and can fail

The third is the one that is easy to skip and the one that matters most.

---

## Channel 1 — genetic

Inherited at birth, without interaction. Slow, blind, shaped only by selection.

**What is heritable:**

| | why this and not something else |
|---|---|
| body morphology | limb lengths, sensor ranges, metabolic rate. Physical, with unavoidable fitness consequences, and nothing about them says "gene for behaviour" |
| drive setpoints and weights | what feels good. This is the honest answer to *the designer built the value landscape* — selection sets it, not me |
| initial network weights | small, heavily mutated. A **prior**, not a policy: what you are born inclined toward, not what you know |
| plasticity | learning rate. How much a life can change you is itself heritable |

**Mechanism.** Not a designed mutation operator. Replication with imperfect
copying: the child is a copy of the parent's genome, and copying is physically
noisy. Heredity is what "copying mostly works" means; mutation is what "copying
is imperfect" means. Both are consequences of one thing rather than two
mechanisms bolted together.

**Note on the initial weights.** These must stay a weak prior. If a well-trained
policy is inherited wholesale, the cultural channel becomes decorative and this
collapses back into weight copying. Heavy mutation is not a nuisance here, it is
load-bearing.

---

## Channel 2 — cultural

Transmitted by observation during a juvenile period. Fast, directed, and lossy.

**What the child can actually see:**

- the parent's **position and movement** through the world
- **what happened** — the parent ate, was injured, fled
- the **state of the world** it shares with the parent

**What the child cannot see, ever:**

- the parent's torques — only the resulting motion
- the parent's **drives**: its hunger, its temperature, its damage
- the parent's **fear**
- any of the parent's reasons

This asymmetry is the entire design. The child observes a *trajectory* and has
to infer a *policy* — in its own body, under its own drives, with its own
different genes. It is learning "in situations that look like that, something
like this worked," which is a far weaker signal than a copied weight matrix and
exactly the signal a real animal gets from watching another animal.

**Mechanism.** During the juvenile period the child accumulates observed
(situation, apparent-action) pairs and trains on them as a weak prior, then
refines through its own experience. Transmission can simply fail: a juvenile
whose parent dies early, or which never sees the parent solve a particular
problem, gets nothing for it.

**Horizontal too.** Nothing restricts this to parents. An organism can observe
any conspecific in view. That allows knowledge to spread within a generation,
not only down it — which is how culture actually moves.

---

## Mutation — and why a parameter vector will not do

The failure mode is easy to build by accident. If the genome is a **fixed-length
vector of parameters**, then mutation is adding noise to numbers, and the space
of possible organisms was fixed the day it was written. You can tune what exists.
You can never gain a sensor, a metabolic pathway, or a limb. Every lineage
explores the same bounded box, converges, and stops — which is precisely the
stagnation that ended Tierra, Avida and Polyworld.

A genome has to be able to get **longer**.

### The operations

Copying is imperfect in more than one way, and the ways are not equivalent:

| | effect |
|---|---|
| **substitution** | a value changes. Tuning. |
| **deletion** | something is lost. Usually bad, occasionally simplifying. |
| **insertion** | novel material appears from nothing. Rare, mostly junk. |
| **duplication** | a segment is copied. **This is where novelty comes from.** |
| **translocation** | a segment moves. Changes context and regulation. |

**Duplication and divergence is the whole engine.** Copy a gene, and one copy
keeps doing the original job while the other is free to drift without cost. Almost
all genuine novelty in biology traces to this: the olfactory receptor family is
hundreds of genes descended by duplication from a handful, each drifted to detect
something different.

Without duplication, selection can only refine. With it, a lineage can acquire a
sensor it never had by copying one it did and letting the copy wander.

### What is structural and what is weights

Full neuroevolution — a genome that specifies network topology — is powerful and
notoriously finicky. The tractable split, which is also the biologically honest
one:

- **Structural, variable-length, duplicable:** which sensors exist and what each
  detects, which metabolic pathways can be catalysed, body morphology segments,
  which effectors exist. The organism's **interface to the world**.
- **Continuous, fixed-shape, inherited as a weak prior:** the network weights.
  The **brain**, which adapts within a life rather than across generations.

So structure evolves and behaviour learns. A lineage gains a new receptor by
duplication, and then has to learn what to do with it — which is a much more
interesting sequence than either alone, and it means a newly duplicated sensor is
initially useless and only becomes valuable through lifetime learning. That is a
Baldwin scenario arriving for free.

### Neutral mutation matters

Most mutations do nothing. That is not waste — it is **stored variation**. A
change that is neutral today can become useful after the world shifts, and a
design where every mutation has an immediate fitness effect throws that away.

Duplicated genes are neutral almost by definition at the moment of duplication:
the organism is unchanged, carrying a spare. Everything interesting happens later.

### Where mutations actually come from

Not "with probability p, perturb the genome." That is a parameter, and it is the
wrong shape. In a real organism mutation has physical causes, and they are not
interchangeable:

**Replication error.** The copying enzyme mismatches roughly 1 in 10^4-10^5
bases; proofreading and mismatch repair drag that to about 1 in 10^9. The rate is
not low because copying is careful — it is low because an enormous repair
apparatus sits behind it, and even that only reaches one-in-a-billion. Perfect
copying is not purchasable at any price: speed and fidelity trade against each
other.

**Living damages the genome.** This is the one that is easy to miss and the most
important here. DNA is a molecule in warm water and it degrades unassisted —
roughly 10,000 purines spontaneously lost per cell per day, hundreds of cytosines
deaminating. On top of that, the organism's own metabolism produces reactive
oxygen that attacks it. **Respiration is corrosive to the thing encoding you.**
Being alive costs genomic fidelity continuously, with no external cause at all.

**Repair is itself a source.** Fixing a double-strand break routinely loses or
adds material. The remedy mutates.

**Unequal crossing over.** When chromosomes pair and swap, misalignment leaves
one with two copies of a region and the other with none. **This is where gene
duplication comes from** — the engine of novelty above is a copying error during
recombination.

**Transposable elements.** Sequences that copy themselves around the genome;
about 45% of the human genome is this. Insertions and rearrangements from
parasitic DNA.

**External insult.** Radiation, chemical mutagens. The only category that is
about the environment rather than about being alive.

### Therefore the rate is not a setting

It should **emerge** from three things already in this design:

| source | consequence |
|---|---|
| metabolic throughput | living fast and hot damages the genome faster |
| repair investment | a heritable trait with a real energy cost, trading fidelity against everything else that energy could buy |
| time before reproducing | damage accumulates, so breeding late passes on more of it |

An organism that lives hard mutates more. That is true of real organisms —
metabolic rate correlates with mutation rate — and here it falls out rather than
being imposed.

### One process, three faces

The same accumulated genomic damage is **senescence**, is **cancer**, and is the
**raw material of evolution**. Whether it ages you, kills you, or varies your
descendants depends only on which cell it happened in and when.

Implement imperfect copying once, driven by metabolism and imperfectly repaired,
and deaths 4 and 7 in `BODY_AND_MATTER.md` and the entire genetic channel here
are the same mechanism seen from different angles. Nothing about that should be
three separate systems.

### The mutation rate is itself heritable

Real organisms have repair machinery whose fidelity is under selection, and the
trade-off is genuine: too high and errors accumulate faster than selection can
clear them (error catastrophe); too low and the lineage cannot track a changing
world. Fen should not set this number. It should be a gene, and the world should
settle it.

Expect it to co-vary with how fast the environment changes — which is a testable
prediction rather than an assumption.

### Germline and somatic are the same mechanism

Copying errors at reproduction are heritable: that is evolution.

Copying errors during repair within a life are not heritable: that is **ageing and
cancer** (`BODY_AND_MATTER.md`, deaths 4 and 7).

One mechanism, two consequences, distinguished only by which copy is being made.
Implement imperfect copying once and both fall out — and it means the same gene
that controls mutation rate controls both evolvability and lifespan, which is a
real and uncomfortable trade-off that organisms actually face.

## The juvenile period

Reproduction on sustained energy surplus. The child is born near the parent and
begins dependent:

- reduced metabolic cost early, so it survives long enough to watch
- proximity to the parent is where observation happens
- independence arrives on a timer, or when the parent dies

Parental provisioning is not generosity; it is what buys the transmission window.
An organism that reproduces and immediately abandons the child produces
offspring that inherit genes and nothing else — which is a *legitimate strategy*
and should be allowed to compete. Whether care is worth its cost should be
settled by selection, not by me.

---

## Types

Several kinds of organism, differing in **how they get energy**. Different diets
mean niches — they can coexist rather than purely compete, while still
interacting through shared space, shared danger, and getting in each other's way.

If one type eats another, predation arrives without a hand-designed predator:
the danger has its own goals and reacts to you.

---

## What this predicts, and how it is measured

Not "does it look alive." Four numbers, and the last one is the real result.

**1. Does the lineage improve?**
Naive performance of newborns, plotted across generations. If generation 60
starts better than generation 5 *before any learning of its own*, something
accumulated.

**2. Which channel carries it?** Ablate each:
- genes only (no juvenile period, no observation)
- teaching only (offspring genes randomised)
- both
- neither (fresh organism each generation)

If "both" does not beat the better of the two singles, the channels are not
composing and the design is wrong.

**3. Is transmission actually lossy?**
Compare the child's inferred policy against the parent's. It should be
*measurably worse* and *measurably different*. If a child ends up equivalent to
its parent, observation has become copying somewhere and needs finding.

**4. Genetic assimilation — the Baldwin effect.**
The one worth building this for. Take a behaviour that early generations must
learn, and check whether later generations are born already inclined toward it.
Measure naive newborns from early and late lineages on the same task.

Nobody implements assimilation. It either emerges from learning + heritable
variation + selection, or it does not. If it does, you have watched learned
behaviour become innate behaviour, which is the question underneath all of this.

---

## Honest limits

**Scale.** Populations of tens and generations in the hundreds, not the 10⁵ of
Avida — a neural network in a physics simulation costs orders of magnitude more
per organism-step than a self-replicating program. Too few generations for
evolution to invent much unaided.

The mitigation is structural rather than computational: **learning plus evolution
needs far fewer generations than evolution alone**, because learning does the
fine-tuning within each life and selection only has to shape what it starts from.
This combination is tractable at a scale where pure artificial life would not be.

Plus dispersal: simulate a small live population, archive the leavers with a
survival roll, let some return. A metapopulation with one visible patch, which
keeps genetic diversity without paying to simulate it.

**Stagnation is the default outcome.** Open-ended evolution is an unsolved
problem — Tierra, Avida, Polyworld and their successors mostly converge and stop
producing novelty. Expect that. Sustained drift is the surprising result, not the
assumption, and the measurements above should be able to tell the difference.

**The designed layer does not disappear.** Replication, imperfect copying, and a
world that kills are given. What was given must stay written down, because it is
the boundary of what any result here can claim.
