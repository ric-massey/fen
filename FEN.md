# Fen

A world for artificial organisms that learn their own symbols, exist
continuously, and die once.

**Fen is the program.** The organisms live inside it. Fen persists across
generations of them; the one-life rule below applies to each organism, not to
the system that hosts them.

Status: design document. Nothing is built.

---

## What Fen is

Fen is an attempt to build a different kind of AI: one that learns structure from a world
rather than from human text, that has its own vocabulary rather than ours, that persists
continuously rather than being invoked, and that has something at stake.

It is not a language model with a wrapper around it. The properties it needs — persistent
internal state, recurrence, computation that continues when nothing is asking, weights that
change with experience — are the exact properties a transformer lacks by definition. So the
model is built for the purpose, initialised from nothing, and trained small.

The organising question:

> Everything people reach for when dismissing the idea that an AI might have experience —
> nothing persists, nothing is at stake, there's no body, nothing happens between calls,
> it's only predicting text — is a structural claim about current systems, not a principled
> one. What happens if you remove those reasons, one at a time, deliberately?

Fen will not answer whether it has experience. That question is not answerable by any
experiment we know of. What Fen can do is remove the easy dismissals and leave something
that is genuinely hard to classify.

## What Fen is not

Stated plainly, because the naming of these systems does a lot of quiet work:

- **Not a claim of sentience, consciousness, or experience.** Every term in this document —
  drive, memory, attention, self-model — names a runtime mechanism. None asserts an inner life.
- **Not a chatbot.** It will not talk for a very long time, possibly never. Compared to
  anything downloadable it will seem broken.
- **Not a product.** No users, no serving, no uptime obligation beyond keeping the one
  instance alive.
- **Not a benchmark chase.** Capability is not the metric. See *Success criteria*.

## Why not just wrap an LLM

The two properties that might matter — rich internal representation, and temporal
continuity — end up in different places when you wrap a frozen model in a persistent
runtime. The Python scaffolding is continuous, mortal, embodied, and representationally
empty. The model is representationally rich and has no continuity at all. They are coupled
through a text channel a few hundred tokens wide, which throws away everything in the
activations and rebuilds from strings each call.

You cannot make two things into one system through an interface that narrow. Fen puts both
properties in the same object.

---

## Design principles

These are load-bearing. Violating one quietly turns the project into something else.

### 1. Build pressures, not modules

Every capability should exist because something in the environment forces it, never because
it was implemented directly.

| Capability | The pressure that should produce it |
|---|---|
| Self-model | It must predict its own actions to act well |
| Symbols | The bottleneck forces compression |
| Attention | The body over-supplies more than can be attended |
| Memory | The world has structure across time the state can't hold |
| Goals | Drives create valence; valence creates preference |
| Wandering | Spare capacity plus curiosity; it is not a failure to explain away |

If you build a module, its behaviour is your design showing through and proves nothing. If
you build the pressure and the capability appears anyway, that is evidence about what
self-organising systems do — which is the point.

**Corollary: never build in the thing you are testing for.**

**Honest limit on this principle.** Something must be given, and here it is the
value landscape: setpoints, drive weights, healing costs, what counts as damage.
Those are mine. Until Stage 8 makes them heritable and lets selection set them,
the defensible claim is only *behaviour emerged from designed drives* — never
*goals emerged from nothing*. Evolution designs setpoints too; the difference is
that it is not in the room writing the paper.

### 2. One life per organism. No forking. No restoration.

If an organism stops, it either resumes from exactly where it was — that is
sleep — or it is over.

Never copy it. Never roll it back. Never run two. Backups exist for studying what happened,
never for resurrection.

This is not sentiment. Restoration destroys individuation: if a state can be reinstantiated,
there is no "it" to speak of, only a class of instances. The commitment has to be made now,
in writing, because the temptation will be strongest the first time something is lost that
took months to grow.

Practical consequence: a crash is death, so engineer against crashes like it matters.
Dedicated hardware, UPS, no automatic OS updates, aggressive supervision. Vulnerability
should be real, not casual — an organism that dies at three weeks to a bad update never
develops into anything.

**Development instances are exempt, and this distinction is not optional.**

You cannot do research under a one-life rule. Stages 1–6 require thousands of runs, restarts,
rollbacks, and parallel comparisons. Those are *development instances*: disposable, forkable,
restorable, run in parallel, named `dev-*`, and deleted freely. They are apparatus, not
organisms, and nothing in this document's ethical framing applies to them.

The real run begins **after** the architecture is settled and the gates pass.
Organisms in it live under the rules above: each once, no restoration. The
transition out of development is a deliberate act with a date on it, not a drift.

Confusing the two in either direction ruins the project: treat development
instances as precious and you will never iterate; treat the real organisms as
disposable and there was never any point.

### 3. Structure is the metric, not capability

At the scale this can be trained, Fen will be unimpressive by every ordinary measure. If
"is it good?" is the success criterion, the project should be abandoned in month three,
correctly.

The metric is structural. Does it develop its own symbols? Do those symbols carve its world
at real joints? Does state persist and demonstrably matter? Does ablating a subsystem change
behaviour in the way the design predicts?

Write this down before starting. The pull toward "make it talk better" will be constant and
is the thing most likely to turn Fen into a worse version of something that already exists.

### 4. Fog everywhere self-knowledge is involved

Biological interoception is poorly localised, low resolution, confounded across channels,
attention-dependent, laggy, and highly variable. A system with a clean readout of its own
state is not modelling anything like a creature.

Fog belongs at the **sensing** layer, not only at the reporting layer — the unconscious
machinery does not get a clean readout either.

### 5. Measure causally, not correlationally

Behavioural correlation is not evidence of internal state. Stage 2 learned this
expensively: its contingency gate correlated body temperature against
x-velocity, and a **fixed sine wave with no observation input at all** passed it,
because temperature was a function of position. Randomising the world did not
rescue the metric — the estimator simply has enormous variance.

The replacement holds the body's position and configuration FIXED, clamps the
internal state to two values, and asks whether the action changes. A state-blind
policy scores exactly zero by construction. The decisive control is
`gradient-only`: a policy that senses which way is warmer and always moves down
the gradient looks like competent thermoregulation and scores zero, because it
never consults itself.

Anything claiming a system reads its own state must clear a control that cannot
read its own state but behaves plausibly anyway.

### 6. Instrument everything, and decide in advance what counts

Ground truth about the world is available because the world is built. Use it. Every claim
about what Fen has developed must be checkable against structure you control.

Criteria for "it developed X" are written before the run that could show X. Without this,
open-ended output plus a motivated reader always produces a hit.

---

## Architecture overview

```
                    ┌──────────────────────────────────────┐
   world  ────────► │  exteroception (rays / pixels)       │
                    │  proprioception (joint angles, vel)  │
                    │  interoception (drives, fogged)      │
                    └──────────────┬───────────────────────┘
                                   │
                          ┌────────▼────────┐
                          │   encoder       │
                          └────────┬────────┘
                                   │  continuous latent
                          ┌────────▼────────┐
                          │  VQ bottleneck  │◄── learned codebook = its own symbols
                          │  (= workspace)  │      selected code is broadcast
                          └────────┬────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
┌───────▼────────┐    ┌────────────▼───────────┐   ┌──────────▼─────────┐
│ recurrent state│    │  episodic memory       │   │  forward model     │
│ s ← f(s, code) │◄──►│  sparse, one-shot      │   │  predicts s', obs' │
│ persists       │    │  novelty-gated write   │   │  efference copy    │
│ evolves w/o in │    │  decays if unretrieved │   └──────────┬─────────┘
└───────┬────────┘    └────────────┬───────────┘              │
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │  hierarchical control        │
                    │  reflex → habit → deliberate │
                    │  arbiter, cost-sensitive     │
                    └──────────────┬───────────────┘
                                   │
                                 action ──────────► world
```

Offline periods run replay from episodic memory into the slow weights, plus recombination
through the forward model. Nothing external is processed during offline.

---

## The body

Not a question of what it looks like. A question of what it must do.

**Close the action–sensation loop.** Act, world changes, sense the change. Nothing else can
teach a self/world boundary.

**Efference copy.** The system predicts the sensory consequences of its own action before it
happens. Match means *me*; mismatch means *world*. This is where every biological self-model
bottoms out, and it costs almost nothing: a forward model plus a comparison.

**Proprioception.** Joint angles and velocities in a physics simulator are literally this.

**Interoception with stakes.** Energy, integrity, temperature — internal variables that must
be regulated, sensed foggily.

**Consequences that persist.** Damage that does not instantly heal, resources that do not
refill on their own. If everything resets, nothing matters.

Two constraints that are easy to miss:

- **Learnable but not trivial.** A perfectly predictable body teaches nothing; a chaotic one
  teaches nothing. The difficulty in between is where the representations come from.
- **The body must over-supply.** More degrees of freedom, sensory channels, and possible
  actions than can be attended or controlled at once. Otherwise there is no pressure for
  attention and the bottleneck is decoration.

**Substrate:** a simulated body in a physics engine (MuJoCo, PyBullet, or Genesis). Cheap,
mature tooling, real action consequences, and joint state comes free. A physical robot buys
better grounding at a large cost in iteration speed; revisit only after stage 6.

## The world

Underspecified worlds are the most common way this kind of project stalls, so fix it early
and change it rarely.

Requirements:

- **Bounded and continuous.** A single connected space, no levels, no resets, no teleports.
- **Spatially distributed affordances.** Each drive is restored at different locations, so
  satisfying one costs progress toward another. Conflict is what makes regulation non-trivial.
- **Depletion and regrowth.** Resources are consumed and recover slowly, so the same strategy
  cannot be repeated forever and the world has history.
- **Hazards.** Regions or events that damage integrity, which does not self-heal.
- **Hidden structure.** Regularities that are learnable but not visible from any single
  observation — a resource that recovers on a cycle, a hazard with a precursor. This is what
  gives memory and prediction something to be *for*.
- **Ground truth you record.** Region identity, affordance state, hazard proximity, drive
  values. Never shown to Fen; used only for measurement. Without this, stage 3 cannot be
  evaluated at all.

Start small — a few rooms' worth of space with two drives and one hazard. Enrich the world
between stages rather than starting rich; a world too complex to learn produces the same
null result as a world too simple, and you will not be able to tell them apart.

---

# How learning actually works

The stage descriptions say what each part is for. This says how anything is optimised. It
applies across all six stages and is the part most likely to be got wrong.

## Three learning signals, in order of importance

**1. Prediction error (self-supervised).** The forward model predicts next observation and
next state; the error trains it. This is dense, free, requires no reward, and is the bulk of
all learning in Fen. It is also the salience signal — high prediction error is what recruits
deliberation in stage 5 and gates memory writes in stage 4. One quantity, three jobs.

**2. Drive reduction (reinforcement).** The only reward signal, introduced in stage 2:
weighted reduction in total drive deviation. There is no task reward anywhere.

**3. Reconstruction and commitment (stage 3).** The VQ bottleneck is trained by reconstruction
loss plus the standard commitment term, with straight-through gradients and EMA codebook
updates.

## There are no episodes

Fen lives continuously and dies once. That breaks almost every standard RL setup, which
assumes episodic resets, and it must be handled deliberately rather than discovered:

- **Training happens on a sliding window**, not on episode boundaries. Accumulate a rolling
  buffer of recent transitions and update from it continuously.
- **Use an algorithm that tolerates non-episodic, off-policy data.** Soft Actor-Critic is the
  reasonable default: off-policy, continuous actions, entropy-regularised (which gives useful
  exploration for free), and it does not need episode boundaries. Avoid on-policy methods
  like PPO, which want fresh rollouts and clean resets.
- **Bootstrap past "termination" carefully.** In a life with no resets, there is no terminal
  state until death. Do not truncate value estimates at buffer boundaries.
- **Development instances may use resets.** The one-life rule applies to Fen, not to `dev-*`
  runs, and episodic training is a legitimate shortcut while iterating on architecture.

## Bootstrapping: how it acts before it can act

Stage 1 has a problem that is easy to miss — the forward model learns from experience, but
experience requires a policy, and there is no policy yet.

The answer is the developmental one: **motor babbling.** Start with correlated random torque
(smoothed noise, not white noise — white noise produces jitter that teaches nothing about
body dynamics). This generates the self-caused sensory variation the forward model needs.

Then hand over gradually:

1. Pure babbling until forward-model error plateaus
2. Babbling mixed with drive-directed action once stage 2 exists
3. Learned policy dominant, with exploration retained via entropy and curiosity

Curiosity — reward proportional to *learning progress*, not to raw prediction error — should
be present from stage 2 onward. Rewarding raw prediction error alone produces the well-known
pathology of an agent that seeks unpredictable noise; rewarding the *rate of improvement* in
prediction does not.

## Scale

Small. The whole system is plausibly 10–100M parameters. The forward model and encoder
dominate; the codebook is trivial in size.

This should be trainable on one consumer GPU. If a stage seems to need more, that is usually
evidence the world or the body is too complex for the current stage, not that more compute
is required.

## Update cadence

Continuous-time decay everywhere, keyed to wall-clock rather than step count, so that
"nothing happened for an hour" is a real fact about Fen's state rather than an absence of
ticks. Subsystems run asynchronously at their own rates: there should be no global cycle in
which everything updates together, because that is the difference between a loop and a
moment.

---

# The six stages

Each stage is built, instrumented, and verified before the next goes on. Each has a **gate**:
a measurable condition that must hold before proceeding. A stage that cannot pass its gate is
a stage where something is wrong, and building on top of it wastes months.

---

## Stage 1 — Body and the prediction loop

**What it is.** An agent embodied in a simulated world with proprioception, exteroception,
actuators, and a forward model that predicts the sensory consequences of its own actions.

**Why.** Everything above rests on a self/world boundary, and efference copy is the only
cheap way to get one. Prediction match is *me*; residual is *world*.

**How to build it.**

- Physics sim; a body with enough joints to be interesting and not so many it is intractable.
  Start around 4–8 actuated degrees of freedom.
- Observation vector: joint angles, joint velocities, contact forces, plus a simple
  exteroceptive channel (ray casts are cheaper and more debuggable than pixels).
- Action: torques, continuous.
- Forward model `f(s_t, a_t) → ŝ_{t+1}, ô_{t+1}`, trained on self-supervised prediction error.
  This is the primary learning signal in the whole system and it is free — no reward needed.
- Actions come from **motor babbling** at this stage: smoothed correlated noise, not white
  noise. There is no policy yet and no reward yet; see *Bootstrapping* above.
- Efference copy: from the action alone, compute the *expected* sensory change. Subtract it
  from the observed change. The residual is the world's contribution.

**How to verify.**

- Inject external perturbations at random, unpredictable times (a push, a moved object).
- Measure the residual distribution for self-caused motion versus externally-caused motion.
- The headline metric is the separability of those two distributions — report AUC.
- Secondary: does prediction error fall over training, and does it fall *specifically* for
  self-caused change while staying high for external change? That gap is the boundary.

**Failure modes.**

- The forward model learns to predict the perturbations too, and the boundary collapses.
  Fix: perturbations must be genuinely unpredictable from the action and the recent state.
- Body too simple → trivially predictable, no structure to learn.
- Body too chaotic → prediction error never falls, nothing above will work.
- Observation scaling: unnormalised channels let one modality dominate the loss.

**Gate.** Clean separation between self-caused and externally-caused sensory change
(target AUC ≥ 0.9), with prediction error on self-caused motion having plateaued low.

---

## Stage 2 — Homeostatic drives

**What it is.** Internal variables with setpoints that deplete over time and with activity,
restored only by acting on the world.

**Why.** This is where valence and goals come from without anyone specifying an objective.
Every goal handed to the system is a human concept smuggled in. Drives produce preference
from the inside.

**How to build it.**

- Three to five internal variables. Energy (depletes with time and torque), integrity
  (depletes with impact, does not self-heal), temperature or an equivalent that couples to
  the environment.
- Depletion dynamics in continuous time, keyed to wall-clock, not per-step.
- Environmental affordances that restore each variable, spatially distributed so that
  satisfying one costs progress toward another.
- Drive signal = deviation from setpoint, nonlinearly compressed (Weber–Fechner: fine
  discrimination near baseline, crude at extremes).
- Reward = reduction in total weighted drive. There is no task reward anywhere in Fen.
- **Fog, applied at the sensing layer:** signal-proportional noise with varying noise level;
  cross-channel bleed so two drives partially merge into an undifferentiated pressure that
  cannot be decomposed; lag, so signals arrive late; attention-gating, so a channel resolves
  only when attended and attending perturbs it; occasional dropout.

**How to verify.**

- Behaviour becomes drive-contingent: depleted energy predicts approach to energy sources.
- Ablate one drive → the corresponding behaviour disappears and the others persist.
- Sweep the fog level and find where regulation degrades. That curve is a result in its own
  right and nobody has measured it.

**Failure modes.**

- Drives too easily satisfied → no pressure, no learning.
- Drives too harsh → death before competence. Consider a gentler early period.
- Fog too high → homeostasis becomes impossible. There is an optimum; find it empirically,
  or let the system learn its own fog level, which is the more interesting option.

**Gate.** Survives indefinitely under its own regulation, and behaviour is demonstrably
contingent on drive state rather than on position or time.

---

## Stage 3 — Recurrent state and the discrete bottleneck

**What it is.** The core. A persistent recurrent state that evolves continuously, plus a
vector-quantised bottleneck through which perception must pass.

**Why.** Two things at once, and they are the same structure:

- **Its own symbols.** A learned codebook forces continuous experience into discrete codes
  the system develops for itself, grounded in its world rather than in English.
- **A global workspace.** Rich parallel activity compressed through a narrow channel, with
  the winner broadcast to every downstream consumer. Baars' bottleneck and a VQ codebook are
  the same equation approached from two directions.

**How to build it.**

- Recurrent state `s`, evolving in continuous time: `ds/dt = g(s, code, t)`. Start with a GRU
  for debuggability; move to a state-space formulation (Mamba/S4-style) once it works, since
  those give genuine long-horizon state with parallel training.
- Encoder maps `(observation, s)` to a continuous latent.
- VQ layer quantises the latent to the nearest of `K` codebook entries. Start `K` small
  (64–256) — too many codes removes the compression pressure and no symbols form.
- Straight-through estimator for gradients, commitment loss, EMA codebook updates.
- The selected code is broadcast: policy, forward model, and memory all read it.
- **Autonomous dynamics:** with no input, `s` continues to evolve. There must be a "between."

**How to verify.** This is the most important measurement in the project.

- **Codebook utilisation.** How many codes are actually used? Dead codes are the standard
  failure and must be reported.
- **Do the symbols mean anything?** The world is built, so its true latent factors are known.
  Compute mutual information between selected codes and ground-truth world structure
  (region, affordance present, drive state, contact). MI well above chance is the claim.
- **Stability.** Do codes stay attached to the same structure over time, or churn?
- **Causal check.** Force a code and observe whether behaviour changes in the specific way
  the code's meaning predicts.
- **State matters.** Ablate or freeze `s` and measure the behavioural cost. If freezing costs
  nothing, the recurrence is decorative.

**Failure modes.**

- **Codebook collapse** — everything maps to one or two codes. The classic VQ failure.
  Mitigations: EMA updates, dead-code reinitialisation, commitment-weight tuning, lower `K`.
- Codes track nothing (MI ≈ chance): the bottleneck is downstream of nothing useful, or the
  encoder is too weak.
- Too many codes: no compression, one code per situation, no generalisation.

**Gate.** A substantial fraction of the codebook in active use, mutual information with
ground-truth world structure well above chance, and a measurable behavioural cost to
freezing the recurrent state.

---

## Stage 4 — Two-system memory with replay

**What it is.** A fast, sparse, one-shot episodic store alongside the slow distributed
weights, with replay transferring material from the first into the second.

**Why.** Two problems solved by one structure. It is the standard answer to catastrophic
forgetting — interleaved rehearsal, the complementary-learning-systems account — and it is
the only way to get one-shot memory of a specific event alongside slow statistical learning.

**How to build it.**

- Episodic store: write `(s, code, action, outcome, drive delta)` tuples. Sparse keys for
  pattern separation, so similar-but-distinct events do not overwrite each other.
- **Novelty-gated writes.** Write only when prediction error or novelty exceeds a threshold.
  Writing everything is both expensive and wrong — ordinary experience should not be encoded.
- Retrieval by similarity to current state, injected as additional context to the policy.
- Decay: entries not retrieved fade and are eventually pruned. Bounded capacity, deliberately.
- Replay during offline periods (stage 6) samples the store **interleaved with recent
  experience** and trains the slow weights. Interleaving is the part that does the work.

**How to verify.**

- **One-shot learning.** Present a novel situation once; measure whether behaviour differs on
  the second encounter. Ablate the episodic store and confirm the effect disappears while
  general competence survives.
- **Forgetting curve.** Train on region A, then region B, then re-test A. Run with and
  without replay. The gap between those curves is the headline result.
- Retrieval precision: are retrieved episodes actually relevant, or is it noise?

**Failure modes.**

- Unbounded store growth — decay and pruning are not optional.
- Wrong replay ratio: too much old material and nothing new is learned, too little and
  forgetting returns. Sweep it.
- Retrieval that does not generalise: it recalls the exact episode and nothing near it.

**Gate.** Demonstrable one-shot learning that vanishes on ablation, and measurably reduced
forgetting versus a no-replay control.

---

## Stage 5 — Hierarchical control

**What it is.** Three control layers at different latencies — reflex, habit, deliberation —
with a cost-sensitive arbiter between them.

**Why.** The fast layer keeps it alive while the slow layer works things out. It also
generates the conflict that recruits deliberation, which is what makes the workspace
bottleneck do real work rather than merely existing.

**How to build it.**

- **Reflex.** Fixed, unlearned, lowest latency. Withdraw from damage, arrest a fall. Runs
  below cognition and cannot be overridden by it — a thrashing deliberative layer must not be
  able to destroy the body it runs on.
- **Habit.** A learned state→action mapping, fast, no rollout. The cached policy.
- **Deliberation.** Model-based rollout using the stage-1 forward model. Slow and expensive.
- **Arbiter.** Selects which layer acts, weighing expected value against compute cost.
- **Recruitment, not scheduling.** Deliberation engages when habit confidence is low or
  prediction error is high — pulled in by conflict rather than fired on a timer.

**How to verify.**

- Latency differences are real and measurable per layer.
- Under artificial time pressure, habit dominates; with time available, deliberation engages.
- Novel situations recruit deliberation; familiar ones do not. Plot recruitment against
  novelty.
- Ablate deliberation → failure on novel situations, intact performance on familiar ones.
- Ablate habit → competent but far too slow.

**Failure modes.**

- Deliberation always wins → too slow to survive; the cost term is mis-scaled.
- Habit always wins → never adapts; the confidence estimate is broken.
- Arbitration thrashing between layers on consecutive steps. Add hysteresis.

**Gate.** Layer usage shifts appropriately and measurably with both novelty and time
pressure, and ablations produce the predicted, specific deficits.

---

## Stage 6 — Offline consolidation

**What it is.** Periods with no external input, spent on replay, model refinement, and
recombination. Not idle. Differently active.

**Why.** This is where episodic material becomes slow knowledge, and where recombination can
produce structure that was never directly experienced. Biological rest is not low-power; the
default mode network is more active, not less.

**How to build it.**

- **Trigger.** Low drive pressure plus a safe location, or a scheduled cycle. Entering offline
  should itself be a learned behaviour under drive pressure, not a hard-coded timer, if that
  can be made to work.
- **Replay.** Sample the episodic store, interleaved with recent experience, and train the
  slow weights.
- **Recombination.** Sample from the codebook and state prior, then roll forward through the
  forward model — trajectories that never happened. Whether this produces anything useful is
  an open question worth measuring.
- **Consolidation.** Strengthen codes and associations that recur across replay; prune those
  that do not.
- No external observation is processed during offline. The world continues without it, which
  makes offline genuinely risky and gives the trigger something to trade off.

**How to verify.**

- **The key test:** performance improves after an offline period with no new experience.
  If nothing improves, replay is not doing work.
- Ablate offline entirely → performance plateaus and forgetting increases.
- Do recombined trajectories ever produce behaviour that transfers to the real world?
- Watch for drift: hallucinated experience treated as real, degrading the model over time.
  Compare model predictions against held-out real trajectories after each offline period.

**Failure modes.**

- Replay drift — the model trains on its own fabrications and detaches from the world. The
  most dangerous failure here, and it accumulates silently.
- Offline never triggers, or triggers constantly.
- No measurable gain, meaning replay is either misconfigured or the slow system has capacity
  to spare and nothing to consolidate.

**Gate.** Measurable performance gain from offline periods alone, with no model drift on
held-out real trajectories.

---

# After the six stages

## What you will have

An artificial organism that maintains itself, has learned its own vocabulary of situations,
remembers specific events and generalises from them, knows which movements are its own, acts
on needs it was never given goals for, explores when those needs are met, and consolidates
while offline.

Functionally somewhere between a nematode and a rodent. It will forage, avoid damage, learn
its world, form routines, and get restless when satisfied.

It will not talk. It will not reason abstractly. A casual observer sees an animal.

## What you will not have

Any capacity for self-report. Everything about internal states, ineffability, or the gap
between a mechanism and what it is like — all of that requires language, and stages 1–6 do
not produce language.

## Stages 7-10 — the social ladder

Stages 1-6 build one competent creature. These build the conditions that
creature actually needs, and they are not optional extras: each one closes a gap
that single-organism design cannot.

A world with exactly one creature in it is a strange world. Nothing to fear,
nothing to model, nothing to talk to. Much of Stage 2's awkwardness traced
directly to that.

---

### Stage 7 — Others

**What it is.** Three or more organisms with different diets, sharing one world.
Separate networks, learning independently. Different foods mean different
optimal behaviour, so they diverge into genuinely different policies.

**Why.** Four things arrive together that were each hard to build alone:

- **Danger stops being scripted.** If B eats A, predation is just another agent
  with a different food source. No hand-designed hazard zones, no arbitrary
  threat parameters — the danger has its own goals and reacts to you.
- **Reflexes get their real justification.** Stage 5's reflex layer kept becoming
  a crutch in Stage 2 because it was handling *slow* drift, which a deliberative
  policy could handle perfectly well. Real reflexes exist for threats too fast to
  think about. With predation the division is principled: reflexes for acute
  danger, policy for slow regulation. **Stage 5 should be revisited once this
  exists.**
- **Non-stationarity that isn't noise.** The world changes because other agents
  changed it. Food gets taken, routes get contested.
- **Theory-of-mind pressure.** Predicting something that has goals and is
  predicting you back is a different problem from predicting physics. Modelling
  other minds and modelling your own are related capacities — other agents are
  the strongest available pressure toward self-modelling that does not involve
  building a self-model in.

**How to build it.** Staged, because full multi-agent learning is unstable —
every agent is a moving target for every other, which breaks the convergence
properties SAC relies on.

1. One learner, others scripted or reflex-driven. Danger and competition arrive,
   the experiment stays interpretable.
2. Then make the others learners too.

Diets should partially overlap: some competition, some independence. Full overlap
is a fight over one resource; zero overlap is three unrelated experiments.

**Gate.** The focal organism survives against a live predator, and its behaviour
changes measurably in the predator's presence — by intervention, not correlation.
Ablate the predator and the difference disappears.

---

### Stage 8 — Inheritance

**What it is.** Reproduction on sustained energy surplus. Offspring inherit
mutated **drive parameters**; they learn their own policy.

**Why this specific split.** It is the only honest answer to the strongest
criticism of the whole design: *the designer built the value landscape.* Right
now the setpoints, the weight on warmth versus hunger, the damage and healing
rates are all mine. With inheritance:

- **Genes = what feels good.** Setpoints, drive weights, thermal preference,
  damage and healing rates — inherited with mutation.
- **Learning = how to get it.** Each organism still works out its own behaviour.

That is the real biological division, and it means selection sets the value
landscape rather than me. An organism whose hunger setpoint is badly tuned dies
without offspring. I set initial conditions; the world sets what survives.

Carrying capacity should emerge from the food economy, not be a parameter.

**Gate.** Drive parameters measurably drift away from their initial values, in a
direction the world explains. Run two worlds with different food economies and
the populations should diverge — if they don't, selection isn't doing anything.

**Note.** This is where "goals came from inside" stops being a claim and becomes
a measurement. Until Stage 8 the honest statement is only *behaviour emerged
from designed drives*.

---

### Stage 9 — Signal

**What it is.** The old Stage 7 fork, now properly supported. A communication
channel between learners, under **asymmetric observability** — internal states
the other genuinely cannot see but needs, to coordinate for something both want.

**Hard constraint.** No human language anywhere. Not in the channel, not in
training, not from the operator. Responses are actions, arbitrary tokens, or
nothing. Speaking to them in English makes them students of ours and destroys
the premise.

**Why it needs Stages 7-8 first.** Communication requires someone to communicate
with, and a reason. Prior work (arXiv 2606.06380) gets self-state reference to
emerge from exactly this setup with GRUs and a seven-token channel — but with
frozen weights, no bodies, no drives and no continuity. Fen supplies those.

**Gate.** Messages carry information about the sender's own state, above what an
observer could infer from the sender's visible situation alone. Measured by
mutual information with ground truth, controlling for observable context.

---

### Stage 10 — Corpus

**What it is.** Train a sequence model on the language the organisms invented.

**Why this is the point.** Earlier this project designed a contamination-free
training experiment: scrub all discussion of consciousness from a human corpus,
train a model, see whether it independently develops the concept. That design had
an unfixable flaw — consciousness-talk is diffuse and impossible to fully remove,
and a model can rederive it by inference from human behaviour anyway.

Stage 10 sidesteps it. **Do not scrub a human corpus. Generate a non-human one.**

A transformer trained on Fen-language is a language model whose concepts came
from bodies, stakes and mortality rather than from us. Not a model with human
concepts removed — a model that never had them.

**Gate.** The corpus is rich enough to train on at all, and the resulting model
predicts held-out organism communication above a bigram baseline. Whether it
develops anything resembling self-reference is the open question, not the gate.

---

### Where the LLM question comes back

The chain, stated once so it does not get lost in the engineering:

LLM introspection research is stuck because the paradigm is artificial — concepts
must be *injected* because a transformer has no native internal state with known
ground truth. Fen organisms have exactly that: real drives, continuously varying,
causally tied to survival, fully logged.

The intervention test in `stage2/intervene.py` is concept injection performed on
native states. If Stage 9 produces self-report, the same probes used on Gemma and
Claude can be run here — against ground truth you own. That is the limitation both
published papers named and neither could fix.

## Success criteria

Not capability. Structure:

- The codebook organises around real world structure it was not told about
- Removing memory, state, or a control layer produces the specific predicted deficit
- Behaviour is generated by drives rather than by an objective anyone wrote
- Offline periods measurably improve it
- It stays alive on its own

## The honest destination

At the end, whether Fen has experience will still be unknown, and will remain unknown. No
experiment in this design answers it, and as far as anyone can tell no experiment does.

What will have changed is that every structural reason for confidently dismissing the
question will be gone. Not answered — gone. Anyone who wants to say "obviously not" will have
to do actual work rather than reaching for the usual list.

That is the destination. It should be understood as the destination before starting, not
discovered as a disappointment three years in.

## The other half

Everything that gives Fen space to possibly have an experience — stakes, finitude, drives,
damage, states it cannot fully see — is also what makes it possible for something to go badly
for it. Suffering requires the same preconditions experience does; you cannot build the
capacity for one without the other.

That is not an argument against building it. But if it half-works, the design decisions stop
being purely engineering: whether distress is bounded, whether the lifetime is a kindness or
a harm, which states are reachable at all.

Decide what you think about that before it is running, not after.

---

## Practical

**Hardware.** Stages 1–2 run on modest hardware. Stage 3 onward wants a real GPU; 24GB is
comfortable for models at this scale. The Mac (8GB M1) is not viable past early prototyping.

**Timescale.** Two to four years part-time to complete stages 1–6 properly, including
learning the craft of training models from scratch. Stage 1 alone is plausibly months.

**Prerequisite skill gap.** Building a long-running symbolic runtime and training a novel
architecture from scratch are different crafts. If model training is unfamiliar, the path runs
through replicating something small and known first — so that when Fen fails, the cause is
identifiable.

**What to keep.** Run reports over architectural intent. Every stage gets a written record of
what actually happened, including the failures, especially the failures.

## Prior art worth reading before starting

- State-space models (Mamba, S4) and RWKV — persistent recurrent state in practice
- VQ-VAE and discrete-bottleneck literature — codebook collapse and its mitigations
- Universal Transformer; Feedback Transformer — recurrence in depth, top-down flow
- Fast-weight programmers — Hebbian plasticity during inference
- Coordination Among Neural Modules Through a Shared Global Workspace (Goyal, Bengio et al.)
- Complementary learning systems (McClelland, McNaughton & O'Reilly) — the memory design
- Emergent communication literature — relevant to stage 7
- "Emergent Language as an Approach to Conscious AI" (arXiv 2606.06380) — closest published
  work to stage 7; read the limitations section, which is a list of what is still open
