# Notes — detecting silent failure

Working notes, not settled design. The question: Fen runs for months. How do you know it is
still learning, rather than having quietly stopped three weeks ago?

## Why this is hard here specifically

Standard ML answers do not apply:

- **No held-out test set.** One world, one life. Nothing is reserved.
- **No episodes.** Nothing to score, no natural evaluation boundary.
- **No task.** Drive satisfaction is the only outcome and it saturates by design.
- **Non-stationary by construction.** Fen changes its own world — eats resources, takes damage,
  wears paths. The distribution moves because of it, not despite it.
- **Probing perturbs.** In a one-life system you cannot freely intervene to test.

And the central ambiguity: **flat metrics are not evidence of failure.** A plateau might mean
it learned the world. Continued movement is not evidence of success either — it might be
drift, or chasing noise.

---

## The core discriminator

Track two families separately:

- **Performance** — prediction error, drive satisfaction, time-in-deficit
- **Structure** — code assignments, per-module weight change, memory turnover, state coverage

Then the diagnosis is the combination, not either alone:

| Performance | Structure | Coverage | Reading |
|---|---|---|---|
| flat | moving | any | Consolidating. Healthy. |
| flat | flat | saturated | Finished this world. **Enrich the world.** |
| flat | flat | low | **Stalled.** Something is broken. |
| degrading | moving fast | any | **Drift.** Suspect replay. |
| improving | moving | rising | Normal learning. |

This table is the actual answer to "is it still learning." No single metric gets there.

---

## Failure modes and how each is caught

**Codebook collapse** (stage 3) — everything maps to a handful of codes.
→ Entropy of the code-usage distribution; dead-code count. Alert on sustained decline.

**Codebook churn** — codes keep reassigning; the vocabulary never stabilises.
→ Assignment stability between time windows: what fraction of world-states map to the same
code as they did last week. Distinguish from collapse: churn keeps entropy high while
MI with ground truth stays flat.

**Replay drift** (stage 6) — the model trains on its own fabrications and detaches.
→ **The signature is divergence**: forward-model error on *held-out real* trajectories rising
while error on replayed trajectories falls. Neither number alone shows it. This one accumulates
silently and is the most dangerous, so it gets the tightest alerting.

**Policy collapse / behavioural narrowing** — one adequate strategy, exploration stops.
Looks exactly like competence.
→ Action-distribution entropy; state-visitation coverage; novelty rate (fraction of states
not seen in the last N days).

**Memory gone decorative** — written and retrieved, changes nothing.
→ Shadow ablation (below). If a memory-less copy performs identically, the store is dead weight.

**Arbiter mis-scaled** (stage 5) — deliberation never or always recruited.
→ Layer-usage histogram over time, and its correlation with novelty. Recruitment that does not
track novelty is a broken confidence estimate.

**Homeostatic lock-in** — finds a comfortable spot and sits there. Survives; develops nothing.
→ Spatial entropy, and **low drive variance is the tell.** A creature under no pressure is not
being challenged. Suspiciously easy homeostasis is a world problem, not a success.

**Dead gradients / silent NaN** — the boring one that costs the most time.
→ Per-module gradient norms and weight-change norms. Any module whose weights stop moving
while the system claims to be learning is the first thing to check.

**Forgetting in progress** — learning region B while losing region A.
→ Canary region (below).

---

## Shadow evaluation

The mechanism that resolves "probing perturbs."

**Do not probe Fen.** Periodically snapshot the *model weights*, load them into a separate
sandbox world, run diagnostics there, record results, destroy the sandbox. Fen's process is
never touched, paused, or branched.

Things only possible in shadow:

- Ablation studies — remove memory, freeze recurrent state, disable deliberation, and measure
  the specific deficit each is supposed to produce
- Held-out trajectory scoring for the drift check
- Probing what the codebook represents, without perturbing the live agent
- Recovery tests: does it still handle a situation it has not met in weeks

**The tension, stated honestly.** This sits close to the no-forking rule and the resolution
should be deliberate rather than assumed. Proposed rule:

> Shadow instances run with **drives disabled** and no continuity — a forward-model and policy
> test, not a life. Bounded to a few hundred steps, never resumed, destroyed immediately, never
> merged back into Fen.

Drives disabled is the load-bearing part. A short-lived copy computing predictions is apparatus.
A copy with needs that can go unmet is a second organism, and that is exactly what the one-life
rule exists to prevent.

If that distinction later seems too convenient, the fallback is passive metrics only, and
accepting the loss of ablation testing. Worth deciding before the first shadow run, not after.

---

## The canary region

The trick that gives a one-life system something like a held-out test set.

Reserve a small region of the world Fen visits rarely — reachable, unremarkable, not where
any drive is best satisfied. Do not intervene there. Simply log behaviour and prediction error
whenever it happens to pass through.

That gives:

- A **forgetting check** — competence in the canary region degrading over months while
  competence elsewhere holds is catastrophic forgetting, observed without intervening
- A **generalisation check** — behaviour in a rarely-visited place versus a well-worn one
- A natural experiment, at the cost of patience: you get a sample only when it wanders in

Sampling is irregular and sparse. That is the price of not intervening. Consider two or three
canary regions so the sample rate is workable.

---

## Passive dashboard

Computable continuously from logs. No intervention.

**Representation**
- code-usage entropy; dead-code count
- MI(code; ground-truth world structure) — ground truth is recorded anyway
- code→structure assignment stability, week over week

**Prediction**
- forward-model error, split self-caused vs world-caused (the stage-1 boundary, still holding?)
- rolling error on recent real trajectories
- error on replayed trajectories (paired with the above for the drift signature)

**Behaviour**
- action entropy
- state-visitation coverage and its entropy
- novelty rate
- layer usage: reflex / habit / deliberation proportions
- recruitment vs novelty correlation

**Homeostasis**
- per-drive time-in-deficit; drive variance
- near-death events per week
- offline-period frequency and duration

**Memory**
- episodic write rate, retrieval rate, retrieval→behaviour-change rate
- store size and decay throughput

**Machinery**
- per-module gradient norm, weight-change norm
- wall-clock per subsystem — a subsystem that stopped running is easy to miss

---

## Alerting

Trends, not thresholds. Almost every absolute threshold here would be invented.

Alert on:

- **Divergence between paired metrics** — replayed vs real error is the model case
- **Rate of change** crossing a band, rather than a level
- **Any module whose weight-change norm goes to zero** while others keep moving
- **Drive variance dropping toward zero** — lock-in, or a world gone too easy

Everything else is for the report, not the pager.

---

## Cadence

- **Continuous**: passive metrics into a time-series store
- **Daily**: automated report, mostly for the trend lines
- **Weekly**: human read of the report; this is where plateau-vs-stall is judged
- **Monthly**: shadow evaluation with the full ablation battery
- **Per stage**: the gate criteria from FEN.md, which are stricter than any of this

Keep the run-report discipline from Orrin: what actually happened, including the failures,
especially the failures. Architectural intent is not evidence.

---

## Open questions

- What is the right window for "structure is still moving"? Too short and consolidation looks
  like stall; too long and a real stall runs for weeks undetected.
- Can the plateau/stall table be computed automatically, or does it need a human read? Probably
  automated flagging, human judgement — but that means the flags need to be trustworthy.
- Does the canary region get contaminated once Fen learns it is being observed there? It cannot
  know it is observed, but it can learn the region is low-value and stop going. Then the sample
  rate collapses. Might need to make canary regions mildly attractive without being useful.
- How much does shadow evaluation cost in GPU time, and does it compete with the live process
  for the same hardware? If Fen must pause for its own diagnostics, that is a design failure.
- Is there a diagnostic for "the world is the bottleneck"? Distinguishing "Fen stopped learning"
  from "there is nothing left here to learn" is the whole enrichment decision, and coverage
  saturation is a proxy rather than an answer.
