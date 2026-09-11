# Stage 1 — Body and the prediction loop

Status: **built, gate passed.** See FEN.md, Stage 1.

A 4-DOF arm in MuJoCo learns to predict the sensory consequences of its own
torques. The efference-copy residual — how wrong the prediction was — separates
self-caused sensory change from externally-caused change.

## Run it

```bash
python3 train.py --steps 300000 --tag dev
python3 controls.py --run runs/dev-<timestamp>
```

Training is ~30s per 60k steps on an M1 CPU. The model is small enough that CPU
beats MPS once transfer overhead is counted.

## Files

| File | What it is |
|---|---|
| `world.xml` | The body: 4 hinge joints, capsule links, gravity, damping |
| `env.py` | Step loop, observation, and unpredictable external perturbations |
| `babble.py` | Ornstein-Uhlenbeck motor babbling — correlated, not white, noise |
| `model.py` | Forward model, efference-copy residual, rolling buffer |
| `train.py` | Training loop and the gate metric |
| `controls.py` | The controls that decide whether the result is real |

## Result

At the training perturbation strength (12N), AUC separating self-caused from
world-caused sensory change:

- **trained model: 0.980**
- untrained model, same normalisation: 0.622
- raw magnitude of sensory change: 0.536

Self-caused residual 0.045, world-caused residual 0.443 — roughly a 10× gap.

The gate (AUC ≥ 0.9) passes. But the single number is not the finding.

## The real finding: a sensitivity threshold

AUC against perturbation strength, after 300k steps:

| force (N) | trained | untrained | \|delta\| |
|---:|---:|---:|---:|
| 1.0 | 0.735 | 0.491 | 0.499 |
| 2.0 | 0.856 | 0.496 | 0.500 |
| 4.0 | **0.934** | 0.512 | 0.503 |
| 6.0 | 0.959 | 0.536 | 0.509 |
| 8.0 | 0.970 | 0.564 | 0.516 |
| 12.0 | 0.980 | 0.622 | 0.536 |
| 20.0 | 0.984 | 0.727 | 0.581 |

**The boundary crosses 0.9 at roughly 3.5N.** That threshold is the honest
description of the self/world boundary: below it, the arm cannot tell that
something was done to it.

### …but a third of that was onset detection

Perturbations above have **instantaneous onset**. That means the model might be
detecting a sudden discontinuity rather than an external cause — a different and
much less interesting capability.

Control: the same forces applied through a Hann envelope (smooth onset *and*
offset, no discontinuity anywhere).

| force (N) | step onset | ramped onset | ramped, untrained | ramped, \|delta\| |
|---:|---:|---:|---:|---:|
| 4.0 | 0.934 | 0.782 | 0.516 | 0.501 |
| 8.0 | 0.964 | 0.844 | 0.532 | 0.505 |
| 12.0 | 0.974 | **0.872** | 0.556 | 0.510 |
| 20.0 | 0.982 | 0.902 | 0.609 | 0.525 |

**At 12N the gate passes on step perturbations (0.974) and fails on smooth ones
(0.872).** The ramped threshold is ~20N against ~3.5N for step onset — roughly
6× worse.

So the boundary is real: the model still beats untrained (0.556) and the
magnitude baseline (0.510) by a wide margin on ramps, and is clearly detecting
external force rather than only detecting edges. But a substantial part of the
headline number was onset detection, and the honest figure is the ramped one.

**Both numbers should be reported from here on.** Step-onset AUC alone overstates
the capability by about 0.1 and the sensitivity threshold by about 6×.

## Why the controls matter

Two ways this result could have been fake, both checked:

**The model might not be doing the work.** An untrained network with identical
normalisation scores 0.49–0.73. The trained model beats it by ~0.36 at the
training force. The network learned the boundary; it is not falling out of the
architecture.

**The residual might track something trivial.** Raw magnitude of sensory change
scores 0.50–0.58 — near chance throughout. So the residual is not a proxy for
"how much did things move." It is a prediction error.

Neither control is optional. Without them the headline AUC means nothing.

## An honest note on the training signal

Perturbed transitions are **not** filtered out of training. A real organism has
no label telling it which events were externally caused, so filtering would use
privileged information and inflate the result.

Instead the loss is Huber, which down-weights high-residual outliers without
needing a label — the plausible version of "don't over-learn from surprises."
The perturbation label is used only for evaluation, which is measurement rather
than training.

`Buffer.sample(clean_only=True)` exists as an upper-bound reference. It should
never be reported as the result.

## What did not happen

Training 5× longer (60k → 300k steps) barely moved anything: threshold 3.6N →
3.5N, self-residual 0.050 → 0.045. It converges fast and then plateaus.

Per `notes/diagnostics.md`, flat performance plus flat structure means the limit
is not training time — it is structural. The likely cause is that the forward
model sees a **single step** with no temporal context, so a small sustained
external force is indistinguishable from ordinary dynamics within one transition.

**Prediction for Stage 3 (FALSIFIED — see below):** adding recurrent state should
lower this threshold, because integrating evidence over time is exactly what
detects a small persistent deviation.

### Temporal context does not help. It hurts.

Sweep over frame-stack depth, 3 seeds each, 200k steps, held-out evaluation:

| stack | step-onset | ramped | ramped, point model |
|---:|---:|---:|---:|
| 1 | 0.975 ± 0.003 | 0.883 ± 0.001 | **0.918 ± 0.003** |
| 2 | 0.961 ± 0.003 | **0.896 ± 0.005** | 0.898 ± 0.007 |
| 4 | 0.959 ± 0.001 | 0.874 ± 0.006 | 0.873 ± 0.005 |
| 8 | 0.956 ± 0.007 | 0.865 ± 0.021 | 0.868 ± 0.015 |

Seed variance is small (±0.001–0.02), so these differences are real and the
earlier single-seed numbers were not flukes.

**Why more history makes detection worse.** With enough context the model can see
an ongoing perturbation in its own recent states and extrapolate the perturbed
trajectory — so it *predicts* the perturbation and stops being surprised by it.
History lets the model absorb exactly the events it is supposed to flag. The
effect is strongest for ramped bursts, which last 12 steps and are therefore
visible in an 8-frame history.

**This is a warning for Stage 3, not a detail.** A recurrent model integrating
evidence over time will also adapt to sustained external forces and stop
reporting them. There is a real tension between *integrate over time to detect
small deviations* and *do not adapt to the deviation you are trying to detect*.
Stage 3 must be tested for this specific pathology, and "recurrence improved
things" should not be assumed.

### Calibrated surprise costs discrimination

At `stack=1` the uncalibrated point residual (0.918) beats the NLL surprise
(0.883) on ramped detection. They converge from `stack=2` onward.

The likely mechanism is the mirror of the above: NLL learns to widen its variance
where dynamics are inherently hard — and then a perturbation occurring in one of
those regions is no longer surprising. **Calibrated surprise can explain away
perturbations by attributing them to inherent unpredictability.**

Both are kept and both are reported. NLL surprise is still what `detect.py` uses,
because a calibrated score is what makes a threshold meaningful, and a threshold
is what an organism actually needs. But on raw discrimination it currently costs
about 3.5 points.

### Where the gate stands

- **Point residual, stack=1, ramped: 0.918** — passes.
- **NLL surprise, stack=2, ramped: 0.896** — just below.
- Step-onset numbers (0.956–0.975) pass everywhere and should be disregarded.

The gate passes on the honest metric, with the least sophisticated version of
the model. That is worth sitting with before adding machinery.

## Known limitations

- No exteroception yet. The arm senses only its own joints, so "world" currently
  means only "force applied to me." Stage 2's world adds contact and affordances.
- Perturbations are forces on links. Other kinds of external event (a moved
  object, a change in the world's state) are untested.
- The body is fixed-base. It cannot locomote, which stage 2 may want.
- Single-step prediction only, which is the threshold limitation above.
