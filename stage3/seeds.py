"""
Run the Stage 3 representation gate across seeds, and put error bars on it.

A single run of train_repr.py reports `context gain on mean MI` as one number
and rules on it against a fixed 0.02 threshold. That is a point estimate from
one initialisation, and in the v1 run its magnitude (-0.020) was smaller than
the seed-to-seed scatter visible within its own table — no-attention beat
attention by 0.091 on temp while losing by 0.024 on health. A verdict of
"CONTEXT IS DECORATION" cannot be read off that.

This runs the same experiment N times, varying only the seed (model init and
batch sampling; the trajectory data is fixed, so this measures the stability of
the estimate rather than of the world), and reports the gain as mean, spread,
and a confidence interval. The verdict is then a three-way decision rather than
a coin flip:

    CI entirely above threshold  -> context earns its place
    CI entirely below threshold  -> context is decoration
    CI straddling it             -> undecided, and how many more seeds would help

    python3 seeds.py --seeds 5 --core transformer --steps 5000
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
RUN_RE = re.compile(r"written to (\S+)")


def one_seed(seed: int, core: str, steps: int, data: str, tag: str,
             codes: int, stride: int) -> Path:
    """Run train_repr.py once and return the directory it wrote."""
    cmd = [sys.executable, "train_repr.py", "--seed", str(seed), "--core", core,
           "--steps", str(steps), "--data", data, "--tag", tag,
           "--codes", str(codes), "--stride", str(stride), "--quiet"]
    p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout + p.stderr)
        raise SystemExit(f"seed {seed} failed")
    m = RUN_RE.search(p.stdout)
    if not m:
        sys.stderr.write(p.stdout)
        raise SystemExit(f"seed {seed}: could not find the run directory")
    return HERE / m.group(1)


def summarise(x: np.ndarray) -> dict:
    """Mean, sd, and a 95% CI on the mean. n is small, so use t not normal."""
    n = len(x)
    mean, sd = float(np.mean(x)), float(np.std(x, ddof=1)) if n > 1 else 0.0
    if n > 1:
        half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    else:
        half = float("nan")
    return {"mean": mean, "sd": sd, "ci_lo": mean - half, "ci_hi": mean + half,
            "n": n}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--core", type=str, default="transformer",
                   choices=["gru", "transformer"])
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--codes", type=int, default=128)
    p.add_argument("--data", type=str, default="data/traj.npz")
    p.add_argument("--tag", type=str, default="seeds")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--threshold", type=float, default=0.02,
                   help="the bar train_repr.py judges against")
    a = p.parse_args()

    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    print(f"=== representation gate, {a.core} core, {len(seeds)} seeds "
          f"({seeds[0]}..{seeds[-1]}), {a.steps} steps, stride {a.stride} ===",
          flush=True)

    runs, t0 = [], time.time()
    for i, s in enumerate(seeds, 1):
        ts = time.time()
        d = one_seed(s, a.core, a.steps, a.data, f"{a.tag}-{a.core}", a.codes,
                     a.stride)
        r = json.loads((d / "results.json").read_text())
        runs.append({"seed": s, "dir": str(d), "results": r})
        keys = [k for k in r if k != "untrained"]
        gain = float(np.mean([r[keys[0]]["mi"][f] - r[keys[1]]["mi"][f]
                              for f in r[keys[0]]["mi"]]))
        print(f"  seed {s}: mi_gain {gain:+.4f}   ({time.time()-ts:.0f}s, "
              f"{i}/{len(seeds)}, {time.time()-t0:.0f}s total)", flush=True)

    # Condition names come from the results themselves: the transformer core
    # calls them attention/no-attention, the gru core recurrent/no-recurrence.
    r0 = runs[0]["results"]
    with_ctx, without_ctx = [k for k in r0 if k != "untrained"]
    factors = list(r0[with_ctx]["mi"].keys())

    per_factor = {f: np.array([r["results"][with_ctx]["mi"][f]
                               - r["results"][without_ctx]["mi"][f]
                               for r in runs]) for f in factors}
    mi_gain = np.array([np.mean([per_factor[f][i] for f in factors])
                        for i in range(len(runs))])
    pred_gain = np.array([r["results"][without_ctx]["pred_loss"]
                          - r["results"][with_ctx]["pred_loss"] for r in runs])
    live = np.array([r["results"][with_ctx]["live_codes"] for r in runs])
    perp = np.array([r["results"][with_ctx]["perplexity"] for r in runs])

    g = summarise(mi_gain)
    print("\n" + "=" * 74)
    print(f"{with_ctx} vs {without_ctx}, per factor "
          f"(mean over {len(runs)} seeds)")
    print(f"{'factor':14} {'delta':>9} {'sd':>8} {'95% CI':>19} {'seeds +':>9}")
    for f in factors:
        s = summarise(per_factor[f])
        pos = int(np.sum(per_factor[f] > 0))
        print(f"{f:14} {s['mean']:+9.3f} {s['sd']:8.3f} "
              f"[{s['ci_lo']:+7.3f},{s['ci_hi']:+7.3f}] {pos:>4}/{len(runs)}")

    print(f"\ncodebook ({with_ctx}): live {live.mean():.1f}/{r0[with_ctx]['n_codes']}"
          f"  perplexity {perp.mean():.1f} ± {perp.std(ddof=1) if len(perp)>1 else 0:.1f}")
    print(f"context gain on prediction: {pred_gain.mean():+.4f} "
          f"± {summarise(pred_gain)['sd']:.4f}")
    print(f"context gain on mean MI:    {g['mean']:+.4f} ± {g['sd']:.4f}   "
          f"95% CI [{g['ci_lo']:+.4f}, {g['ci_hi']:+.4f}]   n={g['n']}")
    print(f"per-seed: {', '.join(f'{v:+.4f}' for v in mi_gain)}")

    if len(runs) > 1:
        t, pv = stats.ttest_1samp(mi_gain, 0.0)
        print(f"vs zero: t={t:+.2f}  p={pv:.3f}")

    thr = a.threshold
    if g["ci_lo"] > thr:
        verdict = "CONTEXT EARNS ITS PLACE"
    elif g["ci_hi"] < thr:
        verdict = "CONTEXT IS DECORATION"
    else:
        verdict = "UNDECIDED — the CI straddles the threshold"
    print(f"\nverdict (threshold {thr:+.3f}): {verdict}")

    if verdict.startswith("UNDECIDED") and g["sd"] > 0:
        # How many seeds to shrink the half-width below the distance to the
        # threshold, if the mean holds. Rough, normal-approximation guide.
        d = abs(g["mean"] - thr)
        if d > 1e-9:
            need = int(np.ceil((1.96 * g["sd"] / d) ** 2))
            print(f"  to separate {g['mean']:+.4f} from {thr:+.3f} at 95%: "
                  f"~{need} seeds (have {g['n']})")

    out = HERE / "runs" / f"{a.tag}-{a.core}-summary.json"
    out.write_text(json.dumps({
        "core": a.core, "steps": a.steps, "stride": a.stride,
        "seeds": seeds, "threshold": thr,
        "with_context": with_ctx, "without_context": without_ctx,
        "mi_gain": {"per_seed": mi_gain.tolist(), **g},
        "pred_gain": {"per_seed": pred_gain.tolist(), **summarise(pred_gain)},
        "per_factor": {f: {"per_seed": per_factor[f].tolist(),
                           **summarise(per_factor[f])} for f in factors},
        "verdict": verdict,
        "runs": [{"seed": r["seed"], "dir": r["dir"]} for r in runs],
    }, indent=2))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
