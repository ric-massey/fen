"""
Multi-seed sweep over frame-stack depth.

Two questions at once:

  - Does temporal context lower the detection threshold? If frame stacking helps,
    recurrence in Stage 3 will help, and Stage 3 inherits a target. If it does
    not, something else is the bottleneck and it is cheaper to find out here.

  - Is any of this stable across seeds? Everything before this was n=1, which
    is not a result.

    python3 sweep.py --seeds 3 --stacks 1 2 4 8 --steps 200000
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def run_one(seed: int, stack: int, steps: int) -> dict:
    out = subprocess.run(
        [sys.executable, str(HERE / "train.py"), "--steps", str(steps),
         "--seed", str(seed), "--stack", str(stack), "--tag", "sweep", "--quiet"],
        capture_output=True, text=True, cwd=HERE,
    )
    if out.returncode != 0:
        print(out.stderr[-800:], file=sys.stderr)
        raise RuntimeError(f"seed={seed} stack={stack} failed")
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--stacks", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--steps", type=int, default=200_000)
    args = p.parse_args()

    results: list[dict] = []
    for stack in args.stacks:
        for seed in range(args.seeds):
            r = run_one(seed, stack, args.steps)
            results.append(r)
            print(f"  stack={stack} seed={seed}  "
                  f"step-onset {r['auc_step']:.3f}  ramped {r['auc_ramp']:.3f}", flush=True)

    (HERE / "runs" / "sweep.json").write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 70)
    print(f"{'stack':>6} {'step-onset':>18} {'ramped':>18} {'point (ramped)':>18}")
    print("-" * 70)
    for stack in args.stacks:
        rows = [r for r in results if r["stack"] == stack]
        def agg(key):
            v = np.array([r[key] for r in rows])
            return f"{v.mean():.3f} ± {v.std():.3f}"
        print(f"{stack:>6} {agg('auc_step'):>18} {agg('auc_ramp'):>18} "
              f"{agg('auc_point_ramp'):>18}")
    print("=" * 70)
    print("Read the ramped column. Step-onset overstates the capability because")
    print("a discontinuity is detectable without understanding the body at all.")


if __name__ == "__main__":
    main()
