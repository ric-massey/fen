"""
Multi-seed sweep. Every Stage 2 number before this was n=1 on one world layout.

Reports mean and spread across seeds, and the median lifespan alongside the mean
so a few long runs cannot carry the average.

    python3 sweep.py --seeds 4 --steps 300000
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def run_one(seed: int, steps: int, extra: list[str]) -> dict:
    out = subprocess.run(
        [sys.executable, str(HERE / "train.py"), "--steps", str(steps),
         "--seed", str(seed), "--tag", "sweep", "--quiet", *extra],
        capture_output=True, text=True, cwd=HERE)
    if out.returncode != 0:
        print(out.stderr[-800:], file=sys.stderr)
        raise RuntimeError(f"seed {seed} failed")
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--extra", type=str, nargs="*", default=[])
    a = p.parse_args()

    rows = []
    for s in range(a.seeds):
        r = run_one(s, a.steps, a.extra)
        rows.append(r)
        print(f"  seed {s}: mean {r['mean_lifespan']:.0f}  "
              f"median {r['median_lifespan']:.0f}  "
              f"cycles {r['lifespan_cycles']:.1f}  deaths {r['deaths']}", flush=True)

    (HERE / "runs" / "sweep.json").write_text(json.dumps(rows, indent=2))
    for k in ("mean_lifespan", "median_lifespan", "lifespan_cycles"):
        v = np.array([r[k] for r in rows])
        print(f"{k:22} {v.mean():9.1f} +/- {v.std():.1f}")
    print("\ngate: >= 10 drive cycles")
