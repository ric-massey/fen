"""
Online detection: latency and false alarms.

AUC is threshold-free and offline, which is convenient for evaluation and
useless to an organism. A creature needs a decision *now*: something is
happening to me, react. That requires a threshold, and a threshold has a cost.

This measures what a usable decision rule actually delivers:

  - the threshold that holds false alarms to a target rate on clean steps
  - how many steps after onset the surprise crosses it
  - what fraction of perturbations are caught at all

    python3 detect.py --run runs/<dir> --fa-rate 0.01
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from babble import Babbler
from env import Body
from model import ForwardModel, FrameStack


def rollout(fm: ForwardModel, stack_k: int, force: float, ramp_prob: float,
            steps: int, seed: int):
    body = Body(seed=seed, perturb_force=force, ramp_prob=ramp_prob)
    bab = Babbler(body.nu, seed=seed + 31)
    stack = FrameStack(stack_k, body.obs_dim, body.nu)

    obs = body.reset()
    stack.reset(obs)
    surprise, perturbed, onset, ramp = [], [], [], []

    for _ in range(steps):
        feat = stack.feature()
        act = bab()
        nxt, info = body.step(act)
        with torch.no_grad():
            s = fm.surprise(
                torch.tensor(feat).unsqueeze(0),
                torch.tensor(act, dtype=torch.float32).unsqueeze(0),
                torch.tensor(nxt - obs, dtype=torch.float32).unsqueeze(0),
            ).item()
        surprise.append(s)
        perturbed.append(info["perturbed"])
        onset.append(info["onset"])
        ramp.append(info["ramp"])
        stack.push(nxt, act)
        obs = nxt

    return (np.array(surprise), np.array(perturbed),
            np.array(onset), np.array(ramp))


def analyse(surprise, perturbed, onset, ramp, fa_rate: float, window: int) -> dict:
    clean = ~perturbed
    if clean.sum() < 100:
        return {}

    # Threshold set purely from clean steps: the organism can calibrate this
    # from its own quiet periods without ever seeing a perturbation label.
    thr = float(np.quantile(surprise[clean], 1.0 - fa_rate))
    fired = surprise > thr

    onsets = np.flatnonzero(onset)
    latencies, caught, kinds = [], 0, []
    for o in onsets:
        end = min(o + window, len(surprise))
        hits = np.flatnonzero(fired[o:end])
        kinds.append(bool(ramp[o]))
        if len(hits):
            caught += 1
            latencies.append(int(hits[0]))

    lat = np.array(latencies) if latencies else np.array([np.nan])
    kinds = np.array(kinds)

    def rate_for(is_ramp: bool) -> float:
        sel = [i for i, o in enumerate(onsets) if bool(ramp[o]) is is_ramp]
        if not sel:
            return float("nan")
        hit = sum(
            1 for i in sel
            if fired[onsets[i]: min(onsets[i] + window, len(surprise))].any()
        )
        return hit / len(sel)

    return {
        "threshold": thr,
        "target_false_alarm_rate": fa_rate,
        "observed_false_alarms_per_1000_clean": float(fired[clean].mean() * 1000),
        "n_bursts": int(len(onsets)),
        "detection_rate_all": float(caught / max(len(onsets), 1)),
        "detection_rate_step": rate_for(False),
        "detection_rate_ramp": rate_for(True),
        "median_latency_steps": float(np.nanmedian(lat)),
        "p90_latency_steps": float(np.nanpercentile(lat, 90)),
        "window": window,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, required=True)
    p.add_argument("--steps", type=int, default=40_000)
    p.add_argument("--fa-rate", type=float, default=0.01)
    p.add_argument("--window", type=int, default=20, help="steps allowed to detect")
    p.add_argument("--seed", type=int, default=23)
    args = p.parse_args()

    run = Path(args.run)
    cfg = json.loads((run / "args.json").read_text())
    k = cfg["stack"]

    body = Body(seed=0)
    stack = FrameStack(k, body.obs_dim, body.nu)
    fm = ForwardModel(stack.feat_dim, body.nu, body.obs_dim)
    fm.load_state_dict(torch.load(run / "forward_model.pt"))
    fm.eval()

    print(f"{'force':>7} {'det step':>10} {'det ramp':>10} "
          f"{'med lat':>9} {'p90 lat':>9} {'FA/1000':>9}")
    print("-" * 60)

    out = []
    for force in [4.0, 8.0, 12.0, 20.0]:
        data = rollout(fm, k, force, cfg["ramp_prob"], args.steps, args.seed)
        m = analyse(*data, fa_rate=args.fa_rate, window=args.window)
        if not m:
            continue
        m["force"] = force
        out.append(m)
        print(f"{force:>7.1f} {m['detection_rate_step']:>10.3f} "
              f"{m['detection_rate_ramp']:>10.3f} "
              f"{m['median_latency_steps']:>9.1f} {m['p90_latency_steps']:>9.1f} "
              f"{m['observed_false_alarms_per_1000_clean']:>9.1f}")

    (run / "detection.json").write_text(json.dumps(out, indent=2))
    print(f"\nwritten to {run / 'detection.json'}")
    print(f"Threshold calibrated on clean steps only, targeting a "
          f"{args.fa_rate:.0%} false-alarm rate.")


if __name__ == "__main__":
    main()
