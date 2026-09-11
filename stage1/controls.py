"""
Controls for the Stage 1 result.

A high AUC on its own proves nothing. Three ways it could be an artifact:

  1. The model isn't doing the work — an untrained network might separate the
     classes just as well, because the residual tracks something trivial like
     velocity magnitude. This is the control that matters most.

  2. The perturbations are enormous relative to normal dynamics, so "detection"
     is just noticing a large shove. The honest result is not one AUC but the
     curve of AUC against perturbation strength: where does the boundary fail?

  3. Evaluation on data the model trained on. Needs a held-out sample.

    python3 controls.py --run runs/<tag>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from babble import Babbler
from env import Body
from model import Buffer, ForwardModel


def collect(force: float, steps: int, seed: int, obs_dim: int, nu: int) -> Buffer:
    """Fresh rollout at a given perturbation strength, never trained on."""
    body = Body(seed=seed, perturb_force=force)
    bab = Babbler(nu, seed=seed + 999)
    buf = Buffer(steps, obs_dim, nu)
    obs = body.reset()
    for _ in range(steps):
        act = bab()
        nxt, pert = body.step(act)
        buf.add(obs, act, nxt - obs, pert)
        obs = nxt
    return buf


def auc_of(fm: ForwardModel, buf: Buffer) -> float:
    obs, act, delta, pert = buf.all()
    if pert.sum() < 20 or (~pert).sum() < 20:
        return float("nan")
    with torch.no_grad():
        r = fm.residual(torch.tensor(obs), torch.tensor(act), torch.tensor(delta)).numpy()
    return float(roc_auc_score(pert, r))


def velocity_baseline(buf: Buffer) -> float:
    """Trivial statistic: does raw magnitude of sensory change separate them?

    If this scores as well as the model, no boundary was learned — the residual
    is just tracking how much things moved.
    """
    _, _, delta, pert = buf.all()
    score = np.linalg.norm(delta, axis=1)
    if pert.sum() < 20 or (~pert).sum() < 20:
        return float("nan")
    return float(roc_auc_score(pert, score))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, required=True)
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    run = Path(args.run)
    trained = ForwardModel(8, 4)
    trained.load_state_dict(torch.load(run / "forward_model.pt"))
    trained.eval()

    # Untrained model, but given the same normalisation, so the only difference
    # is whether the network learned anything.
    untrained = ForwardModel(8, 4)
    untrained.obs_mean.copy_(trained.obs_mean)
    untrained.obs_std.copy_(trained.obs_std)
    untrained.delta_mean.copy_(trained.delta_mean)
    untrained.delta_std.copy_(trained.delta_std)
    untrained.eval()

    print(f"{'force':>7} {'trained':>9} {'untrained':>10} {'|delta|':>9}   verdict")
    print("-" * 58)

    results = []
    for force in [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0]:
        buf = collect(force, args.steps, args.seed, 8, 4)
        a_tr = auc_of(trained, buf)
        a_un = auc_of(untrained, buf)
        a_vel = velocity_baseline(buf)
        margin = a_tr - max(a_un, a_vel)
        verdict = "model earns it" if margin > 0.05 else "NOT the model"
        results.append({"force": force, "trained": a_tr, "untrained": a_un,
                        "velocity": a_vel, "margin": margin})
        print(f"{force:>7.1f} {a_tr:>9.3f} {a_un:>10.3f} {a_vel:>9.3f}   {verdict}")

    (run / "controls.json").write_text(json.dumps(results, indent=2))
    print(f"\nwritten to {run / 'controls.json'}")
    print("\nRead the curve, not the peak. The force at which 'trained' falls")
    print("below 0.9 is the real sensitivity of the self/world boundary.")


if __name__ == "__main__":
    main()
