"""
Stage 1 training loop.

Motor babbling generates experience; the forward model learns a distribution
over the sensory consequences of its own actions; surprise (the NLL of what
actually happened) is measured against unpredictable external perturbations.

Gate (FEN.md, Stage 1): AUC >= 0.9 separating self-caused from externally-caused
sensory change. Reported separately for step-onset and ramped perturbations —
the ramped number is the honest one.

    python3 train.py --steps 300000 --stack 4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from babble import Babbler
from env import Body
from model import Buffer, ForwardModel, FrameStack

OUT = Path(__file__).parent / "runs"


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    if label.sum() < 20 or (~label).sum() < 20:
        return float("nan")
    return float(roc_auc_score(label, score))


def evaluate(fm: ForwardModel, buf: Buffer) -> dict:
    """Scored on held-out transitions only."""
    feat, act, delta, pert, ramp = buf.eval_split()
    if len(feat) < 100:
        return {}

    with torch.no_grad():
        t = (torch.tensor(feat), torch.tensor(act), torch.tensor(delta))
        s = fm.surprise(*t).numpy()
        r = fm.point_residual(*t).numpy()

    clean = ~pert
    step_mask = clean | (pert & ~ramp)      # clean vs step-onset perturbations
    ramp_mask = clean | (pert & ramp)       # clean vs ramped perturbations

    return {
        "auc_all": _auc(s, pert),
        "auc_step": _auc(s[step_mask], pert[step_mask]),
        "auc_ramp": _auc(s[ramp_mask], pert[ramp_mask]),
        "auc_point_ramp": _auc(r[ramp_mask], pert[ramp_mask]),
        "surprise_self": float(s[clean].mean()),
        "surprise_world": float(s[pert].mean()),
        "n_eval": int(len(feat)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stack", type=int, default=4, help="frames of history (1 = no context)")
    p.add_argument("--buffer", type=int, default=100_000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-every", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--warmup", type=int, default=5_000, help="steps before any training")
    p.add_argument("--point-warmup", type=int, default=40_000, help="Huber steps before NLL")
    p.add_argument("--force", type=float, default=12.0)
    p.add_argument("--ramp-prob", type=float, default=0.5)
    p.add_argument("--tag", type=str, default="dev")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    device = "cpu"  # model is small; MPS transfer overhead dominates
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    body = Body(seed=args.seed, perturb_force=args.force, ramp_prob=args.ramp_prob)
    babbler = Babbler(body.nu, seed=args.seed)
    stack = FrameStack(args.stack, body.obs_dim, body.nu)
    buf = Buffer(args.buffer, stack.feat_dim, body.nu, body.obs_dim, seed=args.seed)
    fm = ForwardModel(stack.feat_dim, body.nu, body.obs_dim).to(device)
    opt = torch.optim.Adam(fm.parameters(), lr=args.lr)

    OUT.mkdir(exist_ok=True)
    run_dir = OUT / f"{args.tag}-s{args.seed}-k{args.stack}-{time.strftime('%H%M%S')}"
    run_dir.mkdir(parents=True)
    history: list[dict] = []

    obs = body.reset()
    stack.reset(obs)
    normalised = False
    t0 = time.time()

    for step in range(args.steps):
        feat = stack.feature()
        act = babbler()
        nxt, info = body.step(act)
        buf.add(feat, act, nxt - obs, info["perturbed"], info["ramp"])
        stack.push(nxt, act)
        obs = nxt

        if step == args.warmup:
            f, a, d, _, _ = buf.all()
            fm.fit_normalisation(f, d)
            normalised = True

        if normalised and step % args.train_every == 0:
            batch = buf.sample(args.batch, rng)
            if batch is not None:
                bf, ba, bd = (torch.tensor(x, device=device) for x in batch)
                # Point warmup, then switch to NLL. Training variance from
                # scratch tends to collapse to "everything is uncertain".
                loss = (fm.huber(bf, ba, bd) if step < args.point_warmup
                        else fm.nll(bf, ba, bd))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
                opt.step()

        if normalised and step > 0 and step % args.eval_every == 0:
            m = evaluate(fm, buf)
            if m:
                m["step"] = step
                history.append(m)
                if not args.quiet:
                    print(f"step {step:>7}  step-onset {m['auc_step']:.3f}  "
                          f"ramped {m['auc_ramp']:.3f}  "
                          f"(point-model ramped {m['auc_point_ramp']:.3f})  "
                          f"{time.time() - t0:.0f}s")

    final = evaluate(fm, buf)
    final.update({"step": args.steps, "seed": args.seed, "stack": args.stack,
                  "gate_passed_ramped": bool(final.get("auc_ramp", 0) >= 0.9)})

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "final.json").write_text(json.dumps(final, indent=2))
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    torch.save(fm.state_dict(), run_dir / "forward_model.pt")

    if not args.quiet:
        print("\n" + "=" * 62)
        print(f"FINAL (held-out)  step-onset {final['auc_step']:.4f}   "
              f"ramped {final['auc_ramp']:.4f}")
        print(f"  gate on ramped: {'PASSED' if final['gate_passed_ramped'] else 'NOT PASSED'}")
        print(f"  {run_dir}")
        print("=" * 62)
    else:
        print(json.dumps(final))


if __name__ == "__main__":
    main()
