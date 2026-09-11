"""
Wire the Stage 3 representation into the policy.

The question: Stage 2 failed at managing accumulated health damage. The
explanation offered was that health moves over ~1000 steps and depends on the
*history* of regulation, which a feedforward policy reading an instantaneous
observation cannot represent. Stage 3 built that capacity. Does it fix it?

Deliberately the simple version first. The trained world model is FROZEN and used
as a feature extractor; SAC runs on top of its hidden state. That isolates "is
the representation useful" from "can recurrent RL be trained", which are two hard
problems that should not be debugged together. If frozen features help, end-to-end
recurrent SAC is worth building. If they do not, it would not have saved it.

Three conditions, same seed, same budget:

  raw        SAC on the raw observation. This is Stage 2's setup, the baseline.
  +hidden    SAC on observation + frozen recurrent hidden state.
  +code      SAC on observation + the discrete code only. Tests whether the
             symbol alone carries what the policy needs, or whether it needs the
             continuous state behind it.

    python3 train_policy.py --world runs/full-*/recurrent.pt --steps 400000
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "stage2"))
from env import Stage2  # noqa: E402
from sac import SAC  # noqa: E402

from world_model import WorldModel  # noqa: E402


class Features:
    """Frozen world model as an online feature extractor."""

    def __init__(self, path: str, obs_dim: int, act_dim: int, mode: str):
        self.mode = mode
        self.wm = WorldModel(obs_dim, act_dim, n_codes=128)
        self.wm.load_state_dict(torch.load(path))
        self.wm.eval()
        for p in self.wm.parameters():
            p.requires_grad_(False)
        self.h = self.wm.init_state(1)
        self.extra = {"raw": 0, "+hidden": self.wm.hidden,
                      "+code": self.wm.vq.dim}[mode]

    def reset(self) -> None:
        self.h = self.wm.init_state(1)

    @torch.no_grad()
    def __call__(self, obs: np.ndarray, prev_act: np.ndarray) -> np.ndarray:
        if self.mode == "raw":
            return obs
        code, _, self.h, _ = self.wm.step(
            torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(prev_act, dtype=torch.float32).unsqueeze(0),
            self.h,
        )
        tail = (self.h if self.mode == "+hidden" else code).squeeze(0).numpy()
        return np.concatenate([obs, tail]).astype(np.float32)


def run(mode: str, world_path: str, steps: int, seed: int, report_every: int):
    env = Stage2(seed=seed)
    feat = Features(world_path, env.obs_dim, env.nu, mode)
    in_dim = env.obs_dim + feat.extra
    agent = SAC(in_dim, env.nu, env.act_low, env.act_high, gamma=0.999, seed=seed)
    rng = np.random.default_rng(seed)

    obs = env.reset()
    feat.reset()
    prev_act = np.zeros(env.nu, dtype=np.float32)
    x = feat(obs, prev_act)

    lifespans, all_lifespans = deque(maxlen=50), []
    since, deaths = 0, 0
    hist = []
    t0 = time.time()

    for step in range(steps):
        a = rng.uniform(env.act_low, env.act_high) if step < 5000 else agent.act(x)
        nxt, r, done, info = env.step(a)
        executed = info.get("executed_action", a)
        x2 = feat(nxt, executed)
        agent.replay.add(x, executed, r, x2, info.get("dead", False))

        since += 1
        if info.get("dead"):
            lifespans.append(since)
            all_lifespans.append(since)
            since, deaths = 0, deaths + 1
            obs = env.reset()
            feat.reset()
            prev_act = np.zeros(env.nu, dtype=np.float32)
            x = feat(obs, prev_act)
        else:
            obs, prev_act, x = nxt, executed, x2

        if step >= 5000 and step % 2 == 0:
            agent.update(256)

        if step > 0 and step % report_every == 0:
            m = {"step": step, "deaths": deaths,
                 "recent_lifespan": float(np.mean(lifespans)) if lifespans else float(step),
                 "all_lifespan": float(np.mean(all_lifespans)) if all_lifespans else float(step),
                 "health": info["health"], "elapsed": round(time.time() - t0)}
            hist.append(m)
            print(f"  [{mode:8}] {step:>7}  recent {m['recent_lifespan']:>7.0f}  "
                  f"all {m['all_lifespan']:>7.0f}  deaths {deaths:>4}  "
                  f"hp {m['health']:.2f}  ({m['elapsed']}s)", flush=True)

    # Report ALL lifespans, not a trailing window: the deque hides early
    # catastrophic behaviour and biases toward late performance.
    return {"mode": mode, "deaths": deaths, "steps": steps,
            "mean_lifespan_all": float(np.mean(all_lifespans)) if all_lifespans else float(steps),
            "median_lifespan_all": float(np.median(all_lifespans)) if all_lifespans else float(steps),
            "mean_lifespan_recent50": float(np.mean(lifespans)) if lifespans else float(steps),
            "history": hist}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--world", type=str, default=None)
    p.add_argument("--steps", type=int, default=400_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report-every", type=int, default=100_000)
    p.add_argument("--modes", type=str, nargs="+", default=["raw", "+hidden", "+code"])
    p.add_argument("--tag", type=str, default="dev")
    a = p.parse_args()

    world = a.world or sorted(glob.glob("runs/full-*/recurrent.pt"))[-1]
    print(f"frozen world model: {world}\n")

    out = Path("runs") / f"policy-{a.tag}-s{a.seed}-{time.strftime('%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for mode in a.modes:
        print(f"=== {mode} ===")
        results[mode] = run(mode, world, a.steps, a.seed, a.report_every)
        (out / "results.json").write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 66)
    print(f"{'mode':10} {'mean life':>11} {'median':>9} {'deaths':>8}")
    for k, v in results.items():
        print(f"{k:10} {v['mean_lifespan_all']:11.0f} "
              f"{v['median_lifespan_all']:9.0f} {v['deaths']:8d}")
    print(f"\nStage 2 baseline for reference: ~673 with health, reflexes on")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
