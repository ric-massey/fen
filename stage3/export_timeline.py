"""
Export a TRAINING TIMELINE viewer: one rollout per checkpoint.

Showing a single recording of a finished policy is showing the answer. What is
actually interesting — and what is honest — is watching it across training:
flailing early, and either improving or not. If it does not improve, the page
should show that rather than quietly displaying the hand-written policy instead.

    python3 export_timeline.py --run ../stage2/runs/<dir> --out ../web/fen.html
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "stage2"))
from env import Stage2  # noqa: E402
from sac import Actor  # noqa: E402

from export_web import load_results, scripted  # noqa: E402
from transformer_core import TransformerWorldModel  # noqa: E402

TEMPLATE = Path(__file__).parent / "viewer_template.html"


def roll(env, wm, act_fn, steps, stride):
    o = env.reset()
    h = wm.init_state(1)
    prev = np.zeros(env.nu, dtype=np.float32)
    e, t = 1.0, 0.0
    frames, lives, cur = [], [], 0

    for i in range(steps):
        a = act_fn(o, e, t)
        nxt, r, d, info = env.step(a)
        executed = info.get("executed_action", a)
        with torch.no_grad():
            _, idx, h, _ = wm.step(
                torch.as_tensor(o, dtype=torch.float32).unsqueeze(0),
                torch.as_tensor(prev, dtype=torch.float32).unsqueeze(0), h)
        cur += 1
        if i % stride == 0:
            tip = env._tip_xy()
            frames.append([
                round(float(info["x"]), 3), round(float(info["y"]), 3),
                round(float(tip[0]), 3), round(float(tip[1]), 3),
                round(float(info["energy"]), 3), round(float(info["temp"]), 3),
                round(float(info["health"]), 3), int(idx.item()),
                round(float(info["ate"]), 4),
                {None: 0, "temp": 1, "energy": 2}[info.get("reflex")],
                [round(float(c), 2) for c in info["food_charge"]],
                1 if info["dead"] else 0,
                [round(float(v), 3) for v in info["thermal_axis"]],
                [[round(float(q), 2) for q in f[:2]] for f in env.food.pos],
            ])
        e, t = info["energy"], info["temp"]
        prev = executed
        if info["dead"]:
            lives.append(cur); cur = 0
            o = env.reset(); h = wm.init_state(1); e, t = 1.0, 0.0
        else:
            o = nxt
    return frames, (float(np.mean(lives)) if lives else float(steps))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--out", type=str, default="../web/fen.html")
    a = p.parse_args()

    env = Stage2(seed=a.seed)
    world = max(glob.glob("runs/tfm*/attention.pt"),
                key=lambda q: pathlib.Path(q).stat().st_mtime)
    wm = TransformerWorldModel(env.obs_dim, env.nu, n_codes=128)
    wm.load_state_dict(torch.load(world)); wm.eval()

    ckpts = sorted(glob.glob(f"{a.run}/actor_*.pt"),
                   key=lambda q: int(re.search(r"actor_(\d+)", q).group(1)))
    clips = []

    for c in ckpts:
        step = int(re.search(r"actor_(\d+)", c).group(1))
        actor = Actor(env.obs_dim, env.nu, env.act_low, env.act_high)
        actor.load_state_dict(torch.load(c)); actor.eval()

        def act_fn(o, e, t, _a=actor):
            with torch.no_grad():
                v, _ = _a(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0),
                          deterministic=True, with_logprob=False)
            return v.squeeze(0).numpy()

        f, life = roll(env, wm, act_fn, a.steps, a.stride)
        clips.append({"label": f"learned @ {step//1000}k steps", "frames": f,
                      "cycles": round(life / 1000, 2)})
        print(f"  {clips[-1]['label']}: {life:.0f} steps ({life/1000:.1f} cycles)")

    f, life = roll(env, wm, scripted, a.steps, a.stride)
    clips.append({"label": "hand-written (6 lines)", "frames": f,
                  "cycles": round(life / 1000, 2)})
    print(f"  hand-written: {life:.0f} steps ({life/1000:.1f} cycles)")

    data = {"food": env.food.pos[:, :2].round(3).tolist(), "reach": float(env.food.reach),
            "arena": 3.0, "dt": float(env.dt * a.stride), "n_codes": 128,
            "clips": clips, "frames": clips[0]["frames"],
            "results": load_results()}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE.read_text().replace(
        "/*__DATA__*/null", json.dumps(data, separators=(",", ":"))))
    print(f"wrote {out} ({len(clips)} clips, {out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
