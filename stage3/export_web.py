"""
Export a rollout as a self-contained HTML viewer.

Video is fine for watching once. This is better for a website: top-down, scrubbable,
pausable, and small enough to embed. Everything ends up inside one .html file with
the trajectory baked in — no server, no assets, no dependencies. Drop it anywhere.

Top-down rather than a 3D camera on purpose. A first-person view of a creature
wandering around looks impressive and tells you almost nothing; from above, with
the drives and the symbol stream alongside, you can actually read what it is doing
and why.

    python3 export_web.py --steps 6000 --out ../web/fen.html
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import pathlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "stage2"))
from env import Stage2  # noqa: E402

from transformer_core import TransformerWorldModel  # noqa: E402

TEMPLATE = Path(__file__).parent / "viewer_template.html"


def scripted(o, energy, temp, tcap=0.45):
    # obs: smell 14:17 | thermal gradient 17:19 | drives 19:22
    a = np.zeros(6)
    g = o[17:19]
    if abs(temp) > tcap:
        # move DOWN the thermal gradient when hot, up when cold. The gradient is
        # the same either way; which way to go comes from its own temperature.
        a[0], a[1] = -np.sign(temp) * 1.2 * g[0], -np.sign(temp) * 1.2 * g[1]
    elif energy < 0.9:
        a[0] = np.clip(o[14] * 1.2, -1.2, 1.2)
        a[1] = np.clip(o[15] * 1.2, -1.2, 1.2)
    return a


def rollout(steps: int, seed: int, world: str, stride: int = 2) -> dict:
    env = Stage2(seed=seed)
    wm = TransformerWorldModel(env.obs_dim, env.nu, n_codes=128)
    wm.load_state_dict(torch.load(world))
    wm.eval()

    o = env.reset()
    h = wm.init_state(1)
    prev = np.zeros(env.nu, dtype=np.float32)
    e, t = 1.0, 0.0
    frames = []

    for i in range(steps):
        act = scripted(o, e, t)
        nxt, r, d, info = env.step(act)
        executed = info.get("executed_action", act)
        with torch.no_grad():
            _, idx, h, _ = wm.step(
                torch.as_tensor(o, dtype=torch.float32).unsqueeze(0),
                torch.as_tensor(prev, dtype=torch.float32).unsqueeze(0), h)

        if i % stride == 0:
            tip = env._tip_xy()
            frames.append([
                round(float(info["x"]), 3), round(float(info["y"]), 3),
                round(float(tip[0]), 3), round(float(tip[1]), 3),
                round(float(info["energy"]), 3), round(float(info["temp"]), 3),
                round(float(info["health"]), 3),
                int(idx.item()),
                round(float(info["ate"]), 4),
                {None: 0, "temp": 1, "energy": 2}[info.get("reflex")],
                [round(float(c), 2) for c in info["food_charge"]],
                1 if info["dead"] else 0,
                # thermal axis is re-drawn each episode, so it is per-frame data
                [round(float(v), 3) for v in info["thermal_axis"]],
                [[round(float(q), 2) for q in f[:2]] for f in env.food.pos],
                [round(float(q), 3) for q in info["hazard"]],
                round(float(info["arousal"]), 3),
            ])

        e, t = info["energy"], info["temp"]
        prev = executed
        if info["dead"]:
            o = env.reset()
            h = wm.init_state(1)
            e, t = 1.0, 0.0
        else:
            o = nxt

    return {
        "food": env.food.pos[:, :2].round(3).tolist(),
        "reach": float(env.food.reach), "hazrad": float(env.hazard.radius),
        "arena": 3.0,
        "dt": float(env.dt * stride),
        "n_codes": 128,
        "frames": frames,
        "schema": ["x", "y", "tipx", "tipy", "energy", "temp", "health",
                   "code", "ate", "reflex", "charge", "dead", "axis", "food",
                   "hazard", "arousal"],
    }


def load_results() -> dict | None:
    """Attach the measurements so the page shows the evidence, not just the show."""
    out = {}
    tfm = sorted(glob.glob("runs/tfm*/results.json"))
    if tfm:
        r = json.load(open(tfm[-1]))
        key = "attention" if "attention" in r else "recurrent"
        out["mi"] = {k: [round(r[key]["mi"][k], 3), round(r["untrained"]["mi"][k], 3)]
                     for k in r[key]["mi"]}
    ctl = Path("../stage2/runs/controls.json")
    if ctl.exists():
        out["controls"] = json.load(open(ctl))
    sw = Path("../stage2/runs/sweep.json")
    if sw.exists():
        rows = json.load(open(sw))
        learned = float(np.mean([q["lifespan_cycles"] for q in rows]))
        out["survival"] = [["hand-written (6 lines)", 30.0],
                           ["learned (SAC)", round(learned, 2)],
                           ["doing nothing", 1.0],
                           ["random", 0.4]]
    return out or None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--world", type=str, default=None)
    p.add_argument("--out", type=str, default="../web/fen.html")
    a = p.parse_args()

    world = a.world or max(glob.glob("runs/tfm*/attention.pt"), key=lambda q: pathlib.Path(q).stat().st_mtime)
    print(f"world model: {world}")
    data = rollout(a.steps, a.seed, world, a.stride)
    data["results"] = load_results()

    html = TEMPLATE.read_text().replace(
        "/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    kb = out.stat().st_size / 1024
    print(f"wrote {out}  ({len(data['frames'])} frames, {kb:.0f} KB, self-contained)")


if __name__ == "__main__":
    main()
