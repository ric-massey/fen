"""
Watch the organism — body and internal state together.

Renders the world alongside a live readout of what is happening inside: its
drives, and critically the SYMBOL it is currently emitting from its learned
codebook.

The code strip along the bottom is the interesting part. Each colour is one of
the symbols the organism invented for itself. If the vocabulary means anything,
you should see it settle into one symbol while travelling, switch to another at
a food site, another again when it retreats from the heat — distinct situations
getting distinct symbols, without anyone having named them.

Frames are composited in numpy rather than matplotlib, which is ~20x faster and
matters at 900 frames.

    python3 watch.py --steps 3000 --policy scripted
"""
from __future__ import annotations

import argparse
import glob
import sys
import pathlib
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "stage2"))
from env import Stage2  # noqa: E402

from transformer_core import TransformerWorldModel  # noqa: E402

PANEL_W = 300
BG = np.array([21, 21, 26], dtype=np.uint8)


def code_colour(idx: int) -> np.ndarray:
    """Stable distinct colour per symbol — golden-angle hue rotation."""
    h = (idx * 0.618033988749895) % 1.0
    i = int(h * 6)
    f = h * 6 - i
    v, p, q, t = 235, 60, int(235 * (1 - f) * 0.75 + 60), int(235 * f * 0.75 + 60)
    return np.array([(v, t, p), (q, v, p), (p, v, t),
                     (p, q, v), (t, p, v), (v, p, q)][i % 6], dtype=np.uint8)


def bar(canvas, x, y, w, h, frac, colour, bidirectional=False):
    canvas[y:y + h, x:x + w] = np.array([40, 40, 48], dtype=np.uint8)
    if bidirectional:
        mid = x + w // 2
        n = int(abs(frac) * (w // 2))
        if frac >= 0:
            canvas[y:y + h, mid:mid + n] = colour
        else:
            canvas[y:y + h, mid - n:mid] = colour
        canvas[y:y + h, mid:mid + 1] = np.array([140, 140, 150], dtype=np.uint8)
    else:
        canvas[y:y + h, x:x + int(max(frac, 0) * w)] = colour


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--every", type=int, default=3, help="render every Nth step")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--world", type=str, default=None)
    p.add_argument("--out", type=str, default="viz/watch_inside.mp4")
    a = p.parse_args()

    env = Stage2(seed=a.seed)
    world = a.world or max(glob.glob("runs/tfm*/attention.pt"), key=lambda q: pathlib.Path(q).stat().st_mtime)
    wm = TransformerWorldModel(env.obs_dim, env.nu, n_codes=128)
    wm.load_state_dict(torch.load(world))
    wm.eval()
    print(f"world model: {world}")

    H, W = 480, 640
    renderer = mujoco.Renderer(env.model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 90, -58, 8.5
    cam.lookat[:] = [0, 0, 0.5]

    strip_h, strip_len = 42, PANEL_W - 24
    strip = np.tile(BG, (strip_h, strip_len, 1))

    o = env.reset()
    h = wm.init_state(1)
    prev_act = np.zeros(env.nu, dtype=np.float32)
    e, t = 1.0, 0.0
    frames = []

    for i in range(a.steps):
        act = scripted(o, e, t)
        nxt, r, d, info = env.step(act)
        executed = info.get("executed_action", act)

        with torch.no_grad():
            _, idx, h, _ = wm.step(
                torch.as_tensor(o, dtype=torch.float32).unsqueeze(0),
                torch.as_tensor(prev_act, dtype=torch.float32).unsqueeze(0), h)
        code = int(idx.item())

        col = code_colour(code)
        strip = np.roll(strip, -2, axis=1)
        strip[:, -2:] = col

        if i % a.every == 0:
            renderer.update_scene(env.data, cam)
            canvas = np.tile(BG, (H, W + PANEL_W, 1))
            canvas[:, :W] = renderer.render()
            px = W + 16

            bar(canvas, px, 60, PANEL_W - 32, 22, info["energy"],
                np.array([95, 208, 106], dtype=np.uint8))
            bar(canvas, px, 120, PANEL_W - 32, 22, info["temp"],
                np.array([224, 135, 63], dtype=np.uint8), bidirectional=True)
            bar(canvas, px, 180, PANEL_W - 32, 22, info["health"],
                np.array([122, 166, 224], dtype=np.uint8))

            # current symbol: a solid block of its colour
            canvas[240:300, px:px + 60] = col
            canvas[240:300, px + 60:px + 64] = BG

            canvas[360:360 + strip_h, px:px + strip_len] = strip
            frames.append(canvas)

        e, t = info["energy"], info["temp"]
        prev_act = executed
        if info["dead"]:
            o = env.reset()
            h = wm.init_state(1)
            e, t = 1.0, 0.0
        else:
            o = nxt

    Path(a.out).parent.mkdir(exist_ok=True)
    imageio.mimsave(a.out, frames, fps=30)
    print(f"wrote {a.out} ({len(frames)} frames)")
    print("panel top-to-bottom: energy | temperature | health | current symbol | symbol history")


if __name__ == "__main__":
    main()
