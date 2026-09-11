"""
Visualisation for Stage 2.

Two modes, and the default is deliberately not a game view:

  --mode plot   top-down arena + drive traces (PNG). This is the one to use for
                debugging. Render the world from above and read the numbers;
                a first-person view of a creature wandering around tells you
                almost nothing about why it did what it did.

  --mode video  offscreen MuJoCo render (MP4). For seeing that the body moves
                the way you think it does — worth doing once per body change,
                since the arm-through-the-floor bug was invisible in every
                metric and obvious in one frame.

    python3 viz.py --run runs/reflex-... --mode plot
    python3 viz.py --policy scripted --mode video
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from env import Stage2


def scripted_action(o, energy, temp, tcap=0.45):
    a = np.zeros(6)
    if abs(temp) > tcap:
        a[0] = -np.sign(temp) * 1.2
    elif energy < 0.9:
        a[0] = np.clip(o[14] * 1.2, -1.2, 1.2)
        a[1] = np.clip(o[15] * 1.2, -1.2, 1.2)
    return a


def rollout(env, policy, steps, actor=None):
    import torch
    log = {k: [] for k in ("x", "y", "energy", "temp", "reflex", "ate", "effort", "death")}
    o = env.reset()
    e, t = 1.0, 0.0
    for _ in range(steps):
        if policy == "scripted":
            a = scripted_action(o, e, t)
        elif policy == "random":
            a = np.random.uniform(env.act_low, env.act_high)
        elif policy == "idle":
            a = np.zeros(6)
        else:
            with torch.no_grad():
                a, _ = actor(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0),
                             deterministic=True, with_logprob=False)
            a = a.squeeze(0).numpy()
        o, r, d, info = env.step(a)
        e, t = info["energy"], info["temp"]
        for k in ("x", "y", "energy", "temp", "ate", "effort"):
            log[k].append(info[k])
        log["reflex"].append(info.get("reflex"))
        log["death"].append(bool(info["dead"]))
        if info["dead"]:
            o = env.reset()
            e, t = 1.0, 0.0
    return {k: np.array(v, dtype=object if k == "reflex" else None) for k, v in log.items()}


def make_plot(env, log, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig = plt.figure(figsize=(14, 9), facecolor="#15151a")
    gs = fig.add_gridspec(4, 2, width_ratios=[1.15, 1], hspace=0.45, wspace=0.22)
    for ax in []:
        pass

    # ── top-down arena ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[:, 0], facecolor="#1c1c23")
    ax.add_patch(plt.Rectangle((0, -3), 3, 6, color="#6b3226", alpha=0.30, lw=0))
    ax.add_patch(plt.Rectangle((-3, -3), 3, 6, color="#26406b", alpha=0.30, lw=0))
    ax.text(1.5, 2.7, "WARM", color="#c8836a", ha="center", fontsize=11, weight="bold")
    ax.text(-1.5, 2.7, "COOL", color="#6a94c8", ha="center", fontsize=11, weight="bold")

    pts = np.array([log["x"], log["y"]]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="viridis", norm=plt.Normalize(0, 1), lw=1.2, alpha=0.85)
    lc.set_array(log["energy"][:-1])
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.02, label="energy")

    for fx, fy in env.food.pos[:, :2]:
        ax.scatter([fx], [fy], s=420, facecolor="none", edgecolor="#5fd06a", lw=2.0, zorder=5)
        ax.scatter([fx], [fy], s=60, color="#5fd06a", zorder=5)

    deaths = np.flatnonzero(log["death"])
    if len(deaths):
        ax.scatter(log["x"][deaths], log["y"][deaths], marker="x", s=110,
                   color="#ff4d4d", lw=2.2, zorder=6, label=f"deaths ({len(deaths)})")
        ax.legend(loc="lower left", facecolor="#1c1c23", labelcolor="#ddd", edgecolor="#444")

    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
    ax.set_title(title, color="#eee", fontsize=13, pad=12)
    for s in ax.spines.values():
        s.set_color("#444")
    ax.tick_params(colors="#888")

    # ── traces ───────────────────────────────────────────────────────────────
    def trace(row, y, color, label, hlines=()):
        a = fig.add_subplot(gs[row, 1], facecolor="#1c1c23")
        a.plot(y, color=color, lw=1.0)
        for hv, hc in hlines:
            a.axhline(hv, color=hc, ls="--", lw=0.9, alpha=0.7)
        for d in deaths:
            a.axvline(d, color="#ff4d4d", lw=0.7, alpha=0.5)
        a.set_ylabel(label, color="#ccc", fontsize=9)
        a.tick_params(colors="#888", labelsize=8)
        for s in a.spines.values():
            s.set_color("#444")
        return a

    trace(0, log["energy"], "#5fd06a", "energy", [(0.0, "#ff4d4d"), (0.30, "#ffa64d")])
    trace(1, log["temp"], "#e0873f", "body temp",
          [(0.92, "#ff4d4d"), (-0.92, "#ff4d4d"), (0.55, "#ffa64d"), (-0.55, "#ffa64d")])
    trace(2, log["x"], "#7aa6e0", "x position", [(0.0, "#666")])

    a = fig.add_subplot(gs[3, 1], facecolor="#1c1c23")
    rt = np.array([1 if r == "temp" else 0 for r in log["reflex"]])
    re = np.array([1 if r == "energy" else 0 for r in log["reflex"]])
    a.fill_between(range(len(rt)), 0, rt, color="#e0873f", step="mid", label="thermal")
    a.fill_between(range(len(re)), 0, -re, color="#5fd06a", step="mid", label="feeding")
    a.set_ylim(-1.2, 1.2); a.set_yticks([])
    a.set_ylabel("reflex", color="#ccc", fontsize=9)
    a.set_xlabel("step", color="#ccc", fontsize=9)
    a.legend(loc="upper right", fontsize=7, facecolor="#1c1c23",
             labelcolor="#ddd", edgecolor="#444", ncol=2)
    a.tick_params(colors="#888", labelsize=8)
    for s in a.spines.values():
        s.set_color("#444")

    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"wrote {out_path}")


def make_video(env, policy, steps, out_path, actor=None):
    import imageio.v2 as imageio
    import mujoco
    import torch

    renderer = mujoco.Renderer(env.model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 90, -55, 8.5
    cam.lookat[:] = [0, 0, 0.5]

    frames = []
    o = env.reset()
    e, t = 1.0, 0.0
    for i in range(steps):
        if policy == "scripted":
            a = scripted_action(o, e, t)
        elif policy == "random":
            a = np.random.uniform(env.act_low, env.act_high)
        else:
            with torch.no_grad():
                a, _ = actor(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0),
                             deterministic=True, with_logprob=False)
            a = a.squeeze(0).numpy()
        o, r, d, info = env.step(a)
        e, t = info["energy"], info["temp"]
        if i % 4 == 0:
            renderer.update_scene(env.data, cam)
            frames.append(renderer.render())
        if info["dead"]:
            o = env.reset(); e, t = 1.0, 0.0
    imageio.mimsave(out_path, frames, fps=30)
    print(f"wrote {out_path} ({len(frames)} frames)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, default=None, help="run dir with actor.pt")
    p.add_argument("--policy", type=str, default=None,
                   choices=["scripted", "random", "idle", "learned"])
    p.add_argument("--mode", type=str, default="plot", choices=["plot", "video"])
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--no-reflex", action="store_true")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    policy = args.policy or ("learned" if args.run else "scripted")
    env = Stage2(seed=args.seed, reflexes=not args.no_reflex)

    actor = None
    if policy == "learned":
        import torch
        from sac import Actor
        run = Path(args.run or sorted(glob.glob("runs/*/actor.pt"))[-1]).parent \
            if not args.run else Path(args.run)
        actor = Actor(env.obs_dim, env.nu, env.act_low, env.act_high)
        actor.load_state_dict(torch.load(run / "actor.pt"))
        actor.eval()

    Path("viz").mkdir(exist_ok=True)
    stem = args.out or f"viz/{policy}-{args.steps}"

    if args.mode == "plot":
        log = rollout(env, policy, args.steps, actor)
        alive = args.steps - int(np.sum(log["death"]))
        title = (f"Fen stage 2 — {policy} policy, {args.steps} steps\n"
                 f"deaths {int(np.sum(log['death']))}   "
                 f"eaten {log['ate'].sum():.1f}   "
                 f"mean effort {log['effort'].mean():.2f}")
        make_plot(env, log, f"{stem}.png", title)
    else:
        make_video(env, policy, args.steps, f"{stem}.mp4", actor)


if __name__ == "__main__":
    main()
