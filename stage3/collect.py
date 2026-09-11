"""
Collect trajectories from the Stage 2 world, with ground truth recorded.

The ground-truth factors are never model inputs. They exist so Stage 3's gate is
answerable: do the learned codes carry real world structure, or did they just
partition the input space somewhere convenient? Because the world is built, the
answer is checkable rather than a matter of interpretation.

Behaviour policy is a mix. The scripted policy visits the interesting regions —
food sites, thermal extremes, the shuttle between them — while random action
covers state space the scripted policy never enters. Training only on competent
behaviour gives a codebook that has never seen trouble.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "stage2"))
from env import Stage2  # noqa: E402


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


def collect(steps: int = 200_000, seed: int = 0, p_random: float = 0.35,
            switch_every: int = 400):
    env = Stage2(seed=seed)
    rng = np.random.default_rng(seed)

    obs_l, act_l, delta_l, done_l = [], [], [], []
    truth = {k: [] for k in ("x", "y", "energy", "temp", "health",
                             "near_food", "warm", "reflex", "stress",
                             "thermal_proj")}

    o = env.reset()
    e, t = 1.0, 0.0
    mode_random = False

    for i in range(steps):
        if i % switch_every == 0:
            mode_random = rng.random() < p_random
        a = (rng.uniform(env.act_low, env.act_high) if mode_random
             else scripted(o, e, t))

        nxt, r, d, info = env.step(a)
        obs_l.append(o)
        act_l.append(a.astype(np.float32))
        delta_l.append(nxt - o)
        done_l.append(bool(info["dead"]))

        base = np.array([info["x"], info["y"]])
        dists = np.linalg.norm(env.food.pos[:, :2] - base[None, :], axis=1)
        truth["x"].append(info["x"])
        truth["y"].append(info["y"])
        truth["energy"].append(info["energy"])
        truth["temp"].append(info["temp"])
        truth["health"].append(info["health"])
        truth["near_food"].append(int(np.argmin(dists)) if dists.min() < 0.9 else -1)
        # "warm" must be relative to THIS episode's thermal axis; x > 0 was
        # only correct back when the axis was fixed.
        truth["warm"].append(int(np.dot(base, info["thermal_axis"]) > 0))
        truth["reflex"].append({None: 0, "temp": 1, "energy": 2}[info.get("reflex")])
        truth["stress"].append(info["stress"])
        # signed distance along the thermal axis — the axis-relative
        # position, which is what "where am I thermally" actually means
        truth["thermal_proj"].append(float(np.dot(base, info["thermal_axis"])))

        e, t = info["energy"], info["temp"]
        o = env.reset() if info["dead"] else nxt
        if info["dead"]:
            e, t = 1.0, 0.0

    return {
        "obs": np.array(obs_l, dtype=np.float32),
        "act": np.array(act_l, dtype=np.float32),
        "delta": np.array(delta_l, dtype=np.float32),
        "done": np.array(done_l, dtype=bool),
        "truth": {k: np.array(v) for k, v in truth.items()},
    }


def sequences(data: dict, length: int, batch: int, rng: np.random.Generator,
              stride: int = 1, margin: int = 0):
    """Sample contiguous chunks that do not straddle a death.

    A sequence spanning a reset teaches the recurrent state that the world
    discontinuously jumps, which is a fact about the harness rather than the
    world.

    `stride` samples every nth step instead of every step, so a window of the
    same length covers `stride` times the horizon at the same compute. This
    matters here: measured on the v1 data, every ground-truth factor
    autocorrelates above 0.86 at 16 steps and above 0.53 across a whole 64-step
    window, so a stride-1 window contains almost no change for a recurrent
    state to carry. Stride buys horizon where sequence length costs attention
    quadratically.

    `margin` additionally requires the window to be death-free for that many
    steps BEFORE it starts, so a probe may look back at truth[t - k] for any
    k <= margin without the lagged index landing in a previous life.
    """
    n = len(data["obs"])
    done = data["done"]
    span = (length - 1) * stride + 1
    lo = margin
    hi = n - span - 1
    if hi <= lo:
        raise ValueError(f"data too short for length={length} stride={stride} "
                         f"margin={margin}")
    starts = []
    while len(starts) < batch:
        s = int(rng.integers(lo, hi))
        if not done[s - margin: s + span].any():
            starts.append(s)
    idx = np.array([np.arange(s, s + span, stride) for s in starts])
    return data["obs"][idx], data["act"][idx], data["delta"][idx], idx


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="data/traj.npz")
    a = p.parse_args()

    d = collect(a.steps, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, obs=d["obs"], act=d["act"], delta=d["delta"],
                        done=d["done"], **{f"truth_{k}": v for k, v in d["truth"].items()})
    print(f"wrote {a.out}: {len(d['obs'])} steps, {int(d['done'].sum())} deaths")
    print("ground truth recorded:", ", ".join(d["truth"].keys()))
