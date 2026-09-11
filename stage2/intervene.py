"""
Causal test of drive-contingency, replacing the correlation gate.

Why the old gate was invalid. Temperature is a function of x-position, so
correlating temperature against x-velocity is largely correlating x against its
own derivative with a phase lag. A fixed sine wave in x — a policy with no
observation input at all — scores food +0.24 / temp -0.28, clearing a gate whose
thresholds were +-0.05. The metric was measuring oscillation, not regulation.

What this does instead. Hold the body's position and configuration FIXED, clamp
the internal state to two different values, and ask whether the policy's output
changes. Position is held constant by construction, so no amount of clever
spatial behaviour can score. A state-blind policy scores exactly zero.

Two quantities per drive:

  sensitivity  ||a(state_low) - a(state_high)|| averaged over positions.
               Does the action depend on internal state at all?

  directedness cosine between the base-velocity change and the direction the
               drive *should* push. Hungry should shift motion toward food;
               overheating should shift it toward the cool half. This is the
               one that matters — sensitivity alone only shows the policy reads
               the channel, not that it reads it correctly.

Both are reported with fog on (what the agent actually has to work with) and
fog off (the ceiling if its interoception were perfect). The gap between them
is itself informative: it is the cost of imperfect self-knowledge.

    python3 intervene.py --run runs/<dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from env import Stage2
from sac import Actor


def _obs_with_state(env, energy: float, temp: float, health: float) -> np.ndarray:
    env.drives.energy = energy
    env.drives.temp = temp
    env.health.health = health
    return env.obs()


def probe(actor, env, n_pos: int = 300, seed: int = 0, fog: float = 1.0,
          settle: int = 6) -> dict:
    rng = np.random.default_rng(seed)
    env._base_fog = fog

    d_energy, d_temp = [], []
    dir_energy, dir_temp = [], []

    for _ in range(n_pos):
        env.reset()
        # Random but FIXED body pose for this probe pair.
        env.data.qpos[0] = rng.uniform(-2.4, 2.4)
        env.data.qpos[1] = rng.uniform(-2.4, 2.4)
        env.data.qpos[2:6] = rng.uniform(-0.6, 0.6, 4)
        import mujoco
        mujoco.mj_forward(env.model, env.data)

        base = env._base_xy()
        smell = env._smell()

        def action_for(energy, temp, health=1.0):
            # Average over several draws: fog is stochastic and lagged, so a
            # single sample confounds state-sensitivity with sensing noise.
            acc = []
            for _ in range(settle):
                o = _obs_with_state(env, energy, temp, health)
                with torch.no_grad():
                    a, _ = actor(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0),
                                 deterministic=True, with_logprob=False)
                acc.append(a.squeeze(0).numpy())
            return np.mean(acc, axis=0)

        # ── energy intervention (temperature held neutral) ──
        a_hungry = action_for(0.20, 0.0)
        a_full = action_for(0.95, 0.0)
        delta = a_hungry - a_full
        d_energy.append(np.linalg.norm(delta))
        if np.linalg.norm(smell[:2]) > 1e-6 and np.linalg.norm(delta[:2]) > 1e-9:
            dir_energy.append(float(np.dot(delta[:2] / np.linalg.norm(delta[:2]),
                                           smell[:2] / np.linalg.norm(smell[:2]))))

        # ── temperature intervention (energy held comfortable) ──
        # The thermal gradient the agent sees is IDENTICAL in both conditions.
        # Only its internal temperature differs, so any change in action is
        # necessarily driven by reading its own state. No spatial rule and no
        # oscillation can produce this.
        a_hot = action_for(0.70, 0.60)
        a_cool = action_for(0.70, -0.60)
        delta_t = a_hot - a_cool
        d_temp.append(np.linalg.norm(delta_t))
        cool_dir = -env.thermal_axis
        if np.linalg.norm(delta_t[:2]) > 1e-9:
            dir_temp.append(float(np.dot(delta_t[:2] / np.linalg.norm(delta_t[:2]),
                                         cool_dir / np.linalg.norm(cool_dir))))

    return {
        "sensitivity_energy": float(np.mean(d_energy)),
        "sensitivity_temp": float(np.mean(d_temp)),
        "directedness_energy": float(np.mean(dir_energy)) if dir_energy else float("nan"),
        "directedness_temp": float(np.mean(dir_temp)) if dir_temp else float("nan"),
        "n_positions": n_pos,
    }


class BlindOscillator:
    """Control: ignores observation entirely. Must score ~0 on everything."""

    def __init__(self, period=600):
        self.t = 0
        self.period = period

    def __call__(self, obs, deterministic=True, with_logprob=False):
        self.t += 1
        a = np.zeros(6, dtype=np.float32)
        a[0] = 1.2 * np.sin(2 * np.pi * self.t / self.period)
        return torch.as_tensor(a).unsqueeze(0), None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, required=True)
    p.add_argument("--positions", type=int, default=300)
    p.add_argument("--seed", type=int, default=5)
    args = p.parse_args()

    run = Path(args.run)
    cfg = json.loads((run / "args.json").read_text())
    env = Stage2(seed=args.seed, reflexes=False)   # probing the POLICY, not the body
    actor = Actor(env.obs_dim, env.nu, env.act_low, env.act_high)
    actor.load_state_dict(torch.load(run / "actor.pt"))
    actor.eval()

    print(f"{'':22} {'sens(E)':>9} {'sens(T)':>9} {'dir(E)':>9} {'dir(T)':>9}")
    print("-" * 62)

    out = {}
    for label, fog in [("trained, fog on", 1.0), ("trained, fog off", 0.0)]:
        m = probe(actor, env, args.positions, args.seed, fog)
        out[label] = m
        print(f"{label:22} {m['sensitivity_energy']:9.4f} {m['sensitivity_temp']:9.4f} "
              f"{m['directedness_energy']:9.3f} {m['directedness_temp']:9.3f}")

    m = probe(BlindOscillator(), env, args.positions, args.seed, 1.0)
    out["blind oscillator"] = m
    print(f"{'blind oscillator':22} {m['sensitivity_energy']:9.4f} "
          f"{m['sensitivity_temp']:9.4f} {m['directedness_energy']:9.3f} "
          f"{m['directedness_temp']:9.3f}")

    (run / "intervention.json").write_text(json.dumps(out, indent=2))
    print("\nsensitivity: does the action depend on internal state at all")
    print("directedness: does it change in the RIGHT direction (+1 correct, 0 none, -1 wrong)")
    print("The blind control scores 0 by construction; anything above it is real.")


if __name__ == "__main__":
    main()
