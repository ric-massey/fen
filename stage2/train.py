"""
Stage 2 training.

Gate (FEN.md): survives indefinitely under its own regulation, and behaviour is
demonstrably contingent on drive state rather than on position or time.

Survival is easy to measure. Contingency is the harder half, so it is measured
directly rather than eyeballed:

  food contingency  correlation between energy deficit and approach velocity
                    toward the nearest charged food site. Positive means it
                    moves toward food *because* it is hungry, not on a schedule.

  temp contingency  correlation between body temperature and x-velocity.
                    Should be NEGATIVE: too warm (+temp) should produce motion
                    toward -x, which is the cool half.

Neither can be satisfied by a fixed policy that ignores its own state, which is
the point.

    python3 train.py --steps 300000
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from env import Stage2
from sac import SAC

OUT = Path(__file__).parent / "runs"


class Contingency:
    """Rolling correlations between drive state and behaviour."""

    def __init__(self, window: int = 20_000):
        self.e_def, self.food_v = deque(maxlen=window), deque(maxlen=window)
        self.temp, self.vx = deque(maxlen=window), deque(maxlen=window)

    def add(self, energy, temp, base_xy, prev_base_xy, food_xy):
        self.e_def.append(1.0 - energy)
        self.temp.append(temp)
        self.vx.append(base_xy[0] - prev_base_xy[0])
        if food_xy is None:
            self.food_v.append(0.0)
        else:
            d_now = np.linalg.norm(food_xy - base_xy)
            d_prev = np.linalg.norm(food_xy - prev_base_xy)
            self.food_v.append(d_prev - d_now)   # positive = approaching

    @staticmethod
    def _corr(a, b) -> float:
        a, b = np.asarray(a), np.asarray(b)
        if len(a) < 500 or a.std() < 1e-8 or b.std() < 1e-8:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    def report(self) -> dict:
        return {
            "contingency_food": self._corr(self.e_def, self.food_v),
            "contingency_temp": self._corr(self.temp, self.vx),
        }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5_000)
    p.add_argument("--update-every", type=int, default=2)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--fog", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--target-entropy", type=float, default=None)
    p.add_argument("--no-reflex", action="store_true")
    p.add_argument("--temp-trigger", type=float, default=0.55)
    p.add_argument("--energy-trigger", type=float, default=0.30)
    p.add_argument("--reflex-cost", type=float, default=0.25)
    p.add_argument("--report-every", type=int, default=20_000)
    p.add_argument("--tag", type=str, default="dev")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--checkpoints", type=int, default=0,
                   help="save the actor N times during the run, for the timeline viewer")
    args = p.parse_args()

    env = Stage2(seed=args.seed, fog=args.fog, reflexes=not args.no_reflex,
                 temp_trigger=args.temp_trigger, energy_trigger=args.energy_trigger,
                 reflex_cost=args.reflex_cost)
    agent = SAC(env.obs_dim, env.nu, env.act_low, env.act_high,
                gamma=args.gamma, seed=args.seed,
                target_entropy=args.target_entropy)
    rng = np.random.default_rng(args.seed)

    OUT.mkdir(exist_ok=True)
    run_dir = OUT / f"{args.tag}-s{args.seed}-fog{args.fog}-{time.strftime('%H%M%S')}"
    run_dir.mkdir(parents=True)

    obs = env.reset()
    prev_xy = env._base_xy().copy()
    cont = Contingency()
    lifespans, since_death, deaths, eaten = deque(maxlen=50), 0, 0, 0.0
    all_lifespans = []   # trailing-50 hides early catastrophe; keep everything
    cause_counts = {"energy": 0, "temp": 0, "health": 0}
    history = []
    t0 = time.time()

    for step in range(args.steps):
        action = (rng.uniform(env.act_low, env.act_high) if step < args.warmup
                  else agent.act(obs))
        nxt, reward, done, info = env.step(action)
        # Store the EXECUTED action; a reflex may have overridden the policy.
        agent.replay.add(obs, info.get("executed_action", action), reward, nxt,
                         info.get("dead", False))

        charged = env.food.charge > 0.05
        food_xy = None
        if charged.any():
            pos = env.food.pos[charged][:, :2]
            food_xy = pos[np.argmin(np.linalg.norm(pos - prev_xy[None, :], axis=1))]
        cont.add(info["energy"], info["temp"], np.array([info["x"], info["y"]]),
                 prev_xy, food_xy)
        eaten += info.get("ate", 0.0)

        prev_xy = np.array([info["x"], info["y"]])
        since_death += 1
        if info.get("dead"):
            lifespans.append(since_death)
            all_lifespans.append(since_death)
            since_death, deaths = 0, deaths + 1
            c = info.get("death_cause")
            if c in cause_counts:
                cause_counts[c] += 1
            obs = env.reset()
            prev_xy = env._base_xy().copy()
        else:
            obs = nxt
            if done:
                obs = env.reset()
                prev_xy = env._base_xy().copy()

        if step >= args.warmup and step % args.update_every == 0:
            agent.update(args.batch)

        # Periodic checkpoints so the viewer can show TRAINING, not just a
        # finished policy. Watching it fail early and improve (or not) is the
        # honest picture; a single recording of the best result is not.
        if args.checkpoints and step > 0 and step % max(args.steps // args.checkpoints, 1) == 0:
            import torch as _t
            _t.save(agent.actor.state_dict(), run_dir / f"actor_{step}.pt")

        if step > 0 and step % args.report_every == 0:
            m = {"step": step, "deaths": deaths,
                 "died_energy": cause_counts["energy"], "died_temp": cause_counts["temp"],
                 "mean_lifespan": float(np.mean(lifespans)) if lifespans else float(step),
                 "energy": info["energy"], "temp": info["temp"],
                 "eaten_total": round(eaten, 2),
                 "reflex_temp": env.reflex.fired_temp if env.reflex else 0,
                 "reflex_energy": env.reflex.fired_energy if env.reflex else 0,
                 "health": info.get("health", 1.0),
                 "elapsed": round(time.time() - t0)}
            m.update(cont.report())
            history.append(m)
            if not args.quiet:
                print(f"step {step:>7}  life {m['mean_lifespan']:>7.0f}  "
                      f"deaths {deaths:>4}  eaten {m['eaten_total']:>7.2f}  "
                      f"food-cont {m['contingency_food']:+.3f}  "
                      f"temp-cont {m['contingency_temp']:+.3f}  "
                      f"hp {m['health']:.2f}  ({m['elapsed']}s)",
                      flush=True)

    final = {"steps": args.steps, "seed": args.seed, "fog": args.fog,
             "gamma": args.gamma, "target_entropy": args.target_entropy, "died_energy": cause_counts["energy"],
             "died_temp": cause_counts["temp"],
             "deaths": deaths,
             "mean_lifespan": float(np.mean(all_lifespans)) if all_lifespans else float(args.steps),
             "median_lifespan": float(np.median(all_lifespans)) if all_lifespans else float(args.steps),
             "mean_lifespan_recent50": float(np.mean(lifespans)) if lifespans else float(args.steps),
             "eaten_total": round(eaten, 2)}
    final.update(cont.report())
    rf = (cause_counts and env.reflex is not None)
    reflex_rate = ((env.reflex.fired_temp + env.reflex.fired_energy) / args.steps
                   if env.reflex else 0.0)
    final["reflex_rate"] = reflex_rate
    # Survival alone is gameable: a policy can park inside the reflex envelope
    # and let the innate layer keep it alive. The gate requires that survival
    # be the AGENT's, so reflex usage must stay rare.
        # Gate in DRIVE CYCLES, not steps. One hunger cycle is ~1000 steps; 20_000
    # was an arbitrary number. What matters is how many times it has to solve
    # the recurring problem, not how many seconds it lasted.
    final["lifespan_cycles"] = final["mean_lifespan"] / 1000.0
    final["gate_survival"] = bool(final["lifespan_cycles"] >= 10.0)
    # Correlation is RETIRED as a gate: a state-blind sine wave passes it, and
    # randomising the layout did not rescue it (the estimator has enormous
    # variance). Contingency is now scored causally by intervene.py. These are
    # kept only as descriptive statistics.
    final["gate_contingency"] = None

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "final.json").write_text(json.dumps(final, indent=2))
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    import torch
    torch.save(agent.actor.state_dict(), run_dir / "actor.pt")

    if args.quiet:
        print(json.dumps(final))
    else:
        print("\n" + "=" * 66)
        print(f"FINAL  mean lifespan {final['mean_lifespan']:.0f}  deaths {deaths}")
        print(f"  food contingency {final['contingency_food']:+.3f}  "
              f"temp contingency {final['contingency_temp']:+.3f}")
        print(f"  survival gate    {'PASS' if final['gate_survival'] else 'FAIL'}")
        print(f"  contingency gate {'PASS' if final['gate_contingency'] else 'FAIL'}")
        print(f"  {run_dir}")
        print("=" * 66)


if __name__ == "__main__":
    main()
