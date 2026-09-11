"""
Train the transformer AS the organism.

Recurrent SAC over stored sequences. The transformer core produces a symbol each
cycle from its own context; the action head reads only that symbol; the critic
scores (symbol, action).

Three things that are easy to get wrong here and all of which I got wrong once:

  Sequences, not transitions. A flat replay buffer of (s, a, r, s') tuples cannot
  train a context model — it discards the history attention needs. Chunks are
  sampled contiguous and never across a death.

  Burn-in. Context rebuilt from zeros is systematically wrong for its first
  steps, so the early positions of each chunk contribute no loss.

  The auxiliary prediction loss stays. Reward here is sparse and nearly flat;
  prediction error is dense and free, and it is what made the codebook carve the
  world at real joints in the first place. Training on reward alone collapses the
  codebook to a handful of symbols within a few thousand updates.

    python3 train_tfm_policy.py --steps 300000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "stage2"))
from env import Stage2  # noqa: E402

from transformer_policy import SequenceReplay, TransformerActor  # noqa: E402


class Critic(nn.Module):
    """Twin Q over (symbol, action)."""

    def __init__(self, code_dim, act_dim, hidden=256):
        super().__init__()
        mk = lambda: nn.Sequential(
            nn.Linear(code_dim + act_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.q1, self.q2 = mk(), mk()

    def forward(self, code, act):
        x = torch.cat([code, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=8_000)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--burn-in", type=int, default=8)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--update-every", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--aux", type=float, default=1.0, help="weight on prediction loss")
    p.add_argument("--report-every", type=int, default=50_000)
    p.add_argument("--checkpoints", type=int, default=5)
    p.add_argument("--tag", type=str, default="tfmpol")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = Stage2(seed=args.seed)

    actor = TransformerActor(env.obs_dim, env.nu, env.act_low, env.act_high)
    critic = Critic(actor.core.vq.dim, env.nu)
    critic_t = Critic(actor.core.vq.dim, env.nu)
    critic_t.load_state_dict(critic.state_dict())
    for q in critic_t.parameters():
        q.requires_grad_(False)

    opt_a = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr)
    log_alpha = torch.zeros(1, requires_grad=True)
    opt_al = torch.optim.Adam([log_alpha], lr=args.lr)
    target_entropy = -float(env.nu)

    buf = SequenceReplay(300_000, env.obs_dim, env.nu, seed=args.seed)
    out = Path("runs") / f"{args.tag}-s{args.seed}-{time.strftime('%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)

    obs = env.reset()
    h = actor.core.init_state(1)
    prev = np.zeros(env.nu, dtype=np.float32)
    lifespans, since, deaths = [], 0, 0
    causes = {"neglect": 0, "hazard": 0}
    hist, t0 = [], time.time()
    norm_done = False

    for step in range(args.steps):
        if step < args.warmup:
            a = rng.uniform(env.act_low, env.act_high).astype(np.float32)
            with torch.no_grad():
                _, _, h, _ = actor.core.step(
                    torch.as_tensor(obs).unsqueeze(0),
                    torch.as_tensor(prev).unsqueeze(0), h)
        else:
            with torch.no_grad():
                av, _, h, _ = actor.step(torch.as_tensor(obs).unsqueeze(0),
                                         torch.as_tensor(prev).unsqueeze(0), h)
            a = av.squeeze(0).numpy()

        nxt, r, done, info = env.step(a)
        executed = info.get("executed_action", a).astype(np.float32)
        buf.add(obs, prev, executed, r, nxt, info.get("dead", False))

        since += 1
        if info.get("dead"):
            lifespans.append(since); since = 0; deaths += 1
            c = info.get("death_cause")
            if c in causes:
                causes[c] += 1
            obs = env.reset(); h = actor.core.init_state(1)
            prev = np.zeros(env.nu, dtype=np.float32)
        else:
            obs, prev = nxt, executed

        if step == args.warmup and not norm_done:
            actor.core.fit_normalisation(buf.obs[: buf.n], buf.delta[: buf.n])
            norm_done = True

        if step >= args.warmup and step % args.update_every == 0:
            batch = buf.sample(args.batch, args.seq_len)
            if batch is None:
                continue
            bo, bp, ba, br, bd, bdead = batch
            code, _, pred, vq_loss = actor.sequence(bo, bp, burn_in=args.burn_in)

            # auxiliary prediction — dense signal where reward is flat
            tgt = (bd[:, args.burn_in:] - actor.core.delta_mean) / actor.core.delta_std
            aux = F.huber_loss(pred, tgt)

            cur, nxt_c = code[:, :-1], code[:, 1:].detach()
            rew, dead_ = br[:, args.burn_in:-1], bdead[:, args.burn_in:-1]
            act_ = ba[:, args.burn_in:-1]
            alpha = log_alpha.exp().detach()

            with torch.no_grad():
                a2, lp2 = actor.act_from_code(nxt_c)
                q1t, q2t = critic_t(nxt_c, a2)
                target = rew + args.gamma * (1 - dead_) * (torch.min(q1t, q2t) - alpha * lp2)

            q1, q2 = critic(cur.detach(), act_)
            closs = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            opt_c.zero_grad(); closs.backward(); opt_c.step()

            for q in critic.parameters():
                q.requires_grad_(False)
            pi, lp = actor.act_from_code(cur)
            q1p, q2p = critic(cur, pi)
            aloss = (alpha * lp - torch.min(q1p, q2p)).mean() + args.aux * aux + vq_loss
            opt_a.zero_grad(); aloss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt_a.step()
            for q in critic.parameters():
                q.requires_grad_(True)

            al = -(log_alpha * (lp.detach() + target_entropy)).mean()
            opt_al.zero_grad(); al.backward(); opt_al.step()

            with torch.no_grad():
                for q, qt in zip(critic.parameters(), critic_t.parameters()):
                    qt.mul_(1 - args.tau).add_(args.tau * q)

        if args.checkpoints and step > 0 and step % max(args.steps // args.checkpoints, 1) == 0:
            torch.save(actor.state_dict(), out / f"actor_{step}.pt")

        if step > 0 and step % args.report_every == 0:
            s = actor.core.vq.stats()
            m = {"step": step, "deaths": deaths,
                 "mean_cycles": float(np.mean(lifespans)) / 1000 if lifespans else step / 1000,
                 "median_cycles": float(np.median(lifespans)) / 1000 if lifespans else step / 1000,
                 "live_codes": s["live_codes"], "perplexity": round(s["perplexity"], 1),
                 "neglect": causes["neglect"], "hazard": causes["hazard"],
                 "elapsed": round(time.time() - t0)}
            hist.append(m)
            print(f"  {step:>7}  cycles {m['mean_cycles']:6.2f} (med {m['median_cycles']:5.2f})  "
                  f"deaths {deaths:>4} [N{causes['neglect']}/H{causes['hazard']}]  "
                  f"codes {s['live_codes']:>3} perp {m['perplexity']:>5.1f}  "
                  f"({m['elapsed']}s)", flush=True)

    final = {"steps": args.steps, "seed": args.seed, "deaths": deaths,
             "mean_cycles": float(np.mean(lifespans)) / 1000 if lifespans else args.steps / 1000,
             "median_cycles": float(np.median(lifespans)) / 1000 if lifespans else args.steps / 1000,
             "all_lifespans": [int(x) for x in lifespans],
             "causes": causes, "vq": actor.core.vq.stats(), "history": hist}
    (out / "final.json").write_text(json.dumps(final, indent=2))
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    torch.save(actor.state_dict(), out / "actor_final.pt")

    print("\n" + "=" * 60)
    print(f"transformer-as-organism: {final['mean_cycles']:.2f} cycles "
          f"(median {final['median_cycles']:.2f}), {deaths} deaths")
    print(f"  codebook: {final['vq']['live_codes']} live, "
          f"perplexity {final['vq']['perplexity']:.1f}")
    print(f"  reference: competent hand-written ~7.0 cycles, idle ~2.3")
    print(f"  {out}")


if __name__ == "__main__":
    main()
