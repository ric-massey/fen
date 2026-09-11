"""
Re-probe a trained Stage 3 representation without retraining it.

The gate asks whether the recurrent/attentional context earns its place. The
probe it was asked with cannot answer that, for a structural reason: every
factor in the table is recoverable from the current frame. Position is in the
observation; so is the thermal axis, and `_ambient` is their dot product, so
even "never sensed" ambient temperature is one multiplication away. A model with
no memory at all should score well on all of it, and does — which is why the
context gain sits in the noise regardless of whether context works.

What memory is *for* is the part of the past the present does not reveal. So
probe the same codes against the ground truth at t-k:

    MI(code_t ; factor_{t-k})

At k=0 this is the original measurement. As k grows, a memoryless encoder can
only score through whatever the present still correlates with, while a model
carrying state can do better. The context gain as a function of k is the actual
question, and its slope is the answer.

Lags are capped at the burn-in, so a lagged index never reaches back past the
start of a death-free window.

    python3 probe.py --runs "runs/v1seeds-transformer-s*"
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import mutual_info_score

from collect import sequences
from train_repr import discretise, load
from transformer_core import NoAttention, TransformerWorldModel
from world_model import NoRecurrence, WorldModel

HERE = Path(__file__).parent
CLASSES = {"attention": TransformerWorldModel, "no-attention": NoAttention,
           "recurrent": WorldModel, "no-recurrence": NoRecurrence,
           "untrained": None}


def build(label: str, core: str, obs_dim: int, act_dim: int, n_codes: int):
    cls = CLASSES[label]
    if cls is None:
        cls = TransformerWorldModel if core == "transformer" else WorldModel
    return cls(obs_dim, act_dim, n_codes=n_codes)


def mi_at_lags(model, data, rng, lags, n_seq, seq_len, burn_in,
               stride=1) -> dict:
    """Normalised MI between the code at t and each factor at t-k."""
    model.eval()
    obs, act, _, idx = sequences(data, seq_len, n_seq, rng, stride=stride,
                                 margin=max(lags))
    with torch.no_grad():
        _, codes, _, _ = model(torch.tensor(obs), torch.tensor(act),
                               burn_in=burn_in)
    codes = codes.numpy().reshape(-1)
    live = idx[:, burn_in:]

    out = {}
    for k in lags:
        keep = (live - k).reshape(-1)
        per = {}
        for name, vals in data["truth"].items():
            f = discretise(vals[keep])
            h = mutual_info_score(f, f)
            per[name] = float(mutual_info_score(f, codes) / h) if h > 1e-9 else float("nan")
        out[k] = per
    return out


def ci(x: np.ndarray) -> tuple:
    n = len(x)
    m, sd = float(np.mean(x)), float(np.std(x, ddof=1)) if n > 1 else 0.0
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan")
    return m, sd, m - half, m + half


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=str, required=True,
                   help="glob of run directories written by train_repr.py")
    p.add_argument("--data", type=str, default="data/traj.npz")
    p.add_argument("--lags", type=str, default="0,4,8,16")
    p.add_argument("--burn-in", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--n-seq", type=int, default=64)
    p.add_argument("--core", type=str, default="transformer")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--tag", type=str, default="probe")
    a = p.parse_args()

    lags = [int(x) for x in a.lags.split(",")]
    # A lag is safe because sequences() reserves a death-free margin before the
    # window, not because it is shorter than the burn-in.

    dirs = sorted(d for d in glob.glob(a.runs) if Path(d).is_dir())
    if not dirs:
        raise SystemExit(f"no run directories match {a.runs}")
    data = load(a.data)
    obs_dim, act_dim = data["obs"].shape[1], data["act"].shape[1]
    print(f"=== lagged probe, {len(dirs)} runs, lags {lags} ===")

    per_run = []
    for d in dirs:
        res = json.loads((Path(d) / "results.json").read_text())
        labels = list(res.keys())
        n_codes = res[labels[0]]["n_codes"]
        rng = np.random.default_rng(0)          # same sequences for every model
        got = {}
        for label in labels:
            m = build(label, a.core, obs_dim, act_dim, n_codes)
            m.load_state_dict(torch.load(Path(d) / f"{label}.pt"))
            m.fit_normalisation(data["obs"], data["delta"])
            got[label] = mi_at_lags(m, data, np.random.default_rng(0), lags,
                                    a.n_seq, a.seq_len, a.burn_in,
                                    stride=a.stride)
        per_run.append({"dir": d, "labels": labels, "mi": got})
        print(f"  probed {Path(d).name}", flush=True)

    labels = per_run[0]["labels"]
    with_ctx, without_ctx = [l for l in labels if l != "untrained"]
    factors = list(per_run[0]["mi"][with_ctx][lags[0]].keys())

    print("\n" + "=" * 78)
    print(f"context gain = MI({with_ctx}) - MI({without_ctx}), by lag")
    print(f"{'lag':>5} {'gain':>9} {'sd':>8} {'95% CI':>20} {'seeds +':>9} "
          f"{'vs untrained':>13}")
    by_lag = {}
    for k in lags:
        g = np.array([np.mean([r["mi"][with_ctx][k][f] - r["mi"][without_ctx][k][f]
                               for f in factors]) for r in per_run])
        u = np.array([np.mean([r["mi"][with_ctx][k][f] - r["mi"]["untrained"][k][f]
                               for f in factors]) for r in per_run])
        m, sd, lo, hi = ci(g)
        by_lag[k] = {"gain": g.tolist(), "mean": m, "sd": sd,
                     "ci_lo": lo, "ci_hi": hi, "vs_untrained": float(u.mean())}
        print(f"{k:>5} {m:+9.4f} {sd:8.4f} [{lo:+8.4f},{hi:+8.4f}] "
              f"{int(np.sum(g > 0)):>4}/{len(g)} {u.mean():+13.4f}")

    print(f"\nper-factor gain at the longest lag ({lags[-1]}):")
    print(f"{'factor':14} {'gain':>9} {'sd':>8} {'seeds +':>9}")
    for f in factors:
        v = np.array([r["mi"][with_ctx][lags[-1]][f] - r["mi"][without_ctx][lags[-1]][f]
                      for r in per_run])
        m, sd, _, _ = ci(v)
        print(f"{f:14} {m:+9.4f} {sd:8.4f} {int(np.sum(v > 0)):>4}/{len(v)}")

    # The slope is the finding: does context matter MORE as the past recedes?
    slope = by_lag[lags[-1]]["mean"] - by_lag[lags[0]]["mean"]
    print(f"\ngain at lag {lags[0]}: {by_lag[lags[0]]['mean']:+.4f}   "
          f"gain at lag {lags[-1]}: {by_lag[lags[-1]]['mean']:+.4f}   "
          f"slope {slope:+.4f}")
    if by_lag[lags[-1]]["ci_lo"] > 0 and slope > 0:
        print("  context carries the past -> the recurrence is doing work")
    elif by_lag[lags[-1]]["ci_hi"] < 0:
        print("  context is worse than its ablation even on the past")
    else:
        print("  still indistinguishable from zero at every lag")

    out = HERE / "runs" / f"{a.tag}-summary.json"
    out.write_text(json.dumps({"runs": dirs, "lags": lags,
                               "with_context": with_ctx,
                               "without_context": without_ctx,
                               "by_lag": by_lag}, indent=2))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
