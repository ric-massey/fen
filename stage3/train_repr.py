"""
Train the Stage 3 representation self-supervised, and score its gate.

Gate (FEN.md, Stage 3):
  - a substantial fraction of the codebook in active use
  - mutual information between codes and ground-truth world structure well
    above chance
  - a measurable behavioural cost to freezing the recurrent state

The third is scored here as a prediction cost rather than behaviour, since the
policy is not wired in yet: does removing recurrence hurt? The `NoRecurrence`
ablation is the control, identical in capacity and trained identically.

    python3 train_repr.py --steps 8000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mutual_info_score

from collect import sequences
from transformer_core import NoAttention, TransformerWorldModel
from world_model import NoRecurrence, WorldModel


def load(path: str) -> dict:
    z = np.load(path)
    truth = {k[6:]: z[k] for k in z.files if k.startswith("truth_")}
    return {"obs": z["obs"], "act": z["act"], "delta": z["delta"],
            "done": z["done"], "truth": truth}


def discretise(v: np.ndarray, bins: int = 8) -> np.ndarray:
    if v.dtype.kind in "iub" or len(np.unique(v)) <= bins:
        return v.astype(int)
    edges = np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1])
    return np.digitize(v, edges)


def code_structure_mi(model, data, rng, n_seq=64, seq_len=64, burn_in=16,
                      stride=1) -> dict:
    """Normalised MI between emitted codes and each ground-truth factor.

    Normalised by the factor's own entropy, so the number reads as 'fraction of
    this factor's information the codebook carries'. Chance is ~0.
    """
    model.eval()
    obs, act, delta, idx = sequences(data, seq_len, n_seq, rng, stride=stride)
    with torch.no_grad():
        _, codes, _, _ = model(torch.tensor(obs), torch.tensor(act), burn_in=burn_in)
    codes = codes.numpy().reshape(-1)
    keep = idx[:, burn_in:].reshape(-1)

    out = {}
    for name, vals in data["truth"].items():
        f = discretise(vals[keep])
        h = mutual_info_score(f, f)
        out[name] = float(mutual_info_score(f, codes) / h) if h > 1e-9 else float("nan")
    model.train()
    return out


def train(model, data, steps, seq_len, burn_in, batch, lr, seed, label, quiet,
          stride=1):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model.fit_normalisation(data["obs"], data["delta"])
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    t0 = time.time()
    for step in range(steps):
        obs, act, delta, _ = sequences(data, seq_len, batch, rng, stride=stride)
        loss, parts, _ = model.loss(torch.tensor(obs), torch.tensor(act),
                                    torch.tensor(delta), burn_in=burn_in)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if not quiet and step > 0 and step % max(steps // 6, 1) == 0:
            s = model.vq.stats()
            print(f"  [{label}] {step:>6}  pred {parts['pred']:.4f}  "
                  f"vq {parts['vq']:.4f}  live {s['live_codes']}/{s['n_codes']}  "
                  f"perplexity {s['perplexity']:.1f}  ({time.time()-t0:.0f}s)", flush=True)
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/traj.npz")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--burn-in", type=int, default=16)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--codes", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", type=str, default="dev")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--core", type=str, default="gru", choices=["gru", "transformer"])
    # Stride buys temporal horizon at constant compute. At stride 1 every truth
    # factor autocorrelates >0.86 across the burn-in and >0.53 across the whole
    # window, so there is nothing for a recurrent state to carry and the
    # context ablation is measuring noise. See collect.sequences.
    p.add_argument("--stride", type=int, default=1)
    a = p.parse_args()

    data = load(a.data)
    obs_dim, act_dim = data["obs"].shape[1], data["act"].shape[1]
    rng = np.random.default_rng(a.seed + 999)

    out = Path("runs") / f"{a.tag}-s{a.seed}-{time.strftime('%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    # "untrained" is the control that decides whether MI means anything: a
    # random encoder with the same 128 codes will still partition the input
    # space, and any factor correlated with position will show up. Real
    # structure is the margin OVER this, not the raw number.
    if a.core == "transformer":
        conditions = [("attention", TransformerWorldModel, True),
                      ("no-attention", NoAttention, True),
                      ("untrained", TransformerWorldModel, False)]
    else:
        conditions = [("recurrent", WorldModel, True),
                      ("no-recurrence", NoRecurrence, True),
                      ("untrained", WorldModel, False)]
    for label, cls, do_train in conditions:
        print(f"\n=== {label} ===")
        # Seed BEFORE construction. train() seeds too, but it runs after the
        # model exists, so without this the first condition's weights come from
        # an unseeded RNG while later ones inherit the previous condition's —
        # the with-context arm was the only one initialised randomly, and the
        # ablation it is compared against was not.
        torch.manual_seed(a.seed)
        m = cls(obs_dim, act_dim, n_codes=a.codes)
        if do_train:
            m = train(m, data, a.steps, a.seq_len, a.burn_in, a.batch, a.lr,
                      a.seed, label, a.quiet, stride=a.stride)
        else:
            m.fit_normalisation(data["obs"], data["delta"])

        # held-out prediction loss.
        # Re-seed per condition so every arm is scored on the SAME held-out
        # sequences. Sharing one advancing rng gave each condition a different
        # sample, so the paired difference carried the sampling noise of two
        # independent draws — measured at stride 8, that alone was the
        # difference between sd 0.045 and sd 0.008 on the gate metric.
        rng = np.random.default_rng(a.seed + 999)
        m.eval()
        obs, act, delta, _ = sequences(data, a.seq_len, 128, rng, stride=a.stride)
        with torch.no_grad():
            loss, parts, _ = m.loss(torch.tensor(obs), torch.tensor(act),
                                    torch.tensor(delta), burn_in=a.burn_in)
        stats = m.vq.stats()
        mi = code_structure_mi(m, data, rng, burn_in=a.burn_in, stride=a.stride)

        results[label] = {"pred_loss": parts["pred"], **stats, "mi": mi}
        torch.save(m.state_dict(), out / f"{label}.pt")

    (out / "results.json").write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 72)
    print(f"{'':16} {'pred':>8} {'live':>7} {'perplex':>9}")
    for k, v in results.items():
        print(f"{k:16} {v['pred_loss']:8.4f} {v['live_codes']:4d}/{v['n_codes']:<3} "
              f"{v['perplexity']:9.1f}")

    print(f"\nNormalised MI(code ; ground truth) — fraction of each factor captured")
    keys = [k for k in results if k != "untrained"]
    factors = list(results[keys[0]]["mi"].keys())
    print(f"{'factor':12} {'recurrent':>11} {'no-recur':>11} {'untrained':>11} "
          f"{'gain vs untr':>13}")
    for f in factors:
        r, n, u = (results[keys[0]]["mi"][f], results[keys[1]]["mi"][f],
                   results["untrained"]["mi"][f])
        print(f"{f:12} {r:11.3f} {n:11.3f} {u:11.3f} {r-u:+13.3f}")

    # Judge on representation, not prediction. Prediction was the training
    # signal, not the objective — the codebook is what the policy consumes.
    gain_pred = results[keys[1]]["pred_loss"] - results[keys[0]]["pred_loss"]
    mi_gain = np.mean([results[keys[0]]["mi"][f] - results[keys[1]]["mi"][f]
                       for f in factors])
    print(f"\ncontext gain on prediction: {gain_pred:+.4f}")
    print(f"context gain on mean MI:    {mi_gain:+.4f}  "
          f"({'context earns its place' if mi_gain > 0.02 else 'CONTEXT IS DECORATION'})")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
