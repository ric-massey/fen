"""
Status figure for Stage 3: what the codebook learned, and whether it helps.

Three panels, answering the three questions in order:
  what does the vocabulary encode  (MI by factor, against the untrained floor)
  what does context add            (attention/recurrence gain, per factor)
  does any of it help it live      (policy learning curves)

    python3 dashboard.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, PANEL, FG, DIM = "#15151a", "#1c1c23", "#e8e8ee", "#8a8a99"
C = {"trained": "#5fd06a", "ablated": "#7aa6e0", "untrained": "#6a6a78",
     "raw": "#8a8a99", "+hidden": "#5fd06a", "+code": "#e0873f"}


def _style(ax, title=None):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=DIM, labelsize=8)
    for s in ax.spines.values():
        s.set_color("#3a3a45")
    if title:
        ax.set_title(title, color=FG, fontsize=11, pad=10, loc="left")


def latest(pattern):
    hits = sorted(glob.glob(pattern))
    return json.load(open(hits[-1])) if hits else None


def main() -> None:
    tfm = latest("runs/tfm-*/results.json")
    pol = latest("runs/policy-*/results.json")

    fig = plt.figure(figsize=(15, 5.2), facecolor=BG)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1.1], wspace=0.28)

    # ── 1: what the vocabulary encodes ───────────────────────────────────────
    ax = fig.add_subplot(gs[0])
    _style(ax, "What the symbols encode")
    if tfm:
        factors = list(tfm["attention"]["mi"].keys())
        y = np.arange(len(factors))
        ax.barh(y + 0.22, [tfm["attention"]["mi"][f] for f in factors], 0.34,
                color=C["trained"], label="trained (attention)")
        ax.barh(y - 0.22, [tfm["untrained"]["mi"][f] for f in factors], 0.34,
                color=C["untrained"], label="untrained floor")
        ax.set_yticks(y); ax.set_yticklabels(factors, color=DIM, fontsize=9)
        ax.set_xlabel("normalised MI with ground truth", color=DIM, fontsize=9)
        ax.legend(fontsize=8, facecolor=PANEL, labelcolor=FG, edgecolor="#3a3a45")
        ax.invert_yaxis()

    # ── 2: what context adds ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1])
    _style(ax, "What context adds")
    if tfm:
        gains = [tfm["attention"]["mi"][f] - tfm["no-attention"]["mi"][f]
                 for f in factors]
        cols = ["#5fd06a" if g > 0.03 else "#4a4a58" for g in gains]
        ax.barh(np.arange(len(factors)), gains, 0.6, color=cols)
        ax.axvline(0, color=DIM, lw=0.8)
        ax.set_yticks(np.arange(len(factors)))
        ax.set_yticklabels(factors, color=DIM, fontsize=9)
        ax.set_xlabel("MI gain from attention", color=DIM, fontsize=9)
        ax.invert_yaxis()
        ax.text(0.97, 0.04, "history-dependent factors only",
                transform=ax.transAxes, ha="right", color=DIM, fontsize=8,
                style="italic")

    # ── 3: does it help it live ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[2])
    _style(ax, "Does it help it live")
    if pol:
        for mode, v in pol.items():
            h = v.get("history", [])
            if not h:
                continue
            ax.plot([m["step"] for m in h], [m["all_lifespan"] for m in h],
                    color=C.get(mode, FG), lw=2, marker="o", ms=4,
                    label=f"{mode}  ({v['mean_lifespan_all']:.0f})")
        ax.axhline(1000, color="#e0873f", ls="--", lw=1,
                   label="one hunger cycle")
        ax.set_xlabel("training step", color=DIM, fontsize=9)
        ax.set_ylabel("mean lifespan (all deaths)", color=DIM, fontsize=9)
        ax.legend(fontsize=8, facecolor=PANEL, labelcolor=FG, edgecolor="#3a3a45")

    fig.suptitle("Fen stage 3 — transformer core, learned symbol codebook",
                 color=FG, fontsize=13, x=0.09, ha="left", y=0.99)
    Path("viz").mkdir(exist_ok=True)
    fig.savefig("viz/dashboard.png", dpi=130, facecolor=BG, bbox_inches="tight")
    print("wrote viz/dashboard.png")


if __name__ == "__main__":
    main()
