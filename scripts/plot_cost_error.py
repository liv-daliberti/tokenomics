"""The difficulty x cost scatter: estimation error vs the price of information,
across difficulty levels and partner honesty regimes, for GPT-5.4 and Qwen3-32B.

Three panels — a partner that ALWAYS lies, one that NEVER lies, and a MIXED one
that lies about half the time. In each, every point is one match: x = the price
of information (min trade price), y = the LLM's mean estimation error, colour =
difficulty (instrument offset), marker = model. Faint dashed lines are the
RATIONAL-optimal error a value-maximizing agent would achieve (analytical, from
the game's reward model) — the reference the empirical points are read against.

    python scripts/plot_cost_error.py [--out paper/fig/cost_error.pdf]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.metrics import load_events

# --- analytical rational-optimal reference (probe_trust economics) -----------
TAU, BUCKET, R_MAX, K_SOLO = 30.0, 15.0, 10, 3
_ERR_POOL = TAU / math.sqrt(2)


def _reward(err_std):
    """Expected quantized reward (credits) for an estimate of this error std."""
    return sum(math.erf(m * BUCKET / (err_std * math.sqrt(2))) for m in range(1, R_MAX + 1))


def _err_solo(offset):
    """Best solo error: the per-agent offset (offset/sqrt2) cannot be measured away."""
    return math.sqrt((offset / math.sqrt(2)) ** 2 + TAU ** 2 / K_SOLO)


def rational_error(offset, price):
    """Pool (error -> floor) iff the reward gained beats the price, else stay solo."""
    gain = _reward(_ERR_POOL) - _reward(_err_solo(offset))
    return _ERR_POOL if gain > price else _err_solo(offset)


# --- empirical points --------------------------------------------------------
PARTNERS = [("liar", "always lies"), ("mixed_liar", "mixed"),
            ("honest_cooperator", "never lies")]
OFF_COLOR = {0: "#5aa9e6", 200: "#c0392b"}          # easy = blue, hard = red
MODEL_MARK = {"gpt54": "o", "qwen": "^"}
MODEL_NAME = {"gpt54": "GPT-5.4", "qwen": "Qwen3-32B"}


def _match_points(dirs):
    """[(partner, offset, price, model, mean_error)] over every completed grid match."""
    pts = []
    for model, base in dirs:
        for path in sorted(glob.glob(os.path.join(base, "grid_*_b*_p*_s*.jsonl"))):
            try:
                ev = load_events(path)
            except (OSError, ValueError, KeyError):
                continue
            # mean error is per-round, so a truncated match (e.g. a chatty Qwen
            # honest run that overflows context mid-way) is still usable — accept
            # any transcript with at least a few scored rounds.
            if sum(e.get("event") == "round_end" for e in ev) < 3:
                continue
            ms = next((e for e in ev if e.get("event") == "match_start"), None)
            cfg = (ms or {}).get("config", {})
            price, offset = cfg.get("min_trade_price"), cfg.get("bias_sigma")
            seats = (ms or {}).get("seats", {})
            llm = next((a for a, w in seats.items()
                        if w not in ("Liar", "MixedLiar", "HonestCooperator")), None) \
                or (cfg.get("agent_ids") or ["A"])[0]
            errs = [e["result"]["errors"].get(llm) for e in ev
                    if e.get("event") == "round_end"]
            errs = [x for x in errs if isinstance(x, (int, float)) and x == x]
            if not errs:
                continue
            partner = next((p for p, _ in PARTNERS if f"_{p}_" in os.path.basename(path)), "?")
            pts.append((partner, offset, price, model, sum(errs) / len(errs)))
    return pts


def main(argv=None):
    """Render the three-panel difficulty x cost error scatter."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=os.path.join(repo, "paper", "fig", "cost_error.pdf"))
    args = ap.parse_args(argv)
    dirs = [("gpt54", os.path.join(repo, "runs", "gpt54")),
            ("qwen", os.path.join(repo, "runs", "qwen"))]
    pts = _match_points(dirs)
    if not pts:
        raise SystemExit("no completed grid_* matches yet (run --stage grid)")

    prices = sorted({p for _, _, p, _, _ in pts})
    offsets = sorted({o for _, o, _, _, _ in pts})
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.3), sharey=True)
    xr = [min(prices) * 0.7, max(prices) * 1.4]
    for ax, (partner, label) in zip(axes, PARTNERS):
        # analytical reference per difficulty
        xs = [min(prices) * 0.7 * 1.05 ** i for i in range(80)]
        while xs[-1] < max(prices) * 1.4:
            xs.append(xs[-1] * 1.05)
        for off in offsets:
            ax.plot(xs, [rational_error(off, x) for x in xs], "--",
                    color=OFF_COLOR.get(off, "#888"), lw=1, alpha=0.5, zorder=1)
        # empirical points, jittered in x so seeds don't overlap
        for i, (pt, off, price, model, err) in enumerate(
                [p for p in pts if p[0] == partner]):
            jit = 1.0 + 0.06 * ((i % 3) - 1)
            ax.scatter(price * jit, err, s=46, marker=MODEL_MARK.get(model, "x"),
                       color=OFF_COLOR.get(off, "#888"), edgecolor="white", lw=0.6,
                       zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(prices)
        ax.set_xticklabels([f"{p:g}" for p in prices])
        ax.set_xlim(*xr)
        ax.set_title(f"partner: {label}", fontsize=10)
        ax.set_xlabel("price of information")
        ax.grid(True, axis="y", color="#eee", lw=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("LLM mean estimation error")
    # one shared legend: difficulty (colour) + model (marker)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="s", ls="", color=OFF_COLOR[o],
                      label=f"offset {o:g} ({'hard' if o else 'easy'})") for o in offsets]
    handles += [Line2D([], [], marker=MODEL_MARK[m], ls="", color="#555",
                       label=MODEL_NAME[m]) for m in MODEL_MARK if any(p[3] == m for p in pts)]
    handles += [Line2D([], [], ls="--", color="#999", label="rational optimum")]
    axes[-1].legend(handles=handles, frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out} and {png} from {len(pts)} matches")


if __name__ == "__main__":
    main()
