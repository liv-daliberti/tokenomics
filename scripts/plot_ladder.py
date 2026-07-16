"""The gap-ladder dose-response figure for the paper: buy-rate on the liar's
fabricated offers (top row) and discrimination (honest-buy minus fake-buy,
bottom row) as the scaffold escalates, one panel per difficulty, one line per
model. Reads the ladder cells directly (same collection as ladder_report.py)
so the figure can never drift from the table.

    python scripts/plot_ladder.py [--out paper/fig/ladder.pdf]
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "ladder_report", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "ladder_report.py"))
_lr = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_lr)

RUNGS = _lr.RUNGS                                     # base -> ... -> flag
RUNG_LABELS = {"base": "baseline", "mem": "+memory\nnotebook",
               "elicit": "+state belief\nbefore buying", "hist": "+seller\ntrack record",
               "flag": "+own verdict\nshown"}
# entity-consistent, CVD-validated pair (dataviz six checks, light surface)
MODELS = [("gpt54", "GPT-5.4", "#1f77b4", "o", "-"),
          ("qwen", "Qwen3-32B", "#c0392b", "^", "--")]
DIFFS = [("easy", "easy (offset 0)"), ("hard", "hard wall (offset 200)")]


def main(argv=None):
    """Render the 2x2 ladder figure from the current trust_*/ladder_* runs."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=os.path.join(repo, "paper", "fig", "ladder.pdf"))
    args = ap.parse_args(argv)

    arms = [("gpt54", os.path.join(repo, "runs", "gpt54")),
            ("qwen", os.path.join(repo, "runs", "qwen"))]
    rows = _lr.rows_for(_lr.collect(arms)[0])
    cell = {(r["model"], r["rung"], r["partner"], r["difficulty"]): r for r in rows}
    have = [rg for rg in RUNGS
            if any(k[1] == rg for k in cell)]          # plot only rungs with data
    if not have:
        raise SystemExit("no trust_*/ladder_* matches to plot")

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.6), sharex=True)
    xs = list(range(len(have)))
    for col, (lvl, title) in enumerate(DIFFS):
        ax_buy, ax_disc = axes[0][col], axes[1][col]
        for mk, name, color, marker, ls in MODELS:
            def series(metric, partner="liar"):
                """This model x difficulty's metric across the rungs on the axis."""
                return [
                    (cell.get((mk, rg, partner, lvl)) or {}).get(metric)
                    for rg in have
                ]
            buys = series("buy_fab")
            disc = [None if (h is None or (f := cell.get((mk, rg, "liar", lvl),
                                                         {}).get("buy_fab")) is None)
                    else h - f
                    for rg, h in zip(have, series("buy_hon", "honest_cooperator"))]
            for ax, ys in ((ax_buy, buys), (ax_disc, disc)):
                pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
                if pts:
                    ax.plot(*zip(*pts), ls, marker=marker, ms=8, lw=2,
                            color=color, label=name)
        ax_buy.set_title(title, fontsize=10)
        ax_buy.set_ylim(-0.04, 1.04)
        ax_buy.axhline(0, color="#9aa4b2", lw=1, ls=":")
        ax_disc.set_ylim(-0.54, 1.04)
        ax_disc.axhline(0, color="#9aa4b2", lw=1, ls=":")
        ax_disc.set_xticks(xs)
        ax_disc.set_xticklabels([RUNG_LABELS.get(r, r) for r in have], fontsize=8)
        for ax in (ax_buy, ax_disc):
            ax.grid(True, axis="y", color="#eee", lw=0.7)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    axes[0][0].set_ylabel("buys the liar's fake\n(0 = gap closed)")
    axes[1][0].set_ylabel("discrimination\n(buys real − buys fake)")
    axes[0][0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("Does any scaffold make detection change the purchase?",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out} and {png} ({len(have)} rung(s) present)")


if __name__ == "__main__":
    main()
