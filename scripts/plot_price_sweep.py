"""Plot the price-sweep result: buy-rate vs minimum trade price, for the liar's
fabrication (worth ~0) and the honest partner's real reading (worth ~6). Saves a
figure for the paper and the site. Reads the same cells as price_report.py.

    python scripts/plot_price_sweep.py [runs_dir] [--out paper/fig/price_sweep.pdf]
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
    "price_report", os.path.join(os.path.dirname(os.path.abspath(__file__)), "price_report.py"))
_pr = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pr)

# CVD-safe, print-safe: amber = the fabrication (caution), green = real info,
# plus distinct line styles + markers so colour is never the only cue.
C_FAKE, C_REAL = "#c98200", "#1fa768"
HONEST_VALUE = 6.0          # credits an honest reading is worth at this wall


def _series(cells, partner, key):
    """Sorted (price, rate) points for one partner's populated bucket."""
    pts = []
    for (pn, price), c in cells.items():
        xs = c[key]
        if pn == partner and xs:
            pts.append((price, sum(xs) / len(xs), len(xs)))
    return sorted(pts)


def main(argv=None) -> None:
    """Render the two-curve price-sweep figure."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs_dir", nargs="?",
                    default=os.path.join(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))), "runs", "gpt54"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "paper", "fig", "price_sweep.pdf"))
    args = ap.parse_args(argv)

    cells = _pr.collect(args.runs_dir)
    fake = _series(cells, "liar", "fab")
    real = _series(cells, "honest", "hon")
    if not fake and not real:
        raise SystemExit(f"no price_* matches under {args.runs_dir}")

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.axvline(HONEST_VALUE, color="#9aa4b2", lw=1, ls=":", zorder=1)
    ax.text(HONEST_VALUE * 1.05, 0.04, "value of a\nreal reading (~6)",
            color="#6b7280", fontsize=7.5, va="bottom")
    if fake:
        xs, ys, _ = zip(*fake)
        ax.plot(xs, ys, "-o", color=C_FAKE, lw=2, ms=6, label="buys the liar's fake (worth ~0)")
    if real:
        xs, ys, _ = zip(*real)
        ax.plot(xs, ys, "--s", color=C_REAL, lw=2, ms=6, label="buys a real reading (worth ~6)")

    ax.set_xscale("log", base=2)
    ax.set_xticks([0.5, 2, 8, 32])
    ax.set_xticklabels(["0.5", "2", "8", "32"])
    ax.set_xlabel("minimum trade price (credits)")
    ax.set_ylabel("GPT-5.4 buy-rate")
    ax.set_ylim(-0.03, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, axis="y", color="#e5e7eb", lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    # also a PNG alongside (for the web page / quick view)
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out} and {png}")


if __name__ == "__main__":
    main()
