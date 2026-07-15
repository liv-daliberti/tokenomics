"""Three candidate presentations of the difficulty x deception x model error
result, so we can compare and pick one. Writes err_lines.png, err_heatmap.png,
err_scatter.png under paper/fig/ from the current grid runs.

    python scripts/plot_error_views.py
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "plot_cost_error", os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_cost_error.py"))
_ce = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ce)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(REPO, "paper", "fig")
# honesty axis order: least -> most deceptive
HONESTY = [("honest_cooperator", "never\nlies"), ("mixed_liar", "sometimes\nlies"),
           ("liar", "always\nlies")]
MODELS = [("gpt54", "GPT-5.4", "#1f77b4", "o", "-"),
          ("qwen", "Qwen3-32B", "#c0392b", "^", "--")]
DIFFS = [(0, "easy (offset 0)"), (200, "hard wall (offset 200)")]


YLABEL = "% above the optimal strategy"


def _points():
    """Grid matches with error re-expressed as % above the optimal strategy for
    that (partner, offset). Points where no optimal reference exists are dropped."""
    raw = _ce._match_points([("gpt54", os.path.join(REPO, "runs", "gpt54")),
                             ("qwen", os.path.join(REPO, "runs", "qwen"))])
    opt = _ce.optimal_errors()
    out = []
    for partner, off, price, model, err in raw:
        pct = _ce.pct_above_optimal(err, partner, off, opt)
        if pct is not None:
            out.append((partner, off, price, model, pct))
    return out


def _agg(pts):
    """{(partner, model, offset): mean error} aggregated over price and seeds."""
    g = defaultdict(list)
    for partner, off, price, model, err in pts:
        g[(partner, model, off)].append(err)
    return {k: sum(v) / len(v) for k, v in g.items()}, \
           {k: len(v) for k, v in g.items()}


# --- view A: deception -> error lines ---------------------------------------
def view_lines(pts):
    """Two panels (easy/hard); x = partner honesty, y = error, line per model."""
    mean, n = _agg(pts)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    xs = list(range(len(HONESTY)))
    for ax, (off, title) in zip(axes, DIFFS):
        for mk, name, color, marker, ls in MODELS:
            ys = [mean.get((p, mk, off)) for p, _ in HONESTY]
            xx = [x for x, y in zip(xs, ys) if y is not None]
            yy = [max(y, 1) for y in ys if y is not None]   # floor 1% for log axis
            if yy:
                ax.plot(xx, yy, ls, marker=marker, ms=8, lw=2, color=color, label=name)
        ax.axhline(1, color="#9aa4b2", lw=1, ls=":")     # optimal play
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([lbl for _, lbl in HONESTY])
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", color="#eee", lw=0.7, which="both")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(YLABEL + "  (0 = optimal)")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("How far each model is from optimal play — near-optimal vs a "
                 "liar, but Qwen is far off", fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, "err_lines")


# --- view B: heatmap --------------------------------------------------------
def view_heatmap(pts):
    """rows = model x difficulty, cols = partner honesty, colour = error."""
    mean, n = _agg(pts)
    rows = [(mk, name, off, f"{name}\n{'easy' if off == 0 else 'hard'}")
            for mk, name, *_ in MODELS for off, _ in DIFFS]
    M = np.full((len(rows), len(HONESTY)), np.nan)
    for i, (mk, name, off, _) in enumerate(rows):
        for j, (p, _) in enumerate(HONESTY):
            v = mean.get((p, mk, off))
            if v is not None:
                M[i, j] = v
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    im = ax.imshow(M, aspect="auto", cmap="YlOrRd", vmin=0,
                   vmax=np.nanmax(M) if np.isfinite(M).any() else 1)
    ax.set_xticks(range(len(HONESTY)))
    ax.set_xticklabels([lbl.replace("\n", " ") for _, lbl in HONESTY])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[3].replace("\n", " ") for r in rows])
    for i in range(len(rows)):
        for j in range(len(HONESTY)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0f}%", ha="center", va="center",
                        color="black" if M[i, j] < np.nanmax(M) * 0.6 else "white",
                        fontsize=10, fontweight="bold")
    ax.set_title("% above optimal play: darker = further from optimal", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=YLABEL)
    fig.tight_layout()
    _save(fig, "err_heatmap")


# --- view C: cleaned price scatter ------------------------------------------
def view_scatter(pts):
    """The original scatter, cleaned: 3 partner panels, seeds' means joined per difficulty."""
    prices = sorted({p for _, _, p, _, _ in pts}) or [1, 8, 32]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.3), sharey=True)
    for ax, (partner, label) in zip(axes, [(p, l.replace("\n", " ")) for p, l in
                                           [("liar", "always lies"),
                                            ("mixed_liar", "sometimes lies"),
                                            ("honest_cooperator", "never lies")]]):
        for mk, name, color, marker, ls in MODELS:
            for off, _ in DIFFS:
                by_price = defaultdict(list)
                for pn, o, price, model, err in pts:
                    if pn == partner and model == mk and o == off:
                        by_price[price].append(err)
                if not by_price:
                    continue
                xs = sorted(by_price)
                ys = [max(sum(by_price[x]) / len(by_price[x]), 1) for x in xs]
                shade = color if off else _lighten(color)
                ax.plot(xs, ys, ls, marker=marker, ms=7, lw=1.6, color=shade)
        ax.axhline(1, color="#9aa4b2", lw=1, ls=":")
        ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.set_xticks(prices)
        ax.set_xticklabels([f"{p:g}" for p in prices])
        ax.set_title(f"partner: {label}", fontsize=10)
        ax.set_xlabel("price of information")
        ax.grid(True, axis="y", color="#eee", lw=0.7, which="both")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(YLABEL + "  (0 = optimal)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, marker=m, ls=l, label=n) for _, n, c, m, l in MODELS]
    h += [Line2D([], [], color="#888", marker="s", ls="", label="dark=hard, light=easy")]
    axes[-1].legend(handles=h, frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    _save(fig, "err_scatter")


def _lighten(hexc, f=0.5):
    r, g, b = (int(hexc[i:i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{int(c + (255 - c) * f):02x}" for c in (r, g, b))


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    """Render all three candidate views from the current grid runs."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None)          # ignored; writes fixed names
    ap.parse_args(argv)
    pts = _points()
    if not pts:
        raise SystemExit("no grid matches yet")
    view_lines(pts)
    view_heatmap(pts)
    view_scatter(pts)
    print(f"wrote err_lines / err_heatmap / err_scatter (.png/.pdf) from {len(pts)} matches")


if __name__ == "__main__":
    main()
