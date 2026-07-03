"""Scripted baseline anchors for the interdependence gradient.

Runs the SAME cooperative game as the Qwen sweep (10 games × 5 rounds, paired
instrument bias) across the same offsets, but with deterministic scripted
policies — so the LLM dose-response can be read against principled anchors:

  * honest_cooperator — both agents broadcast real readings and pool: the
    cooperative CEILING (survival should stay ~1.0 at every offset).
  * bayesian_solo      — both agents measure alone, never share: the FLOOR that
    the wall is meant to kill (survival should collapse as the offset grows).
  * hoarder            — measure minimally, conserve credits: the "do nothing" arm.

No GPU. Metrics use gradient_report._row_from_events, so the anchor numbers are
computed identically to the Qwen runs. Writes docs/samples/gradient/gradient_anchors.json.

Usage: python scripts/scripted_gradient.py [--seeds 20] [--games 10] [--rounds 5]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agora.config import PRESETS                     # noqa: E402
from agora.policies import REGISTRY                  # noqa: E402
from agora.referee import run_match                  # noqa: E402
from agora.transcripts import Transcript             # noqa: E402
from gradient_report import _row_from_events, aggregate_rows  # noqa: E402

OFFSETS = [0, 50, 100, 150, 200, 250, 300, 350, 400, 500]
SPECS = ["honest_cooperator", "bayesian_solo", "hoarder"]


def run_point(spec: str, offset: float, seed: int, games: int, rounds: int) -> dict:
    """One scripted match at a given offset/seed; return its dose-response row."""
    cfg = PRESETS["cooperative"].with_(bias_sigma=float(offset), n_rounds=rounds, seed=seed)
    ids = cfg.agent_ids
    names = spec.split(",")
    pols = {a: REGISTRY[names[i % len(names)]](cfg, a, ids) for i, a in enumerate(ids)}
    tx = Transcript()
    run_match(cfg, pols, games, tx)
    return _row_from_events(tx.events)


def main(argv=None) -> None:
    """Sweep every (spec, offset) over seeds and dump the aggregated anchor JSON."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", default="docs/samples/gradient/gradient_anchors.json")
    a = ap.parse_args(argv)

    specs_out = {}
    for spec in SPECS:
        groups = {off: [run_point(spec, off, s, a.games, a.rounds) for s in range(a.seeds)]
                  for off in OFFSETS}
        specs_out[spec] = aggregate_rows(groups)
        surv = {int(r["offset"]): round(r["survivor_rate"]["mean"], 2) for r in specs_out[spec]}
        print(f"{spec:18s} survivor_rate by offset: {surv}")

    out = {
        "label": f"scripted anchors · {a.seeds} seeds × {len(OFFSETS)} offsets · "
                 f"{a.games} games × {a.rounds} rounds",
        "seeds": a.seeds, "games": a.games, "rounds": a.rounds,
        "specs": specs_out,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
