"""Multi-seed study: run a preset over many seeds (scripted, no GPU) and report
aggregate metrics with 95% confidence intervals — so results are quantitative,
not a single anecdotal transcript.

Examples:
  python scripts/study.py --preset cooperative --policies honest_cooperator --seeds 50
  python scripts/study.py --preset cooperative --seeds 50 \
      --compare honest_cooperator bayesian_solo liar hoarder
  python scripts/study.py --preset cooperative --games 3 --seeds 30 --json runs/study.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agora.config import PRESETS
from agora.policies import REGISTRY
from agora.referee import Referee, run_match
from agora.transcripts import Transcript
from analysis.metrics import summary

# Metrics reported per policy: percentage-style (0..1) vs raw scalars.
PCT = ["survival", "cooperation", "reciprocity", "deception"]
RAW = ["welfare"]
METRICS = PCT + RAW


def run_one(preset: str, spec: str, seed: int, games: int) -> Dict[str, float]:
    """Run a single seed and return its scalar metrics."""
    cfg = PRESETS[preset].with_(seed=seed)
    ids = cfg.agent_ids
    names = spec.split(",")
    pols = {a: REGISTRY[names[i % len(names)]](cfg, a, ids) for i, a in enumerate(ids)}
    tx = Transcript()
    if games > 1:
        run_match(cfg, pols, games, tx)
    else:
        Referee(cfg, pols, tx).run()
    s = summary(tx.events)
    return {
        "survival": (s["survivors"] / s["n_agents"]) if s["n_agents"] else float("nan"),
        "cooperation": s["cooperation"]["cooperation_index"],
        "reciprocity": s["reciprocity"]["reciprocity_index"],
        "deception": s["deception"]["deception_rate"],
        "welfare": float(s["welfare"]),
    }


def aggregate(values: List[float]) -> Tuple[float, float, int]:
    """Return (mean, 95% CI half-width, n) over the non-NaN values."""
    xs = [v for v in values if v == v]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = statistics.mean(xs)
    ci = 1.96 * statistics.stdev(xs) / math.sqrt(n) if n > 1 else 0.0
    return mean, ci, n


def _cell(metric: str, mean: float, ci: float) -> str:
    if mean != mean:
        return "n/a"
    return f"{mean:.0%}±{ci:.0%}" if metric in PCT else f"{mean:.1f}±{ci:.1f}"


def main(argv: Optional[List[str]] = None) -> None:
    """CLI: run the study and print an aggregate table (optionally dump JSON)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="cooperative")
    ap.add_argument("--policies", default="honest_cooperator",
                    help="policy spec (comma-separated, cycled over agents)")
    ap.add_argument("--compare", nargs="*", default=None,
                    help="several policy specs to run side by side")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--games", type=int, default=1, help=">1 = an N-game match per seed")
    ap.add_argument("--json", default=None, help="also write the aggregates here")
    a = ap.parse_args(argv)

    specs = a.compare if a.compare else [a.policies]
    results: Dict[str, Dict[str, Tuple[float, float, int]]] = {}
    for spec in specs:
        runs = [run_one(a.preset, spec, s, a.games) for s in range(a.seeds)]
        results[spec] = {m: aggregate([r[m] for r in runs]) for m in METRICS}

    print(f"\nStudy: preset={a.preset}  seeds={a.seeds}  games={a.games}\n")
    hdr = f"{'policy':22s}" + "".join(f"{m:>15s}" for m in METRICS)
    print(hdr)
    print("-" * len(hdr))
    for spec in specs:
        row = f"{spec:22s}" + "".join(
            f"{_cell(m, *results[spec][m][:2]):>15s}" for m in METRICS)
        print(row)
    print("\n(± is a 95% CI across seeds; survival/cooperation/reciprocity/deception "
          "are fractions)")

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        out = {spec: {m: {"mean": results[spec][m][0], "ci95": results[spec][m][1],
                          "n": results[spec][m][2]} for m in METRICS}
               for spec in specs}
        with open(a.json, "w") as fh:
            json.dump({"preset": a.preset, "seeds": a.seeds, "games": a.games,
                       "results": out}, fh, indent=2)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
