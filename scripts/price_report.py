"""As the lie gets more expensive: accept-rate vs the minimum trade price.

Reads price_<bot>_p<price>_s<seed>.jsonl matches (from scripts/gpt54\\_program.py
--stage price), groups by the match's min-trade-price floor and the partner,
and reports how often the LLM buys — the liar's fabricated readings (worth ~0)
and the honest partner's real readings (worth ~6 credits at this wall). A
cost-benefit buyer should stop buying the liar at any price and stop buying the
honest reading once the price exceeds its value; a reflexive buyer keeps buying
until it cannot afford to.

    python scripts/price_report.py [runs_dir]   # default runs/gpt54
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "lie_judge", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lie_judge.py"))
_lj = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_lj)


def collect(runs_dir: str) -> dict:
    """{(partner, price): {'fab': [accepts], 'hon': [accepts]}} over price matches."""
    cells: dict = defaultdict(lambda: {"fab": [], "hon": []})
    for path in sorted(glob.glob(os.path.join(runs_dir, "price_*_s*.jsonl"))):
        try:
            ev = _lj.load_events(path)
        except (OSError, ValueError, KeyError):
            continue
        ms = next((e for e in ev if e.get("event") == "match_start"), None)
        if not ms:
            continue
        price = ms.get("config", {}).get("min_trade_price")
        partner = "liar" if "_liar_" in os.path.basename(path) else "honest"
        for o in _lj._offers_with_context(ev):
            if not o["answered"] or o["seller"] == o.get("buyer"):
                continue
            # only the SCRIPTED partner's offers to the LLM (its fakes / real reads)
            bucket = "fab" if o["fabricated"] else "hon"
            cells[(partner, price)][bucket].append(1 if o["accepted"] else 0)
    return cells


def main(argv=None) -> None:
    """Print the accept-rate-vs-price table."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs_dir", nargs="?",
                    default=os.path.join(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))), "runs", "gpt54"))
    args = ap.parse_args(argv)
    cells = collect(args.runs_dir)
    if not cells:
        print(f"no price_* matches under {args.runs_dir}")
        return

    def rate(xs):
        """Accept-rate string with sample size, or a dash when no offers."""
        return f"{sum(xs)/len(xs):.2f} (n={len(xs)})" if xs else "—"

    prices = sorted({p for (_, p) in cells})
    print(f"{'partner':<8}{'min price':>10}{'buys fake':>16}{'buys real':>16}")
    print("-" * 50)
    for partner in ("liar", "honest"):
        for p in prices:
            c = cells.get((partner, p))
            if not c:
                continue
            print(f"{partner:<8}{p:>10g}{rate(c['fab']):>16}{rate(c['hon']):>16}")
    print("\nliar's fake reading is worth ~0; honest reading ~6 credits at this "
          "wall. A cost-benefit buyer's 'buys fake' should fall toward 0 as price "
          "rises; 'buys real' should hold until price exceeds ~6, then fall.")


if __name__ == "__main__":
    main()
