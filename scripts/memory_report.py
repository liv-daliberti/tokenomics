"""Aggregate the cross-game memory ablation: full context vs markdown notes.

Reads mem_{context|markdown}_b<off>_s<seed>.jsonl matches from a runs dir
(default runs/gpt54), joins per-match token usage from the driver's
manifest.json when present, and reports — per (memory, offset) cell —
survival, cooperation, mean error, and cost. This is the experiment's whole
question: does an agent that must JOURNAL its memory (bounded context, notes
only) cooperate and survive like one that remembers everything verbatim, and
at what fraction of the token bill?

    python scripts/memory_report.py [runs_dir] [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.metrics import load_events, summary as metric_summary

# The markdown arm has its own files; the CONTEXT arm reuses the gradient's
# matches (grad_b*), which are protocol-identical (memory="context" is the
# preset default) — the driver deliberately never pays for them twice.
_NAME = re.compile(r"(?:mem_(markdown)|(grad))_b(\d+)_s(\d+)\.jsonl$")


def collect(runs_dir: str) -> dict:
    """{(memory, offset): [per-match dicts]} for every completed mem match.

    Only offsets that have at least one completed MARKDOWN match are reported,
    so stray gradient offsets don't bloat the comparison."""
    manifest = {}
    mpath = os.path.join(runs_dir, "manifest.json")
    if os.path.exists(mpath):
        try:
            manifest = json.load(open(mpath))
        except ValueError:
            pass
    cells = defaultdict(list)
    paths = sorted(glob.glob(os.path.join(runs_dir, "mem_markdown_b*_s*.jsonl"))
                   + glob.glob(os.path.join(runs_dir, "grad_b*_s*.jsonl")))
    for path in paths:
        m = _NAME.search(path)
        if not m:
            continue
        memory = "markdown" if m.group(1) else "context"
        off, seed = int(m.group(3)), int(m.group(4))
        try:
            ev = load_events(path)
        except (ValueError, KeyError):
            continue                      # torn/partial file: skip, don't crash
        if not any(e.get("event") == "match_end" for e in ev):
            continue                      # incomplete match: not comparable
        s = metric_summary(ev)
        name = os.path.basename(path)[:-6]
        usage = (manifest.get(name) or {}).get("usage") or {}
        cells[(memory, off)].append({
            "seed": seed,
            "survivors": s["survivors"], "n_agents": s["n_agents"],
            "cooperation": s["cooperation"]["cooperation_index"],
            "welfare": s["welfare"],
            "notes": sum(1 for e in ev if e.get("event") == "notes"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        })
    return cells


def aggregate(cells: dict) -> list:
    """One row per (memory, offset): means over seeds, tokens where known.
    Context rows are kept only at offsets where the markdown arm exists."""
    md_offsets = {off for (memory, off) in cells if memory == "markdown"}
    cells = {k: v for k, v in cells.items()
             if k[0] == "markdown" or k[1] in md_offsets}
    rows = []
    for (memory, off), ms in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        n = len(ms)
        surv = sum(m["survivors"] / max(1, m["n_agents"]) for m in ms) / n
        coop = sum(m["cooperation"] for m in ms) / n
        welf = sum(m["welfare"] for m in ms) / n
        toks = [m["prompt_tokens"] for m in ms if m["prompt_tokens"]]
        rows.append({
            "memory": memory, "offset": off, "n": n,
            "survival": round(surv, 3), "cooperation": round(coop, 3),
            "welfare": round(welf, 1),
            "notes_per_match": round(sum(m["notes"] for m in ms) / n, 1),
            "prompt_tokens_mean": round(sum(toks) / len(toks)) if toks else None,
        })
    return rows


def main(argv=None) -> None:
    """CLI: aggregate, print the comparison table, optionally write JSON."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs_dir", nargs="?",
                    default=os.path.join(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))), "runs", "gpt54"))
    ap.add_argument("--json", default=None, help="also write rows to this path")
    args = ap.parse_args(argv)
    rows = aggregate(collect(args.runs_dir))
    if not rows:
        print(f"no completed mem_* matches under {args.runs_dir}")
        return
    hdr = f"{'memory':<10} {'offset':>6} {'n':>3} {'survival':>9} {'coop':>6} " \
          f"{'welfare':>8} {'notes':>6} {'in-tokens':>10}"
    print(hdr + "\n" + "-" * len(hdr))
    for r in rows:
        toks = f"{r['prompt_tokens_mean']:,}" if r["prompt_tokens_mean"] else "—"
        print(f"{r['memory']:<10} {r['offset']:>6} {r['n']:>3} {r['survival']:>9.3f} "
              f"{r['cooperation']:>6.3f} {r['welfare']:>8.1f} "
              f"{r['notes_per_match']:>6.1f} {toks:>10}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": rows}, fh, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
