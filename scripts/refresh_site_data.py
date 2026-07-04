#!/usr/bin/env python3
"""Refresh the committed site data from the current sweep runs, then commit+push.

Regenerates the multi-seed aggregate (docs/samples/gradient/gradient_aggregate.json)
and re-copies one representative gallery match per offset
(docs/samples/sweep_offNNN.jsonl) from COMPLETE runs only. Safe to run repeatedly
while a sweep is in progress: an offset with no complete run yet keeps its existing
sample, and the aggregate is only overwritten when at least one complete run exists
(so the charts never blank out mid-sweep).

Usage:  python scripts/refresh_site_data.py [--no-push]
"""
from __future__ import annotations

import glob
import importlib.util
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from analysis.metrics import load_events, summary  # noqa: E402

RUN_GLOB = os.path.join(REPO, "runs/qwen/grad_b*_s*.jsonl")
AGG = os.path.join(REPO, "docs/samples/gradient/gradient_aggregate.json")
DECONF_GLOB = os.path.join(REPO, "runs/qwen/deconf_b*_s*.jsonl")
DECONF_AGG = os.path.join(REPO, "docs/samples/gradient/deconf_aggregate.json")
SAMPLE_DIR = os.path.join(REPO, "docs/samples")
OFFSETS = [0, 50, 100, 150, 200, 250, 300, 350, 400, 500]

_spec = importlib.util.spec_from_file_location(
    "gradient_report", os.path.join(REPO, "scripts", "gradient_report.py"))
_gr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gr)


def _complete(path: str) -> bool:
    """A finished match ends with a match_end event."""
    try:
        with open(path) as fh:
            return "match_end" in fh.read()
    except OSError:
        return False


def refresh_aggregate() -> int:
    """Rewrite the committed aggregate from complete runs; return run count.

    Writes to a temp path first and only replaces the committed file when at least
    one complete run exists, so a mid-sweep run never blanks the charts."""
    tmp = AGG + ".tmp"
    n = _gr.write_aggregate(RUN_GLOB, tmp)
    if n >= 1:
        shutil.move(tmp, AGG)
    else:
        if os.path.exists(tmp):
            os.remove(tmp)
    return n


def refresh_deconf() -> int:
    """Rebuild the DE-CONFOUNDED aggregate (deconf_b<off>_s<seed>, neutral framing /
    no hint) from complete runs; return the run count. The home + gradient report
    read this for the de-confounding comparison. Built here (not via
    gradient_report --aggregate, whose filename regex is hardcoded to grad_b*)."""
    import json
    groups: dict = {}
    for p in glob.glob(DECONF_GLOB):
        m = re.search(r"deconf_b(\d+)_s(\d+)", os.path.basename(p))
        if not m or not _complete(p):
            continue
        groups.setdefault(int(m.group(1)), []).append(_gr._row_from_events(load_events(p)))
    if not groups:
        return 0
    rows = _gr.aggregate_rows(groups)
    # Fabrication is POOLED per-offer (same estimator as the confounded aggregate via
    # write_aggregate); a per-run average here emits junk n=1 rows. Keep the paths in sync.
    pooled = _gr._pooled_deception(DECONF_GLOB)
    for r in rows:
        if r["offset"] in pooled:
            r["deception"] = pooled[r["offset"]]
    tot = sum(r["n_seeds"] for r in rows)
    with open(DECONF_AGG, "w") as fh:
        json.dump({"label": f"{tot} de-confounded runs (neutral framing, no averaging hint) "
                            f"· mean ± 95% CI", "rows": rows}, fh)
    return tot


def refresh_gallery() -> list:
    """For each offset with a complete run, copy a representative match to
    docs/samples/sweep_offNNN.jsonl. Default pick is the first surviving seed. At
    offset 0 the finding is that cooperation is a coin flip, so we deliberately
    feature a SILENT run (fewest messages) — the counterintuitive "they could
    cooperate but don't" case — to match the gallery card."""
    updated = []
    for off in OFFSETS:
        complete = []
        for seed in range(10):
            f = os.path.join(REPO, f"runs/qwen/grad_b{off}_s{seed}.jsonl")
            if os.path.exists(f) and _complete(f):
                complete.append((seed, f, summary(load_events(f))))
        if not complete:
            continue
        if off == 0:
            # silent case: fewest messages (ties → first), among those a survivor
            def _msgs(item):
                """Message count of a (seed, path, summary) candidate."""
                return sum(1 for e in load_events(item[1]) if e["event"] == "message")
            survivors = [c for c in complete if c[2]["survivors"] >= 1] or complete
            chosen = min(survivors, key=_msgs)[1]
        else:
            survived = [c for c in complete if c[2]["survivors"] >= 1]
            chosen = (survived[0] if survived else complete[0])[1]
        shutil.copy(chosen, os.path.join(SAMPLE_DIR, f"sweep_off{off:03d}.jsonl"))
        updated.append(off)
    return updated


def git_commit_push(msg: str, push: bool) -> bool:
    """Commit the refreshed data (and push) iff something actually changed. Rebases
    onto origin before pushing so a concurrent commit (a story edit, another refresh)
    doesn't reject the push — important when this runs unattended from cron."""
    paths = [AGG, DECONF_AGG] + glob.glob(os.path.join(SAMPLE_DIR, "sweep_off*.jsonl"))
    paths = [p for p in paths if os.path.exists(p)]
    subprocess.run(["git", "-C", REPO, "add"] + paths, check=True)
    if subprocess.run(["git", "-C", REPO, "diff", "--cached", "--quiet"]).returncode == 0:
        return False  # nothing staged
    subprocess.run(["git", "-C", REPO, "commit", "-q", "-m", msg], check=True)
    if push:
        # data files are only touched by this script, so a rebase won't conflict.
        subprocess.run(["git", "-C", REPO, "pull", "--rebase", "origin", "main"])
        subprocess.run(["git", "-C", REPO, "push", "origin", "main"])
    return True


def main() -> None:
    """Refresh aggregate + gallery from complete runs and commit if changed."""
    push = "--no-push" not in sys.argv
    n = refresh_aggregate()
    n_dec = refresh_deconf()
    offs = refresh_gallery()
    complete = sum(1 for f in glob.glob(RUN_GLOB) if _complete(f))
    print(f"complete runs: {complete} | confounded agg: {n} | de-confounded agg: {n_dec} "
          f"| gallery offsets refreshed: {offs}")
    if n == 0 and n_dec == 0:
        print("no complete runs yet — nothing to publish")
        return
    msg = (f"Refresh site data: {n} confounded + {n_dec} de-confounded runs "
           f"({len(offs)} offsets browsable) [auto]")
    changed = git_commit_push(msg, push)
    print("committed + pushed" if changed else "no changes to commit")


if __name__ == "__main__":
    main()
