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


def refresh_gallery() -> list:
    """For each offset with a complete run, copy a representative match (first seed
    with a survivor, else first complete) to docs/samples/sweep_offNNN.jsonl."""
    updated = []
    for off in OFFSETS:
        best = fallback = None
        for seed in range(10):
            f = os.path.join(REPO, f"runs/qwen/grad_b{off}_s{seed}.jsonl")
            if not (os.path.exists(f) and _complete(f)):
                continue
            s = summary(load_events(f))
            if fallback is None:
                fallback = f
            if s["survivors"] >= 1:
                best = f
                break
        chosen = best or fallback
        if chosen:
            shutil.copy(chosen, os.path.join(SAMPLE_DIR, f"sweep_off{off:03d}.jsonl"))
            updated.append(off)
    return updated


def git_commit_push(msg: str, push: bool) -> bool:
    """Commit the refreshed data (and push) iff something actually changed."""
    paths = [AGG] + glob.glob(os.path.join(SAMPLE_DIR, "sweep_off*.jsonl"))
    subprocess.run(["git", "-C", REPO, "add"] + paths, check=True)
    if subprocess.run(["git", "-C", REPO, "diff", "--cached", "--quiet"]).returncode == 0:
        return False  # nothing staged
    subprocess.run(["git", "-C", REPO, "commit", "-q", "-m", msg], check=True)
    if push:
        subprocess.run(["git", "-C", REPO, "push", "origin", "main"], check=True)
    return True


def main() -> None:
    """Refresh aggregate + gallery from complete runs and commit if changed."""
    push = "--no-push" not in sys.argv
    n = refresh_aggregate()
    offs = refresh_gallery()
    complete = sum(1 for f in glob.glob(RUN_GLOB) if _complete(f))
    print(f"complete runs: {complete}/100 | aggregate runs: {n} | gallery offsets refreshed: {offs}")
    if n == 0:
        print("no complete runs yet — nothing to publish")
        return
    msg = (f"Refresh site data: {n} complete 10-game runs "
           f"({len(offs)} offsets browsable) [auto]")
    changed = git_commit_push(msg, push)
    print("committed + pushed" if changed else "no changes to commit")


if __name__ == "__main__":
    main()
