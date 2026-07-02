"""Import a transcript JSONL into the web app's gallery so it's viewable there.

    python scripts/import_to_viewer.py <transcript.jsonl> ["Title"]

Writes <id>.jsonl + <id>.json into the app's runs dir (AGORA_RUNS or runs/web),
computing the same meta the app records for its own games. Refresh the app.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from analysis.metrics import load_events, summary


def main() -> None:
    """Import a transcript JSONL into the web app's gallery (copy it in and write the computed metadata)."""
    src = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(src).replace(".jsonl", "")
    runs = os.environ.get("AGORA_RUNS", os.path.join(REPO, "runs", "web"))
    os.makedirs(runs, exist_ok=True)

    jid = re.sub(r"[^a-zA-Z0-9_-]", "-", os.path.basename(src).replace(".jsonl", ""))
    ev = load_events(src)
    s = summary(ev)
    cfg = next(e for e in ev if e["event"] == "game_start")["config"]

    meta = {
        "id": jid, "status": "done", "title": title,
        "created": os.path.getmtime(src),
        "backend": "llm", "preset": "(imported)", "seed": cfg.get("seed"),
        "n_agents": len(cfg.get("agent_ids", [])), "tau": cfg.get("tau"),
        "framing": cfg.get("framing"), "survival_cost": cfg.get("survival_cost"),
        "n_games": s.get("n_games"),
        "horizon": ("known %d-round" % cfg["n_rounds"]) if cfg.get("reveal_horizon")
                   else "hidden (γ=%.2f)" % cfg.get("gamma", 0),
        "rounds": len([e for e in ev if e["event"] == "round_end"]),
        "deception_rate": s["deception"]["deception_rate"],
        "cooperation": s["cooperation"]["cooperation_index"],
        "survivors": s["survivors"], "gini": s["gini_final_credits"],
        "welfare": s["welfare"], "parse_fail_rate": s["diagnostics"]["parse_fail_rate"],
    }
    shutil.copy(src, os.path.join(runs, jid + ".jsonl"))
    with open(os.path.join(runs, jid + ".json"), "w") as fh:
        json.dump(meta, fh)
    print(f"imported '{title}' as id={jid} into {runs}")
    print("refresh the web app to see it in the gallery")


if __name__ == "__main__":
    main()
