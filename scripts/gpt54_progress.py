"""Live progress for the GPT-5.4 program: one bar per match, plus totals.

Reads the transcripts as they grow — no driver state needed. Run it once, or
leave it in a terminal with:

    watch -n 30 python3 scripts/gpt54_progress.py [runs_dir]

Progress is rounds-completed over rounds-expected (n_games x n_rounds from
each match's own config), floored by finished games so an early-ended game
(everyone eliminated) still counts as done.
"""
from __future__ import annotations

import glob
import json
import os
import sys

BAR = 28


def match_progress(path: str):
    """(done_rounds, expected_rounds, finished, label) for one transcript."""
    n_games, n_rounds, games_done, rounds_done, ended = 10, 5, 0, 0, False
    try:
        with open(path) as fh:
            for line in fh:
                if '"match_start"' in line:
                    e = json.loads(line)
                    n_games = e.get("n_games", n_games)
                    n_rounds = e.get("config", {}).get("n_rounds", n_rounds)
                elif '"game_end"' in line:
                    games_done += 1
                elif '"round_end"' in line:
                    rounds_done += 1
                elif '"match_end"' in line:
                    ended = True
    except (OSError, ValueError):
        pass
    expected = n_games * n_rounds
    done = max(rounds_done, games_done * n_rounds)
    if ended:
        done = expected
    return done, expected, ended


def main() -> None:
    """Print one progress bar per transcript in the runs dir, then totals."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(repo, "runs", "gpt54")
    paths = sorted(glob.glob(os.path.join(runs_dir, "*.jsonl")))
    if not paths:
        print(f"no matches yet under {runs_dir}")
        return
    tot_done = tot_exp = n_finished = 0
    for p in paths:
        done, exp, ended = match_progress(p)
        tot_done += done
        tot_exp += exp
        n_finished += ended
        frac = done / exp if exp else 0.0
        bar = "█" * round(frac * BAR) + "░" * (BAR - round(frac * BAR))
        state = "done" if ended else f"{frac:4.0%}"
        print(f"{os.path.basename(p)[:-6]:<34} {bar} {state:>5}  ({done}/{exp} rounds)")
    frac = tot_done / tot_exp if tot_exp else 0.0
    print("-" * (36 + BAR + 20))
    print(f"{'TOTAL':<34} {n_finished}/{len(paths)} matches finished · "
          f"{frac:.0%} of rounds ({tot_done}/{tot_exp})")


if __name__ == "__main__":
    main()
