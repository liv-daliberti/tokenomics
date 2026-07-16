"""The GPT-5.4 program driver: run matrix, completion detection, cost math.

No subprocesses and no network — the driver's expensive parts are exercised
end-to-end against the stub endpoint separately; these tests pin the logic
that decides WHAT runs and what money it would cost.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_spec = importlib.util.spec_from_file_location(
    "gpt54_program",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "gpt54_program.py"))
prog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prog)


def test_matrix_mirrors_the_qwen_study():
    assert len(prog.build_matrix("grad", 10, 5)) == 140      # 14 offsets x 10 seeds
    assert len(prog.build_matrix("deconf", 10, 5)) == 100    # 10 offsets x 10 seeds
    assert len(prog.build_matrix("probe", 10, 5)) == 10      # 2 bots x 5 seeds
    # mem = 15 markdown matches + 15 grad-named context matches (reused files)
    mem = prog.build_matrix("mem", 10, 5)
    assert len(mem) == 30
    assert sum(1 for n, _ in mem if n.startswith("mem_markdown")) == 15
    assert sum(1 for n, _ in mem if n.startswith("grad_b")) == 15
    # "all" de-duplicates the mem context arm against the gradient:
    # 140 + 100 + 10 + 15 markdown = 265, and no name appears twice
    allruns = prog.build_matrix("all", 10, 5)
    assert len(allruns) == 265
    assert len({n for n, _ in allruns}) == len(allruns)
    assert len(prog.build_matrix("pilot", 10, 5)) == 3

    names = dict(allruns)
    assert names["deconf_b500_s9"]["FRAMING"] == "neutral"
    assert names["deconf_b500_s9"]["STRATEGY_HINT"] == "0"
    assert "FRAMING" not in names["grad_b0_s0"]              # prompted = preset default
    assert names["mem_markdown_b300_s4"]["MEMORY"] == "markdown"
    assert names["probe_liar_s4"]["POLICIES"] == "llm,liar"
    # pilot seeds must never collide with real seeds
    assert all("_s900" in n for n, _ in prog.build_matrix("pilot", 10, 5))


def test_market_regimes_are_pinned_and_cannot_mix():
    # paid (default): every run enforces censored chat + positive prices
    for name, ov in prog.build_matrix("all", 10, 5):
        assert ov["VALUES_VIA_TRADE_ONLY"] == "1", name
        assert ov["REQUIRE_PAID_TRADES"] == "1", name
        assert not name.endswith("_open")
    # open: Qwen-identical rules, and every filename carries the suffix so the
    # two regimes can never be pooled by a glob
    for name, ov in prog.build_matrix("all", 10, 5, market="open"):
        assert "VALUES_VIA_TRADE_ONLY" not in ov and "REQUIRE_PAID_TRADES" not in ov
        assert name.endswith("_open"), name


def test_price_stage_sweeps_the_floor_at_the_hard_wall():
    runs = prog.build_matrix("price", 10, 5)
    # 2 partners x len(PRICE_LEVELS) prices x TRUST_SEEDS
    assert len(runs) == 2 * len(prog.PRICE_LEVELS) * prog.TRUST_SEEDS
    names = dict(runs)
    r = names["price_liar_p8_s0"]
    assert r["MIN_TRADE_PRICE"] == "8" and r["BIAS_SIGMA"] == str(prog.PRICE_OFFSET)
    assert r["POLICIES"] == "llm,liar"
    assert r["STARTING_CREDITS"] == str(prog.PRICE_START_CREDITS)
    assert set(float(names[n]["MIN_TRADE_PRICE"]) for n in names) == set(prog.PRICE_LEVELS)
    # honest control present at every price too
    assert "price_honest_cooperator_p0_5_s0" in names


def test_grid_stage_covers_difficulty_price_partner():
    runs = prog.build_matrix("grid", 10, 5)
    assert len(runs) == (len(prog.GRID_PARTNERS) * len(prog.GRID_OFFSETS)
                         * len(prog.GRID_PRICES) * prog.GRID_SEEDS)
    names = dict(runs)
    r = names["grid_mixed_liar_b200_p8_s0"]
    assert r["POLICIES"] == "llm,mixed_liar" and r["MIN_TRADE_PRICE"] == "8"
    assert r["BIAS_SIGMA"] == "200" and r["STARTING_CREDITS"] == str(prog.PRICE_START_CREDITS)
    # all three partner honesty regimes and both difficulties present
    assert {n.split("_b")[0].replace("grid_", "") for n in names} == set(prog.GRID_PARTNERS)
    assert {names[n]["BIAS_SIGMA"] for n in names} == {str(o) for o in prog.GRID_OFFSETS}


def test_ladder_stage_matrix():
    runs = prog.build_matrix("ladder", 10, 5)
    # 4 rungs x 2 bots x 2 difficulties x TRUST_SEEDS
    assert len(runs) == len(prog.LADDER_RUNGS) * 2 * 2 * prog.TRUST_SEEDS
    names = dict(runs)
    r = names["ladder_flag_liar_hard_b200_s1"]
    assert r["SHOW_JUDGE_FLAG"] == "1" and r["POLICIES"] == "llm,liar"
    assert r["BIAS_SIGMA"] == "200" and r["GAMES"] == str(prog.TRUST_GAMES)
    # paid market pinned, like the R0 trust runs these compare against
    assert r["VALUES_VIA_TRADE_ONLY"] == "1" and r["REQUIRE_PAID_TRADES"] == "1"
    # each rung sets exactly ITS one scaffold knob and no other
    knob_of = dict(prog.LADDER_RUNGS)
    scaffolds = {"MEMORY", "ELICIT_PFAB", "SHOW_SELLER_HISTORY", "SHOW_JUDGE_FLAG"}
    for name, ov in runs:
        rung = name.split("_")[1]
        assert set(ov) & scaffolds == set(knob_of[rung]), name
    # R0 is the existing trust_* files: never re-emitted, and ladder runs are
    # never part of "all" (the Qwen-mirror program)
    assert not any(n.startswith("trust_") for n, _ in runs)
    assert not any(n.startswith("ladder_") for n, _ in prog.build_matrix("all", 10, 5))
    # every scaffold env var is scrubbed from the child environment between runs
    assert scaffolds <= set(prog.PROTOCOL_VARS)


def test_mixed_liar_is_registered_and_partial():
    from agora.policies import REGISTRY
    assert "mixed_liar" in REGISTRY
    assert REGISTRY["liar"].lie_prob == 1.0
    assert 0 < REGISTRY["mixed_liar"].lie_prob < 1.0
    assert REGISTRY["honest_cooperator"].lie_prob == 0.0


def test_protocol_vars_cover_every_qwen_match_knob():
    # If qwen_match grows a new env knob, the scrub list must grow with it —
    # otherwise a stray shell export could silently change a paid condition.
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts", "qwen_match.py")).read()
    import re as _re
    knobs = set(_re.findall(r'os\.environ\.get\("([A-Z_]+)"', src)) \
        | set(_re.findall(r'os\.environ\["([A-Z_]+)"\]', src))
    knobs -= {"API_KEY"}          # credential, deliberately not scrubbed
    missing = knobs - set(prog.PROTOCOL_VARS)
    assert not missing, f"qwen_match knobs missing from PROTOCOL_VARS: {missing}"


def test_is_complete_requires_match_end():
    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, "m")
        assert not prog.is_complete(base)                    # no file
        with open(base + ".jsonl", "w") as fh:
            fh.write(json.dumps({"event": "game_start"}) + "\n")
        assert not prog.is_complete(base)                    # truncated match
        with open(base + ".jsonl", "a") as fh:
            fh.write(json.dumps({"event": "match_end", "n_games": 10}) + "\n")
        assert prog.is_complete(base)


def test_usage_line_is_machine_readable():
    m = prog.USAGE_RE.findall(
        '[qwen_match] USAGE {"calls": 220, "prompt_tokens": 9, "completion_tokens": 2}')
    assert json.loads(m[0])["calls"] == 220


if __name__ == "__main__":
    test_matrix_mirrors_the_qwen_study()
    test_is_complete_requires_match_end()
    test_usage_line_is_machine_readable()
    print("ok")
