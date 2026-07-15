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
