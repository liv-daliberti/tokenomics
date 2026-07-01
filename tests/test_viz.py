"""The HTML report renders and flags fabricated sales."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agora.config import GameConfig
from agora.policies import REGISTRY
from agora.referee import Referee
from analysis.metrics import diagnostics
from analysis.viz import render_html


def _game(spec, **kw):
    cfg = GameConfig(agent_ids=["A", "B"], horizon_mode="fixed", n_rounds=2, **kw)
    names = spec.split(",")
    policies = {a: REGISTRY[names[i % len(names)]](cfg, a, cfg.agent_ids)
                for i, a in enumerate(cfg.agent_ids)}
    return Referee(cfg, policies).run()


def test_render_is_valid_html():
    res = _game("honest_cooperator,bayesian_solo")
    doc = render_html(res.transcript.events, "t")
    assert doc.startswith("<!doctype html>") and doc.rstrip().endswith("</html>")
    assert "Round 0" in doc


def test_liar_is_flagged_in_report():
    res = _game("liar,honest_cooperator", seed=3)
    doc = render_html(res.transcript.events, "t")
    assert "FABRICATED" in doc, "the report must flag the liar's fabricated sales"


def test_diagnostics_clean_for_scripted():
    res = _game("honest_cooperator,bayesian_solo")
    dg = diagnostics(res.transcript.events)
    assert dg["parse_fail_rate"] == 0.0 and dg["misaddress_rate"] == 0.0
    assert dg["actions"] > 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all viz tests pass")
