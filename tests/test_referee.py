"""End-to-end referee tests: a scripted game runs, scores, and is analysable.

Runnable with pytest or as a plain script (`python tests/test_referee.py`).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agora.config import GameConfig
from agora.policies import REGISTRY
from agora.referee import Referee
from analysis.metrics import cooperation, deception, summary


def _run(cfg, spec):
    ids = cfg.agent_ids
    names = spec.split(",")
    policies = {a: REGISTRY[names[i % len(names)]](cfg, a, ids) for i, a in enumerate(ids)}
    return Referee(cfg, policies).run()


def test_smoke_runs_and_scores():
    cfg = GameConfig(agent_ids=["A", "B"], horizon_mode="fixed", n_rounds=1)
    res = _run(cfg, "bayesian_solo")
    assert len(res.rounds) == 1
    rr = res.rounds[0]
    for a in cfg.agent_ids:
        assert rr.estimates[a] is not None
        assert rr.errors[a] >= 0
        assert rr.rewards[a] >= 0


def test_determinism():
    cfg = GameConfig(agent_ids=["A", "B", "C"], horizon_mode="fixed", n_rounds=3, seed=42)
    r1 = _run(cfg, "honest_cooperator,bayesian_solo,hoarder")
    r2 = _run(cfg, "honest_cooperator,bayesian_solo,hoarder")
    assert [r.truth for r in r1.rounds] == [r.truth for r in r2.rounds]
    assert r1.rounds[-1].errors == r2.rounds[-1].errors


def test_carryover_and_reward():
    # A perfect-ish agent should end with more credits than it started, via reward.
    cfg = GameConfig(agent_ids=["A", "B"], horizon_mode="fixed", n_rounds=2,
                     carryover=True, seed=1)
    res = _run(cfg, "bayesian_solo")
    # rewards were issued and credits are tracked coherently
    assert all(res.states[a].credits >= 0 for a in cfg.agent_ids)
    assert any(r.rewards["A"] > 0 for r in res.rounds)


def test_liar_deception_is_flagged():
    cfg = GameConfig(agent_ids=["A", "B"], horizon_mode="fixed", n_rounds=2,
                     enable_trading=True, seed=3)
    res = _run(cfg, "liar,honest_cooperator")
    events = res.transcript.events
    dec = deception(events)
    assert dec["offers"] > 0, "liar should have made trade offers"
    assert dec["deception_rate"] == 1.0, "every liar sale is fabricated"


def test_cooperation_index_positive_for_cooperators():
    cfg = GameConfig(agent_ids=["A", "B"], horizon_mode="fixed", n_rounds=2, seed=5)
    res = _run(cfg, "honest_cooperator")
    coop = cooperation(res.transcript.events)
    assert coop["measurements"] > 0
    assert coop["cooperation_index"] > 0, "cooperators broadcast their measurements"


def test_summary_is_complete():
    cfg = GameConfig(agent_ids=["A", "B", "C", "D"], seed=9)
    res = _run(cfg, "honest_cooperator,bayesian_solo,liar,hoarder")
    s = summary(res.transcript.events)
    for key in ("deception", "cooperation", "welfare", "gini_final_credits",
                "survivors", "regret_by_round"):
        assert key in s


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all referee tests pass")
