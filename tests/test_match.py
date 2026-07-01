"""A match = X games back-to-back with the agents' context persisting."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agora.backends import LLMResponse, MockBackend, RawToolCall
from agora.config import GameConfig
from agora.policies import REGISTRY, LLMPolicy
from agora.referee import run_match
from analysis.metrics import _games, summary
from analysis.viz import render_simple

_ACTION_EVENTS = {"measure", "message", "transfer", "propose_trade",
                  "respond_trade", "submit_estimate"}


def _actor(e):
    return e.get("agent") or e.get("sender") or e.get("src") or e.get("seller") or e.get("responder")


def test_match_runs_n_games_and_persists_llm_context():
    cfg = GameConfig(agent_ids=["A", "B"], horizon_mode="fixed", n_rounds=1, seed=0)

    def script(messages, tools, _cfg):
        # Trivially answer to keep the test fast; the point is context/plumbing.
        return LLMResponse(None, [
            RawToolCall("s", "submit_estimate", {"value": cfg.prior_mu}),
            RawToolCall("e", "end_turn", {}),
        ])

    backend = MockBackend(script)
    pols = {a: LLMPolicy(backend, cfg, a, [p for p in cfg.agent_ids if p != a])
            for a in cfg.agent_ids}
    mr = run_match(cfg, pols, n_games=3)

    ev = mr.transcript.events
    assert sum(1 for e in ev if e["event"] == "game_start") == 3
    assert mr.n_games == 3 and len(mr.games) == 3
    assert summary(ev)["n_games"] == 3

    for pol in pols.values():
        # Memory persists: ONE system prompt for the whole match, not one per game.
        assert sum(1 for m in pol.messages if m["role"] == "system") == 1
        # Two "new game" boundary markers (games 2 and 3; game 1 has none).
        markers = sum(1 for m in pol.messages
                      if m["role"] == "user" and "A NEW GAME" in m["content"])
        assert markers == 2


def test_each_game_uses_a_fresh_world():
    # Different per-game seeds -> different hidden truths across games.
    cfg = GameConfig(agent_ids=["A", "B"], horizon_mode="fixed", n_rounds=1, seed=0)
    pols = {a: REGISTRY["bayesian_solo"](cfg, a, cfg.agent_ids) for a in cfg.agent_ids}
    mr = run_match(cfg, pols, n_games=3)
    truths = [e["truth"] for e in mr.transcript.events if e["event"] == "round_start"]
    assert len(truths) == 3 and len(set(truths)) == 3  # a fresh draw each game


def test_eliminated_agents_sit_out_then_revive_next_game():
    # Hoarder keeps credits and survives the per-round survival cost; Random
    # spends itself to ruin and is eliminated for the rest of that game.
    cfg = GameConfig(agent_ids=["A", "B"], horizon_mode="fixed", n_rounds=4,
                     starting_credits=4.0, survival_cost=3.5, seed=1)
    pols = {"A": REGISTRY["hoarder"](cfg, "A", cfg.agent_ids),
            "B": REGISTRY["random"](cfg, "B", cfg.agent_ids)}
    mr = run_match(cfg, pols, n_games=3)

    a_death = False
    for gevs in _games(mr.transcript.events):
        # (1) every game starts with everyone revived
        first = next(e for e in gevs if e["event"] == "round_start")
        assert set(first["alive"]) == set(cfg.agent_ids), "agents must revive each game"
        # (2) once eliminated, an agent takes no further actions in that game
        dead = set()
        for e in gevs:
            if e["event"] in _ACTION_EVENTS:
                assert _actor(e) not in dead, f"eliminated agent {_actor(e)} acted"
            if e["event"] == "round_end":
                newly = {a for a, al in e["result"]["alive"].items() if not al}
                if newly - dead:
                    a_death = True
                dead |= newly
    assert a_death, "test setup should eliminate at least one agent"


def test_prompt_announces_match_length():
    from agora.tools import system_prompt
    assert "5 separate games" in system_prompt(GameConfig(), "A", ["B"], n_games=5)
    assert "separate games" not in system_prompt(GameConfig(), "A", ["B"], n_games=1)


def test_simple_view_is_game_aware():
    cfg = GameConfig(agent_ids=["A", "B"], horizon_mode="fixed", n_rounds=1, seed=1)
    pols = {a: REGISTRY["bayesian_solo"](cfg, a, cfg.agent_ids) for a in cfg.agent_ids}
    mr = run_match(cfg, pols, n_games=3)
    doc = render_simple(mr.transcript.events, "t")
    assert "Game 1 of 3" in doc and "Game 3 of 3" in doc
    assert "games in a row" in doc


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all match tests pass")
