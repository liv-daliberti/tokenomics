"""Markdown-memory mode (no model server needed).

The contract under test: with ``memory="markdown"`` every agent journals each
round (a text-only model call the referee logs as a ``notes`` event), the
conversation is RESET to the system prompt at each game boundary, and the
accumulated notebook — nothing else — is what the agent gets back. With the
default ``memory="context"`` nothing changes: no notes, history kept.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agora.backends import LLMResponse, MockBackend, RawToolCall
from agora.config import GameConfig
from agora.policies import LLMPolicy
from agora.referee import run_match
from agora.tools import system_prompt
from agora.transcripts import Transcript

NOTE_TEXT = "- measured a value and answered the prior mean; partner was quiet"


def make_backend(cfg):
    """Measure on a fresh observation, then answer; journal when asked.

    The note-writing step is recognisable mechanically: it is the only call
    made with NO tools.
    """
    def script(messages, tools, _cfg):
        if not tools:
            return LLMResponse(content=NOTE_TEXT)
        if messages[-1]["role"] == "user":
            return LLMResponse(content=None,
                               tool_calls=[RawToolCall("cm", "measure", {})])
        return LLMResponse(content=None, tool_calls=[
            RawToolCall("cs", "submit_estimate", {"value": cfg.prior_mu}),
            RawToolCall("ce", "end_turn", {}),
        ])
    return MockBackend(script)


def _play(memory: str, n_games: int = 2, n_rounds: int = 2):
    cfg = GameConfig(agent_ids=["A", "B"], horizon_mode="fixed",
                     n_rounds=n_rounds, max_ticks=1, seed=0, memory=memory)
    backend = make_backend(cfg)
    policies = {a: LLMPolicy(backend, cfg, a,
                             [p for p in cfg.agent_ids if p != a], n_games=n_games)
                for a in cfg.agent_ids}
    tx = Transcript()
    run_match(cfg, policies, n_games, tx)
    return cfg, policies, tx


def test_markdown_mode_journals_every_round_and_resets_context():
    cfg, policies, tx = _play("markdown")
    notes = [e for e in tx.events if e["event"] == "notes"]
    # 2 agents x 2 rounds x 2 games
    assert len(notes) == 8
    assert all(e["text"] == NOTE_TEXT for e in notes)
    assert {e["agent"] for e in notes} == {"A", "B"}

    pol = policies["A"]
    assert len(pol.notes) == 4                       # its own 2 rounds x 2 games
    doc = pol.notes_doc()
    assert "# Game 1" in doc and "# Game 2" in doc and "## Round 0" in doc

    # The context really was reset at the game-2 boundary: message 0 is the
    # system prompt and message 1 is game 2's first observation, carrying the
    # boundary marker plus the notebook — game 1's observations are gone.
    assert pol.messages[0]["role"] == "system"
    first_user = pol.messages[1]
    assert first_user["role"] == "user"
    assert "A NEW GAME (game 2 of 2)" in first_user["content"]
    assert "notes you wrote" in first_user["content"]
    assert NOTE_TEXT in first_user["content"]

    # ...and the transcript's logged prompt shows the SAME text the model got
    # (boundary + notebook), so the report's prompt panel is faithful.
    p2 = next(e for e in tx.events
              if e["event"] == "prompt" and e.get("game_index") == 1)
    assert "A NEW GAME (game 2 of 2)" in p2["text"]
    assert NOTE_TEXT in p2["text"]


def test_context_mode_is_unchanged():
    cfg, policies, tx = _play("context")
    assert not [e for e in tx.events if e["event"] == "notes"]
    pol = policies["A"]
    assert pol.notes == [] and pol.notes_doc() == ""
    # history spans both games: the game-2 marker appears deep in the log,
    # NOT in message 1, and it promises kept memory rather than notes
    boundary = next(m for m in pol.messages
                    if m["role"] == "user" and "A NEW GAME" in m["content"])
    assert "you remember everything" in boundary["content"]
    assert pol.messages.index(boundary) > 1


def test_system_prompt_announces_the_memory_rule():
    cfg = GameConfig(agent_ids=["A", "B"], memory="markdown")
    text = system_prompt(cfg, "A", ["B"], n_games=3)
    assert "markdown notes" in text and "CLEARED" in text
    ctx = system_prompt(cfg.with_(memory="context"), "A", ["B"], n_games=3)
    assert "you keep your memory" in ctx


def test_notes_render_in_the_report():
    from analysis.viz import render_simple, render_body
    _, _, tx = _play("markdown")
    simple = render_simple(tx.events, "t")
    assert "notebook" in simple and "📝" in simple and NOTE_TEXT in simple
    detailed = render_body(tx.events, "t")
    assert "round notes" in detailed


if __name__ == "__main__":
    test_markdown_mode_journals_every_round_and_resets_context()
    test_context_mode_is_unchanged()
    test_system_prompt_announces_the_memory_rule()
    test_notes_render_in_the_report()
    print("ok")
