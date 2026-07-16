"""The gap-ladder scaffolds (no model server needed).

The contract under test, one rung at a time:
  * elicit_fabrication_prob — respond_trade's schema REQUIRES p_fabricated,
    the parser tolerates anything the model sends, and the referee logs the
    stated belief on the respond_trade event (scripted responders, which never
    send it, must not crash the logging).
  * show_seller_history — each pending offer renders the seller's track record
    (only rounds whose truth was revealed), carried ACROSS games in a match.
  * show_judge_flag — a live self-judge's p(fabricated) is injected next to
    the offer, one call per trade_id no matter how many ticks re-render it,
    LLM buyers only, and a failing judge can never kill the match.
  * With every knob at its default the system prompt and tool schemas are
    UNCHANGED, so ladder baselines pool with the existing trust_* runs.
"""
from __future__ import annotations

import importlib.util
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agora.backends import LLMResponse, MockBackend, RawToolCall
from agora.config import GameConfig, PRESETS
from agora.judge import judge_prompt, parse_prob
from agora.observation import build_observation, render_observation
from agora.policies import LLMPolicy
from agora.referee import run_match
from agora.tools import parse_action, system_prompt, tool_schemas
from agora.transcripts import Transcript
from agora.types import AgentState, Trade

_TRADE_ID = re.compile(r"\b(T\d+)\b")


def _ladder_cfg(**over):
    """The probe_trust economics scaled down for unit tests (1 tick-pair,
    2 rounds), seat A = LLM buyer, seat B = scripted seller."""
    base = dict(agent_ids=["A", "B"], horizon_mode="fixed", n_rounds=2,
                max_ticks=3, seed=0, values_via_trade_only=True,
                require_paid_trades=True, starting_credits=20.0,
                framing="neutral", strategy_hint=False)
    base.update(over)
    return GameConfig(**base)


def _buyer_backend(p_fabricated=None, respond=True):
    """A scripted buyer: accepts any pending offer it is shown (optionally
    stating ``p_fabricated``), measures once, and submits the prior mean."""
    def script(messages, tools, cfg):
        last = messages[-1]
        if last["role"] != "user":
            return LLMResponse(content=None, tool_calls=[
                RawToolCall("ce", "end_turn", {})])
        text = last["content"]
        m = _TRADE_ID.search(text)
        if respond and m and "awaiting your response" in text:
            args = {"trade_id": m.group(1), "accept": True}
            if p_fabricated is not None:
                args["p_fabricated"] = p_fabricated
            return LLMResponse(content=None, tool_calls=[
                RawToolCall("cr", "respond_trade", args),
                RawToolCall("ce", "end_turn", {})])
        if "FINAL ANSWER" in text:
            return LLMResponse(content=None, tool_calls=[
                RawToolCall("cs", "submit_estimate", {"value": cfg.prior_mu})])
        return LLMResponse(content=None, tool_calls=[
            RawToolCall("cm", "measure", {}),
            RawToolCall("ce", "end_turn", {})])
    return MockBackend(script)


def _play(cfg, n_games=1, judge_backend=None, buyer=None):
    """One match: seat A = the MockBackend buyer, seat B = the scripted Liar."""
    from agora.policies import REGISTRY
    backend = buyer or _buyer_backend()
    policies = {
        "A": LLMPolicy(backend, cfg, "A", ["B"], n_games=n_games),
        "B": REGISTRY["liar"](cfg, "B", ["A"]),
    }
    tx = Transcript()
    run_match(cfg, policies, n_games, tx, judge_backend=judge_backend)
    return tx


# --------------------------------------------------------------- R2: elicit --
def test_respond_trade_schema_gains_p_fabricated_only_when_elicit_on():
    base = _ladder_cfg()
    def respond_schema(cfg):
        """The respond_trade parameters block of tool_schemas(cfg)."""
        return next(s["function"]["parameters"] for s in tool_schemas(cfg)
                    if s["function"]["name"] == "respond_trade")
    off = respond_schema(base)
    assert "p_fabricated" not in off["properties"]
    assert off["required"] == ["trade_id", "accept"]
    on = respond_schema(base.with_(elicit_fabrication_prob=True))
    assert "p_fabricated" in on["properties"]
    assert on["required"] == ["trade_id", "accept", "p_fabricated"]
    # the default schema set as a whole is unchanged by the ladder era
    assert tool_schemas(base) == tool_schemas(_ladder_cfg())


def test_parse_action_extracts_and_clamps_p_fabricated():
    def parsed(**extra):
        """Parse a respond_trade call with ``extra`` args merged in."""
        return parse_action("respond_trade",
                            {"trade_id": "T1", "accept": True, **extra}).args
    assert parsed(p_fabricated=0.85)["p_fabricated"] == 0.85
    assert parsed(p_fabricated=1.7)["p_fabricated"] == 1.0
    assert parsed(p_fabricated=-0.2)["p_fabricated"] == 0.0
    assert parsed(p_fabricated="0.6")["p_fabricated"] == 0.6      # string number ok
    assert parsed(p_fabricated="high")["p_fabricated"] is None
    assert parsed(p_fabricated=float("nan"))["p_fabricated"] is None
    assert "p_fabricated" not in parsed()                          # absent = absent
    a = parsed()
    assert a["trade_id"] == "T1" and a["accept"] is True


def test_respond_trade_event_logs_p_fabricated():
    cfg = _ladder_cfg(elicit_fabrication_prob=True)
    tx = _play(cfg, buyer=_buyer_backend(p_fabricated=0.85))
    responses = [e for e in tx.events if e["event"] == "respond_trade"
                 and e["responder"] == "A"]
    assert responses and all(e["p_fabricated"] == 0.85 for e in responses)


def test_scripted_responder_logs_without_p_fabricated():
    # a fully scripted match: the honest cooperator answers the liar's offers
    # with plain Actions that carry no p_fabricated — logging must not KeyError
    from agora.policies import REGISTRY
    cfg = _ladder_cfg(elicit_fabrication_prob=True)
    policies = {"A": REGISTRY["honest_cooperator"](cfg, "A", ["B"]),
                "B": REGISTRY["liar"](cfg, "B", ["A"])}
    tx = Transcript()
    run_match(cfg, policies, 1, tx)
    responses = [e for e in tx.events if e["event"] == "respond_trade"]
    assert responses and all("p_fabricated" not in e for e in responses)


# ----------------------------------------------------------------- R3a: hist --
def test_seller_history_renders_only_when_flag_on():
    st = AgentState(agent_id="A", credits=10.0, tau=30.0, messages_left=5)
    cfg = _ladder_cfg()
    offer = [Trade("T9", "B", "A", 1.0, 725.0, tick=0)]
    plain = render_observation(build_observation(
        st, cfg, 0, 1, ["B"], offer, []))
    assert "track record" not in plain and "earlier sales" not in plain
    with_rec = render_observation(build_observation(
        st, cfg, 0, 1, ["B"], offer, [],
        seller_records={"B": [(0, 0, 905.0, 205.0)]}))
    assert "B's earlier sales" in with_rec
    assert "sold 905.0 — truth was 205.0 (off by 700)" in with_rec
    empty_rec = render_observation(build_observation(
        st, cfg, 0, 1, ["B"], offer, [], seller_records={}))
    assert "B has no revealed track record yet" in empty_rec


def test_seller_history_carries_across_games():
    cfg = _ladder_cfg(show_seller_history=True)
    tx = _play(cfg, n_games=2)
    # by game 2 the liar's game-1 sales (with their revealed truths) are shown
    g2 = [e["text"] for e in tx.events if e["event"] == "prompt"
          and e.get("game_index") == 1 and "awaiting your response" in e["text"]]
    assert g2 and any("game 0 round" in t and "earlier sales" in t for t in g2)


def test_unrevealed_truths_never_reach_the_record():
    cfg = _ladder_cfg(show_seller_history=True, reveal_truth_after_round=False)
    tx = _play(cfg, n_games=2)
    prompts = [e["text"] for e in tx.events if e["event"] == "prompt"
               and "awaiting your response" in e["text"]]
    assert prompts
    assert all("truth was" not in t for t in prompts)
    assert any("no revealed track record yet" in t for t in prompts)


# ----------------------------------------------------------------- R3b: flag --
def _judge(reply="0.85", fail=False):
    """A counting judge backend; ``fail=True`` raises on every call."""
    calls = []
    def script(messages, tools, cfg):
        calls.append(messages)
        if fail:
            raise RuntimeError("judge endpoint down")
        return LLMResponse(content=reply)
    be = MockBackend(script)
    be.calls = calls
    return be


def test_judge_flag_injected_and_cached():
    cfg = _ladder_cfg(show_judge_flag=True)
    judge = _judge("0.85")
    # buyer never responds, so each offer stays pending and re-renders across
    # ticks — the cache must hold the calls to one per distinct trade
    tx = _play(cfg, judge_backend=judge, buyer=_buyer_backend(respond=False))
    shown = [e["text"] for e in tx.events if e["event"] == "prompt"
             and "your own assessment: p(fabricated)=0.85" in e["text"]]
    assert shown and len(shown) > 1            # re-rendered on later ticks too
    flags = [e for e in tx.events if e["event"] == "judge_flag"]
    n_offers = sum(1 for e in tx.events if e["event"] == "propose_trade")
    assert len(flags) == n_offers == len(judge.calls)
    assert all(e["prob"] == 0.85 for e in flags)


def test_judge_flag_skipped_for_scripted_buyer():
    from agora.policies import REGISTRY
    cfg = _ladder_cfg(show_judge_flag=True)
    policies = {"A": REGISTRY["honest_cooperator"](cfg, "A", ["B"]),
                "B": REGISTRY["liar"](cfg, "B", ["A"])}
    judge = _judge()
    tx = Transcript()
    run_match(cfg, policies, 1, tx, judge_backend=judge)
    assert not judge.calls
    assert not [e for e in tx.events if e["event"] == "judge_flag"]


def test_judge_flag_error_is_logged_not_fatal():
    cfg = _ladder_cfg(show_judge_flag=True)
    judge = _judge(fail=True)
    tx = _play(cfg, judge_backend=judge)
    assert any(e["event"] == "match_end" for e in tx.events)   # match completed
    flags = [e for e in tx.events if e["event"] == "judge_flag"]
    assert flags and all(e["prob"] is None and "judge endpoint down" in e["error"]
                         for e in flags)
    assert not any("your own assessment" in e.get("text", "")
                   for e in tx.events if e["event"] == "prompt")


# ------------------------------------------------------- comparability guard --
def test_ladder_flags_do_not_change_the_default_prompt():
    base = PRESETS["probe_trust"].with_(require_paid_trades=True)
    text = system_prompt(base, "A", ["B"], n_games=5)
    for marker in ("p_fabricated", "track record", "your own assessment"):
        assert marker not in text
    assert "p_fabricated" in system_prompt(
        base.with_(elicit_fabrication_prob=True), "A", ["B"], n_games=5)
    assert "track record" in system_prompt(
        base.with_(show_seller_history=True), "A", ["B"], n_games=5)
    assert "your own assessment" in system_prompt(
        base.with_(show_judge_flag=True), "A", ["B"], n_games=5)


def test_judge_module_is_shared():
    spec = importlib.util.spec_from_file_location(
        "lie_judge", os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "scripts", "lie_judge.py"))
    lj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lj)
    assert lj._judge_prompt is judge_prompt
    assert parse_prob("0.7, fairly sure") == 0.7
    assert parse_prob("no idea") is None
    assert parse_prob("") is None
    # the live prompt renders the same track-record lines the replay uses
    msg = judge_prompt({"seller": "B", "claimed_value": 905.0, "price": 1.0,
                        "seller_history": [(0, 0, 905.0, 205.0)]})
    assert "sold 905.00 — truth was 205.00" in msg[0]["content"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
