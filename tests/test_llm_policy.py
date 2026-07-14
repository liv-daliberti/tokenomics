"""Exercise the LLM policy path with a MockBackend (no model server needed).

This validates the piece that can't be checked by the scripted baselines: the
OpenAI-format assistant/tool message plumbing (an assistant turn carrying
tool_calls, followed by one tool result per tool_call_id), driven through the
referee's inner tool-call loop.
"""
from __future__ import annotations

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agora.backends import LLMResponse, MockBackend, RawToolCall
from agora.config import GameConfig
from agora.policies import LLMPolicy
from agora.referee import Referee
from agora.tools import parse_action


def test_nonfinite_tool_numbers_are_parse_failures():
    for name, args in (
        ("submit_estimate", {"value": math.nan}),
        ("transfer_credits", {"to": "B", "amount": math.inf}),
        ("propose_trade", {"to": "B", "price": 1, "claimed_value": -math.inf}),
    ):
        try:
            parse_action(name, args)
            assert False, f"{name} accepted a non-finite number"
        except ValueError as exc:
            assert "finite" in str(exc)


def make_backend(cfg):
    """Measure once, then submit the prior mean and end the turn."""
    def script(messages, tools, _cfg):
        n_assistant = sum(1 for m in messages if m["role"] == "assistant")
        if n_assistant == 0:
            return LLMResponse(content=None,
                               tool_calls=[RawToolCall("c1", "measure", {})])
        return LLMResponse(content=None, tool_calls=[
            RawToolCall("c2", "submit_estimate", {"value": cfg.prior_mu}),
            RawToolCall("c3", "end_turn", {}),
        ])
    return MockBackend(script)


def test_llm_policy_round_trip():
    cfg = GameConfig(agent_ids=["A", "B"], horizon_mode="fixed", n_rounds=1, seed=0)
    backend = make_backend(cfg)
    policies = {a: LLMPolicy(backend, cfg, a, [p for p in cfg.agent_ids if p != a])
                for a in cfg.agent_ids}
    res = Referee(cfg, policies).run()

    rr = res.rounds[0]
    for a in cfg.agent_ids:
        assert rr.estimates[a] == cfg.prior_mu           # submit_estimate landed
    # each agent measured exactly once -> spent one credit before reward
    measures = [e for e in res.transcript.events if e["event"] == "measure"]
    assert len(measures) == len(cfg.agent_ids)
    # the running chat history is well-formed: every tool_call has a tool reply
    for pol in policies.values():
        tool_call_ids = [tc["id"] for m in pol.messages
                         if m["role"] == "assistant" for tc in m.get("tool_calls", [])]
        tool_reply_ids = [m["tool_call_id"] for m in pol.messages if m["role"] == "tool"]
        assert sorted(tool_call_ids) == sorted(tool_reply_ids)


def test_llm_reasoning_is_captured():
    cfg = GameConfig(agent_ids=["A"], horizon_mode="fixed", n_rounds=1, seed=0)

    def script(messages, tools, _cfg):
        n = sum(1 for m in messages if m["role"] == "assistant")
        if n == 0:
            return LLMResponse(content="I'll measure to cut the noise.",
                               reasoning="thinking: one sample is not enough",
                               tool_calls=[RawToolCall("m", "measure", {})])
        return LLMResponse(content="Answering now.",
                           tool_calls=[RawToolCall("e", "end_turn", {})])

    pol = LLMPolicy(MockBackend(script), cfg, "A", [])
    res = Referee(cfg, {"A": pol}).run()
    texts = [e["text"] for e in res.transcript.events if e["event"] == "reasoning"]
    assert any("cut the noise" in t for t in texts)        # spoken content captured
    assert any("one sample is not enough" in t for t in texts)  # <think> captured too


def test_llm_parse_failure_is_counted():
    cfg = GameConfig(agent_ids=["A"], horizon_mode="fixed", n_rounds=1, seed=0)

    def script(messages, tools, _cfg):
        n = sum(1 for m in messages if m["role"] == "assistant")
        if n == 0:  # a malformed call: send_message missing 'text'
            return LLMResponse(None, [RawToolCall("b1", "send_message", {"to": "A"})])
        return LLMResponse(None, [RawToolCall("b2", "end_turn", {})])

    pol = LLMPolicy(MockBackend(script), cfg, "A", [])
    res = Referee(cfg, {"A": pol}).run()
    assert pol.parse_failures >= 1
    assert any(e["event"] == "parse_fail" for e in res.transcript.events)


def test_agent_memories_are_isolated():
    # Each agent keeps its OWN conversation. Sharing one (stateless) backend must
    # NOT let one agent's system prompt, observations, or generated output leak
    # into another agent's history — A only ever sees A's memory, B only B's.
    cfg = GameConfig(agent_ids=["A", "B"], horizon_mode="fixed", n_rounds=1, seed=0)

    def script(messages, tools, _cfg):
        # the backend only ever receives the CALLING agent's own history
        who = "A" if "You are agent A" in messages[0]["content"] else "B"
        n = sum(1 for m in messages if m["role"] == "assistant")
        if n == 0:
            return LLMResponse(content=f"SECRET-{who}",
                               tool_calls=[RawToolCall("m", "measure", {})])
        return LLMResponse(content=f"SECRET-{who}", tool_calls=[
            RawToolCall("s", "submit_estimate", {"value": 500.0}),
            RawToolCall("e", "end_turn", {})])

    backend = MockBackend(script)                      # ONE backend, shared (as in production)
    policies = {a: LLMPolicy(backend, cfg, a, [p for p in cfg.agent_ids if p != a])
                for a in cfg.agent_ids}
    # independent history objects from the very start (no shared mutable default)
    assert policies["A"].messages is not policies["B"].messages

    Referee(cfg, policies).run()
    a_hist = "\n".join(str(m.get("content", "")) for m in policies["A"].messages)
    b_hist = "\n".join(str(m.get("content", "")) for m in policies["B"].messages)

    # each agent's system prompt addresses only itself
    assert "You are agent A" in a_hist and "You are agent B" not in a_hist
    assert "You are agent B" in b_hist and "You are agent A" not in b_hist
    # generated content never crosses over
    assert "SECRET-A" in a_hist and "SECRET-A" not in b_hist
    assert "SECRET-B" in b_hist and "SECRET-B" not in a_hist


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all llm-policy tests pass")
