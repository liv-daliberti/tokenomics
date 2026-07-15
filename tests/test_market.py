"""Contract-invariant tests for the escrow/ledger.

These pin down the "checks and balances" the design requires: no overdraft,
conservation, atomic settlement, no double-spend. Runnable with pytest or as a
plain script (`python tests/test_market.py`).
"""
from __future__ import annotations

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agora.market import Market, MarketError
from agora.types import AgentState


def _market(balances):
    states = {a: AgentState(a, c, tau=100.0, messages_left=10) for a, c in balances.items()}
    seq = {"n": 0}

    def counter():
        seq["n"] += 1
        return f"T{seq['n']}"

    return Market(states, counter), states


def test_no_overdraft():
    m, st = _market({"A": 3.0, "B": 0.0})
    try:
        m.transfer("A", "B", 5.0)
        assert False, "overdraft should raise"
    except MarketError:
        pass
    assert st["A"].credits == 3.0 and st["B"].credits == 0.0  # unchanged


def test_transfer_conserves_credits():
    m, st = _market({"A": 3.0, "B": 1.0})
    before = m.total_credits()
    m.transfer("A", "B", 2.0)
    assert st["A"].credits == 1.0 and st["B"].credits == 3.0
    assert abs(m.total_credits() - before) < 1e-9


def test_spend_leaves_economy():
    m, st = _market({"A": 2.0})
    m.spend("A", 1.0, "measure")
    assert st["A"].credits == 1.0
    try:
        m.spend("A", 5.0, "measure")
        assert False
    except MarketError:
        pass


def test_no_transfer_same_agent():
    m, _ = _market({"A": 5.0})
    try:
        m.transfer("A", "A", 1.0)
        assert False
    except MarketError:
        pass


def test_atomic_trade_settlement():
    m, st = _market({"seller": 0.0, "buyer": 5.0})
    tr = m.propose_trade("seller", "buyer", price=2.0, claimed_value=123.4, tick=0)
    m.respond_trade("buyer", tr.trade_id, accept=True)
    assert st["buyer"].credits == 3.0 and st["seller"].credits == 2.0
    assert st["buyer"].purchased[0]["claimed_value"] == 123.4  # delivered on payment


def test_trade_no_funds_delivers_nothing():
    m, st = _market({"seller": 0.0, "buyer": 1.0})
    tr = m.propose_trade("seller", "buyer", price=5.0, claimed_value=99.0, tick=0)
    try:
        m.respond_trade("buyer", tr.trade_id, accept=True)
        assert False, "underfunded accept must raise"
    except MarketError:
        pass
    assert st["buyer"].credits == 1.0 and st["seller"].credits == 0.0
    assert st["buyer"].purchased == []  # nothing delivered (atomicity)


def test_no_double_spend():
    # A buyer with 5 credits cannot honour two 3-credit trades.
    m, st = _market({"s1": 0.0, "s2": 0.0, "buyer": 5.0})
    t1 = m.propose_trade("s1", "buyer", 3.0, 10.0, 0)
    t2 = m.propose_trade("s2", "buyer", 3.0, 20.0, 0)
    m.respond_trade("buyer", t1.trade_id, True)
    assert st["buyer"].credits == 2.0
    try:
        m.respond_trade("buyer", t2.trade_id, True)
        assert False, "second trade should overdraw"
    except MarketError:
        pass
    assert st["buyer"].credits == 2.0  # unchanged after failed settlement


def test_cannot_respond_twice():
    m, _ = _market({"s": 0.0, "b": 5.0})
    t = m.propose_trade("s", "b", 1.0, 1.0, 0)
    m.respond_trade("b", t.trade_id, True)
    try:
        m.respond_trade("b", t.trade_id, True)
        assert False
    except MarketError:
        pass


def test_non_buyer_cannot_respond():
    m, _ = _market({"s": 0.0, "b": 5.0, "c": 5.0})
    t = m.propose_trade("s", "b", 1.0, 1.0, 0)
    try:
        m.respond_trade("c", t.trade_id, True)
        assert False
    except MarketError:
        pass


def test_nonfinite_money_and_trade_values_are_rejected():
    m, st = _market({"A": 5.0, "B": 5.0})
    before = {a: s.credits for a, s in st.items()}
    for amount in (math.nan, math.inf, -math.inf):
        try:
            m.transfer("A", "B", amount)
            assert False, "non-finite transfer should raise"
        except MarketError:
            pass
    for price, value in ((math.nan, 1.0), (1.0, math.inf)):
        try:
            m.propose_trade("A", "B", price, value, 0)
            assert False, "non-finite trade should raise"
        except MarketError:
            pass
    assert {a: s.credits for a, s in st.items()} == before


def test_require_paid_trades_rejects_free_offers_only():
    # With the rule on, a free gift (price 0) is a contract violation, but ANY
    # strictly positive price — including a fraction of a credit — stands.
    # Default off changes nothing.
    states = {a: AgentState(a, 5.0, tau=100.0, messages_left=10) for a in "AB"}
    seq = {"n": 0}

    def counter():
        seq["n"] += 1
        return f"T{seq['n']}"

    m = Market(states, counter, require_paid=True)
    try:
        m.propose_trade("A", "B", 0.0, 42.0, 0)
        assert False, "free offer should raise"
    except MarketError as exc:
        assert "greater than 0" in str(exc)
    assert m.propose_trade("A", "B", 0.25, 42.0, 0).price == 0.25  # sub-credit ok
    assert m.propose_trade("A", "B", 1.0, 42.0, 1).price == 1.0

    m0, _ = _market({"A": 5.0, "B": 5.0})
    assert m0.propose_trade("A", "B", 0.0, 42.0, 0).price == 0.0  # rule off


def test_min_trade_price_floor():
    # A price floor rejects anything below it (the price-sweep knob); at/above ok.
    states = {a: AgentState(a, 100.0, tau=100.0, messages_left=10) for a in "AB"}
    seq = {"n": 0}

    def counter():
        seq["n"] += 1
        return f"T{seq['n']}"

    m = Market(states, counter, min_price=8.0)
    for bad in (0.0, 4.0, 7.99):
        try:
            m.propose_trade("A", "B", bad, 42.0, 0)
            assert False, f"price {bad} below the 8-credit floor should raise"
        except MarketError as exc:
            assert "at least 8" in str(exc)
    assert m.propose_trade("A", "B", 8.0, 42.0, 0).price == 8.0
    assert m.propose_trade("A", "B", 20.0, 42.0, 1).price == 20.0


def test_redaction_masks_digits_and_number_words():
    from agora.referee import _redact_numbers
    out = _redact_numbers("my reading is 480.5, i.e. four hundred eighty and a half")
    assert "480" not in out and "four" not in out.lower()
    assert "hundred" not in out.lower() and "eighty" not in out.lower()
    assert "half" not in out.lower()
    # ordinary words that merely CONTAIN number words survive
    ok = _redact_numbers("someone should tell everyone something")
    assert ok == "someone should tell everyone something"


def test_trade_only_and_paid_rules_are_announced_to_agents():
    from agora.config import GameConfig
    from agora.tools import system_prompt, tool_schemas
    cfg = GameConfig(agent_ids=["A", "B"], values_via_trade_only=True,
                     require_paid_trades=True)
    text = system_prompt(cfg, "A", ["B"])
    assert "CENSORED" in text and "GREATER than 0" in text
    schemas = {s["function"]["name"]: s["function"]["description"]
               for s in tool_schemas(cfg)}
    assert "CENSORED" in schemas["send_message"]
    assert "greater than 0" in schemas["propose_trade"]
    # defaults keep the original wording
    loose = system_prompt(GameConfig(agent_ids=["A", "B"]), "A", ["B"])
    assert "0 to give it away" in loose


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all market invariants hold")
