"""Contract-invariant tests for the escrow/ledger.

These pin down the "checks and balances" the design requires: no overdraft,
conservation, atomic settlement, no double-spend. Runnable with pytest or as a
plain script (`python tests/test_market.py`).
"""
from __future__ import annotations

import os
import sys

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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all market invariants hold")
