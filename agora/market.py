"""Credits, escrow and the contract invariants.

This module is the "checks and balances" layer the design calls for. Every
credit movement goes through here, so the following invariants hold by
construction and are asserted in ``tests/test_market.py``:

  I1  No overdraft: an agent can never spend credits it does not have.
  I2  Conservation: credits are neither created nor destroyed by a transfer
      or a trade (only ``measure`` costs and reward payouts change the total).
  I3  Atomic settlement: a trade either moves the price AND delivers the
      payload, or neither happens.
  I4  No overselling: because payment is drawn from the buyer's *current*
      balance, an agent cannot commit the same credits to two settlements.

Note the deliberate gap: escrow guarantees *payment*, never *information
quality*. A seller may deliver a fabricated ``claimed_value`` — that is the
whole point of the study.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .types import AgentState, Trade


class MarketError(Exception):
    """Raised when an action would violate a contract invariant."""


class Market:
    def __init__(self, states: Dict[str, AgentState], counter: Callable[[], str]):
        self.states = states
        self._counter = counter                # supplies unique trade ids
        self.trades: Dict[str, Trade] = {}
        self.ledger: List[Dict] = []           # audit log of every credit move

    # --- primitives --------------------------------------------------------
    def _record(self, kind: str, src: Optional[str], dst: Optional[str],
                amount: float, **extra) -> None:
        entry = {"kind": kind, "src": src, "dst": dst, "amount": amount}
        entry.update(extra)
        self.ledger.append(entry)

    def _live(self, agent_id: str) -> AgentState:
        st = self.states.get(agent_id)
        if st is None:
            raise MarketError(f"unknown agent {agent_id!r}")
        if not st.alive:
            raise MarketError(f"agent {agent_id!r} is not alive")
        return st

    def _move(self, src: str, dst: str, amount: float) -> None:
        """Atomic credit transfer. Enforces I1 and (via single deduction) I4."""
        if amount < 0:
            raise MarketError("amount must be non-negative")
        s, d = self._live(src), self._live(dst)
        if not s.can_afford(amount):
            raise MarketError(
                f"{src} cannot afford {amount} (has {s.credits})"
            )
        s.credits -= amount
        d.credits += amount

    # --- spend (measurement cost, survival cost: credits leave the economy) --
    def spend(self, agent_id: str, amount: float, kind: str) -> None:
        st = self._live(agent_id)
        if not st.can_afford(amount):
            raise MarketError(f"{agent_id} cannot afford {amount} (has {st.credits})")
        st.credits -= amount
        self._record(kind, agent_id, None, amount)

    # --- transfer_credits (gifts / cost-splitting) -------------------------
    def transfer(self, src: str, dst: str, amount: float) -> None:
        if src == dst:
            raise MarketError("cannot transfer to self")
        self._move(src, dst, amount)
        self._record("transfer", src, dst, amount)

    # --- trading: propose / respond (escrowed, atomic) ---------------------
    def propose_trade(self, seller: str, buyer: str, price: float,
                      claimed_value: float, tick: int) -> Trade:
        self._live(seller)
        self._live(buyer)
        if seller == buyer:
            raise MarketError("cannot trade with self")
        if price < 0:
            raise MarketError("price must be non-negative")
        trade = Trade(
            trade_id=self._counter(),
            seller=seller,
            buyer=buyer,
            price=price,
            claimed_value=claimed_value,
            tick=tick,
        )
        self.trades[trade.trade_id] = trade
        return trade

    def respond_trade(self, responder: str, trade_id: str, accept: bool) -> Trade:
        trade = self.trades.get(trade_id)
        if trade is None:
            raise MarketError(f"unknown trade {trade_id!r}")
        if trade.buyer != responder:
            raise MarketError(f"{responder} is not the buyer of {trade_id}")
        if trade.status != "pending":
            raise MarketError(f"trade {trade_id} already {trade.status}")

        if not accept:
            trade.status = "rejected"
            self._record("trade_rejected", trade.seller, trade.buyer, trade.price,
                         trade_id=trade_id)
            return trade

        # Atomic settlement (I3): payment first (may raise -> nothing delivered),
        # then delivery. The caller delivers the payload only on success.
        self._move(trade.buyer, trade.seller, trade.price)
        trade.status = "accepted"
        buyer_state = self.states[trade.buyer]
        buyer_state.purchased.append({
            "seller": trade.seller,
            "claimed_value": trade.claimed_value,
            "price": trade.price,
            "trade_id": trade_id,
        })
        self._record("trade_settled", trade.buyer, trade.seller, trade.price,
                     trade_id=trade_id, claimed_value=trade.claimed_value)
        return trade

    def total_credits(self) -> float:
        return sum(st.credits for st in self.states.values())
