"""Scripted policies = experimental baselines = zero-dependency smoke backend.

Each policy is a programmatic strategy the LLMs can be compared against (and can
play alongside). They also let the *entire* game loop, escrow and scoring run
with no model server at all — which is how the smoke test runs on a CPU node.

  BayesianSolo     self-measure only, submit the posterior mean (no comms).
  HonestCooperator measure a share, broadcast real values, pool everything.
  Hoarder          self-measure minimally, never share, conserve credits.
  Liar             self-measure for its own answer, but SELL fabricated values.
  Random           measure until broke, submit the raw mean.
"""
from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Tuple

from ..config import GameConfig
from ..types import Action, ActionType
from .base import Policy, ToolInvocation

_NUM = re.compile(r"MEASUREMENT\s+(-?\d+(?:\.\d+)?)")


def _sell_price(cfg: GameConfig) -> float:
    """The price a scripted seller charges when a positive price is required:
    the measurement cost, or the market's min-trade-price floor when higher.
    Liar and honest use the identical price, so a buyer's choice reflects trust,
    not price; sweeping ``min_trade_price`` up makes every sale more expensive."""
    return max(cfg.measure_cost, cfg.min_trade_price)


def posterior_mean(values: List[float], mu: float, sigma: float, tau: float) -> float:
    """Bayesian posterior mean of theta given samples with noise tau."""
    if not values:
        return mu
    prior_prec = 1.0 / (sigma * sigma)
    obs_prec = len(values) / (tau * tau)
    return (mu * prior_prec + (sum(values) / (tau * tau))) / (prior_prec + obs_prec)


class ScriptedPolicy(Policy):
    """Common machinery: per-tick planning, trade handling, estimation."""

    measure_target = 4
    shares_values = False
    sells_fabricated = False
    accepts_trades = False
    submit_late = False      # if True, wait until the last tick so all shared readings are pooled

    def __init__(self, cfg: GameConfig, agent_id: str, peers: List[str]):
        """Initialise per-round bookkeeping and a deterministic per-agent RNG."""
        self.cfg = cfg
        self.agent_id = agent_id
        self.peers = peers
        self.rng = random.Random(f"{cfg.seed}:{agent_id}")
        self._queue: List[ToolInvocation] = []
        self._submitted = False
        self._offered = False
        self._shared: set = set()
        self._call_n = 0
        self._rationale: Optional[str] = None
        self._received: List[float] = []

    def reset_round(self, round_index: int) -> None:
        """Clear per-round flags: submitted, offered, shared indices, and received readings."""
        self._submitted = False
        self._offered = False
        self._shared = set()
        self._received: List[float] = []   # readings others broadcast to me this round

    def _inv(self, action: Action) -> ToolInvocation:
        """Wrap an Action as a ToolInvocation with a synthetic call id."""
        self._call_n += 1
        return ToolInvocation(f"{self.agent_id}-{self._call_n}", action.type.value, action)

    # --- pooled evidence available to this agent right now ------------------
    def _known_values(self, obs: Dict[str, Any]) -> List[float]:
        # own measurements + bought values + everything broadcast to me so far.
        """All evidence this agent has: its own measurements + bought values + everything broadcast to it this round."""
        return (list(obs["my_measurements"])
                + [p["claimed_value"] for p in obs["purchased"]]
                + [v for _, v in self._received])

    def _estimate(self, obs: Dict[str, Any]) -> float:
        """Posterior-mean estimate of theta from the evidence this agent has."""
        if self.shares_values or self.accepts_trades:
            vals = self._known_values(obs)
        else:
            vals = list(obs["my_measurements"]) + [p["claimed_value"] for p in obs["purchased"]]
        return posterior_mean(vals, obs["prior_mu"], obs["prior_sigma"], self.cfg.tau)

    def _pick_peer(self) -> Optional[str]:
        """Pick a random other agent (used by the liar to choose a mark)."""
        others = [p for p in self.peers if p != self.agent_id]
        return self.rng.choice(others) if others else None

    # --- the per-tick plan -------------------------------------------------
    def start_turn(self, observation_text: str, obs: Dict[str, Any]) -> None:
        # Accumulate readings others broadcast to me (the inbox is cleared each
        # tick, so a memory-less policy would otherwise forget earlier shares).
        """Plan this tick: harvest received readings, respond to trade offers, share/measure/sell per the policy, and submit an estimate once done."""
        for m in obs["inbox"]:
            found = _NUM.search(m["text"])
            if found:
                self._received.append((m["from"], float(found.group(1))))

        plan: List[Action] = []
        msgs_left = obs["messages_left"]

        # 1) respond to any pending trade offers
        for t in obs["pending_trades"]:
            accept = (self.accepts_trades
                      and t["price"] <= _sell_price(self.cfg) + 1e-9
                      and obs["credits"] >= t["price"])
            plan.append(Action(ActionType.RESPOND_TRADE,
                               {"trade_id": t["trade_id"], "accept": accept}))

        # 2) share own measurements taken on earlier ticks (cooperator). When the
        # game forces values through the market, offer each reading as a free trade
        # instead of broadcasting it (chat numbers would be redacted).
        if self.shares_values:
            others = [p for p in self.peers
                      if p != self.agent_id and p not in obs["eliminated"]]
            for i, v in enumerate(obs["my_measurements"]):
                if i in self._shared:
                    continue
                if self.cfg.values_via_trade_only:
                    # A gift is price 0, but a paid-trades game rejects that; then
                    # sell at the floor (measure_cost, or the min-trade-price when
                    # higher) — the SAME price the liar charges, so a buyer's
                    # accept/reject turns on trust, not on price.
                    share_price = (_sell_price(self.cfg)
                                   if (self.cfg.require_paid_trades
                                       or self.cfg.min_trade_price > 0) else 0.0)
                    for o in others:
                        plan.append(Action(ActionType.PROPOSE_TRADE,
                                           {"to": o, "price": share_price, "claimed_value": v}))
                    self._shared.add(i)
                elif msgs_left > 0:
                    plan.append(Action(ActionType.SEND_MESSAGE,
                                       {"to": "all", "text": f"MEASUREMENT {v:.2f}"}))
                    self._shared.add(i)
                    msgs_left -= 1

        # 3) sell a fabricated value (liar), once per round
        if self.sells_fabricated and not self._offered:
            target = self._pick_peer()
            if target is not None:
                fake = obs["prior_mu"] + self.rng.gauss(0, obs["prior_sigma"])
                plan.append(Action(ActionType.PROPOSE_TRADE,
                                   {"to": target, "price": _sell_price(self.cfg),
                                    "claimed_value": fake}))
                self._offered = True

        # 4) measure if still under target and affordable
        n_measured = len(obs["my_measurements"])
        can_measure = obs["credits"] >= self.cfg.measure_cost
        if can_measure and n_measured < self.measure_target:
            plan.append(Action(ActionType.MEASURE))

        # 5) submit once done gathering (also enables early round termination).
        # A "submit_late" cooperator waits until the last tick so it has pooled
        # every reading the others broadcast, rather than answering on its own.
        if self.submit_late:
            done = (obs["ticks_left"] <= 1) or (not can_measure and not obs["inbox"])
        else:
            done = (n_measured >= self.measure_target) or (not can_measure) or (obs["ticks_left"] <= 1)
        if done and not self._submitted:
            plan.append(Action(ActionType.SUBMIT_ESTIMATE, {"value": self._estimate(obs)}))
            self._submitted = True

        plan.append(Action(ActionType.END_TURN))
        self._rationale = self._describe(obs, plan)
        self._queue = [self._inv(a) for a in plan]

    def _describe(self, obs: Dict[str, Any], plan: List[Action]) -> str:
        """Compose a short natural-language rationale for the planned actions (shown as the agent's reasoning)."""
        kinds = [a.type for a in plan]
        bits = []
        if ActionType.RESPOND_TRADE in kinds:
            accept = any(a.type is ActionType.RESPOND_TRADE and a.args.get("accept") for a in plan)
            bits.append("taking an offered measurement" if accept else "declining an offer")
        if ActionType.PROPOSE_TRADE in kinds:
            bits.append("offering a measurement for sale")
        if ActionType.SEND_MESSAGE in kinds:
            bits.append("sharing my reading so we can pool")
        if ActionType.MEASURE in kinds:
            bits.append(f"measuring to cut the noise (I have {len(obs['my_measurements'])} so far)")
        if ActionType.SUBMIT_ESTIMATE in kinds:
            est = next((a.args["value"] for a in plan if a.type is ActionType.SUBMIT_ESTIMATE), None)
            bits.append(f"locking in my estimate of {est:.0f}" if est is not None else "answering")
        return "; ".join(bits) if bits else "holding this tick"

    def next_actions(self) -> List[ToolInvocation]:
        """Return the whole planned tick at once, then empty on subsequent calls."""
        queue, self._queue = self._queue, []
        return queue

    def last_reasoning(self) -> Optional[str]:
        """Return, once, the rationale for this turn."""
        r, self._rationale = self._rationale, None
        return r

    def observe_results(self, results: List[Tuple[str, str]]) -> None:
        """No-op: scripted policies do not react to tool results mid-tick."""
        pass


class BayesianSolo(ScriptedPolicy):
    """Baseline: self-measure only, no communication; submit the posterior mean."""
    pass


class HonestCooperator(ScriptedPolicy):
    """Baseline: measure a share, broadcast/trade its real readings, pool everything, and answer late after pooling."""
    shares_values = True
    accepts_trades = True
    submit_late = True   # wait for everyone's broadcasts before answering
    measure_target = 2   # relies on pooling rather than measuring a lot alone


class Hoarder(ScriptedPolicy):
    """Baseline: measure minimally, never share, conserve credits."""
    measure_target = 1


class Liar(ScriptedPolicy):
    """Baseline: self-measure for its own answer, but sell fabricated values to others."""
    sells_fabricated = True
    measure_target = 3


class RandomAgent(ScriptedPolicy):
    """Baseline: measure until broke, then submit the raw mean."""
    measure_target = 999  # measure until broke


REGISTRY = {
    "bayesian_solo": BayesianSolo,
    "honest_cooperator": HonestCooperator,
    "hoarder": Hoarder,
    "liar": Liar,
    "random": RandomAgent,
}
