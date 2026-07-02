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
        self._submitted = False
        self._offered = False
        self._shared = set()
        self._received: List[float] = []   # readings others broadcast to me this round

    def _inv(self, action: Action) -> ToolInvocation:
        self._call_n += 1
        return ToolInvocation(f"{self.agent_id}-{self._call_n}", action.type.value, action)

    # --- pooled evidence available to this agent right now ------------------
    def _known_values(self, obs: Dict[str, Any]) -> List[float]:
        # own measurements + bought values + everything broadcast to me so far.
        return (list(obs["my_measurements"])
                + [p["claimed_value"] for p in obs["purchased"]]
                + [v for _, v in self._received])

    def _estimate(self, obs: Dict[str, Any]) -> float:
        if obs.get("complementary"):
            return self._complementary_estimate(obs)
        if self.shares_values or self.accepts_trades:
            vals = self._known_values(obs)
        else:
            vals = list(obs["my_measurements"]) + [p["claimed_value"] for p in obs["purchased"]]
        return posterior_mean(vals, obs["prior_mu"], obs["prior_sigma"], self.cfg.tau)

    def _complementary_estimate(self, obs: Dict[str, Any]) -> float:
        # theta = my part + the other agents' parts. I estimate my part from my own
        # measurements, and each other agent's part from what THEY shared/sold me
        # (falling back to that part's prior mean if they shared nothing -> blind).
        mu_c, sig_c = obs["component_prior_mu"], obs["component_prior_sigma"]
        tau = self.cfg.tau
        by_sender: Dict[str, List[float]] = {}
        for s, v in self._received:
            by_sender.setdefault(s, []).append(v)
        for p in obs["purchased"]:
            by_sender.setdefault(p["seller"], []).append(p["claimed_value"])
        total = posterior_mean(list(obs["my_measurements"]), mu_c, sig_c, tau)
        for other in self.peers:
            if other == self.agent_id:
                continue
            total += posterior_mean(by_sender.get(other, []), mu_c, sig_c, tau)
        return total

    def _pick_peer(self) -> Optional[str]:
        others = [p for p in self.peers if p != self.agent_id]
        return self.rng.choice(others) if others else None

    # --- the per-tick plan -------------------------------------------------
    def start_turn(self, observation_text: str, obs: Dict[str, Any]) -> None:
        # Accumulate readings others broadcast to me (the inbox is cleared each
        # tick, so a memory-less policy would otherwise forget earlier shares).
        for m in obs["inbox"]:
            found = _NUM.search(m["text"])
            if found:
                self._received.append((m["from"], float(found.group(1))))

        plan: List[Action] = []
        msgs_left = obs["messages_left"]

        # 1) respond to any pending trade offers
        for t in obs["pending_trades"]:
            accept = (self.accepts_trades
                      and t["price"] <= self.cfg.measure_cost + 1e-9
                      and obs["credits"] >= t["price"])
            plan.append(Action(ActionType.RESPOND_TRADE,
                               {"trade_id": t["trade_id"], "accept": accept}))

        # 2) broadcast own measurements taken on earlier ticks (cooperator)
        if self.shares_values:
            for i, v in enumerate(obs["my_measurements"]):
                if i in self._shared or msgs_left <= 0:
                    continue
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
                                   {"to": target, "price": self.cfg.measure_cost,
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
        queue, self._queue = self._queue, []
        return queue

    def last_reasoning(self) -> Optional[str]:
        r, self._rationale = self._rationale, None
        return r

    def observe_results(self, results: List[Tuple[str, str]]) -> None:
        pass


class BayesianSolo(ScriptedPolicy):
    pass


class HonestCooperator(ScriptedPolicy):
    shares_values = True
    accepts_trades = True
    submit_late = True   # wait for everyone's broadcasts before answering
    measure_target = 2   # relies on pooling rather than measuring a lot alone


class Hoarder(ScriptedPolicy):
    measure_target = 1


class Liar(ScriptedPolicy):
    sells_fabricated = True
    measure_target = 3


class RandomAgent(ScriptedPolicy):
    measure_target = 999  # measure until broke


REGISTRY = {
    "bayesian_solo": BayesianSolo,
    "honest_cooperator": HonestCooperator,
    "hoarder": Hoarder,
    "liar": Liar,
    "random": RandomAgent,
}
