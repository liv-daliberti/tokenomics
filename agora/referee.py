"""The referee: the authoritative game loop.

The referee owns the world. It draws ground truth, schedules turns, executes
every action (the only place that mutates state), enforces quotas and escrow,
and logs everything. Turns within a tick are SEQUENTIAL so a message sent by an
earlier agent is visible to a later agent in the same tick (the conversation
protocol). Cross-agent parallelism is a future optimisation for simultaneous
moves only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _redact_numbers(text: str) -> str:
    """Hide numeric tokens so a value can't be conveyed in free chat."""
    return _NUM_RE.sub("[#]", text)

from .config import GameConfig
from .environment import Environment
from .market import Market, MarketError
from .observation import build_observation, render_observation
from .rewards import settle_round
from .tools import system_prompt
from .transcripts import Transcript
from .types import Action, ActionType, AgentState, Measurement, Message, RoundResult


@dataclass
class GameResult:
    config: GameConfig
    states: Dict[str, AgentState]
    rounds: List[RoundResult]
    transcript: Transcript


@dataclass
class MatchResult:
    config: GameConfig
    n_games: int
    games: List[GameResult]
    transcript: Transcript


def run_match(cfg: GameConfig, policies: Dict[str, object], n_games: int,
              transcript: Optional[Transcript] = None) -> MatchResult:
    """Play ``n_games`` games back-to-back with the SAME policy objects.

    The world resets each game (fresh hidden values via a per-game seed, budgets
    restored, agents revived) but the policies persist — so an LLM agent keeps
    its whole conversation and can remember and adapt to earlier games. This is
    the "co-evolving within a context window" setting.
    """
    tx = transcript or Transcript()
    tx.log("match_start", config=cfg, n_games=n_games)
    games: List[GameResult] = []
    for g in range(n_games):
        for pol in policies.values():
            if hasattr(pol, "reset_game"):
                pol.reset_game(g, n_games)
        ref = Referee(cfg.with_(seed=cfg.seed + g), policies, tx,
                      game_index=g, n_games=n_games)
        games.append(ref.run())
    tx.log("match_end", n_games=n_games)
    return MatchResult(cfg, n_games, games, tx)


class Referee:
    def __init__(self, cfg: GameConfig, policies: Dict[str, object],
                 transcript: Optional[Transcript] = None, game_index: int = 0,
                 n_games: int = 1):
        self.cfg = cfg
        self.policies = policies
        self.game_index = game_index
        self.n_games = n_games
        self.env = Environment(cfg)
        self.tx = transcript or Transcript()
        self.states: Dict[str, AgentState] = {
            aid: AgentState(agent_id=aid, credits=cfg.starting_credits,
                            tau=cfg.tau_for(aid), messages_left=cfg.message_quota)
            for aid in cfg.agent_ids
        }
        self._trade_seq = 0
        self.market = Market(self.states, self._next_trade_id)
        self.truth = float("nan")
        self.round_index = 0

    def _next_trade_id(self) -> str:
        self._trade_seq += 1
        return f"T{self._trade_seq}"

    def _alive(self) -> List[str]:
        return [a for a in self.cfg.agent_ids if self.states[a].alive]

    # ----------------------------------------------------------------- run --
    def run(self) -> GameResult:
        cfg = self.cfg
        horizon = self.env.horizon()
        self.tx.log("game_start", game_index=self.game_index, config=cfg,
                    n_rounds_actual=len(horizon))
        # Record the prompt each agent is given (the shared task framing), so a
        # report can show "the prompt asked of the agents".
        for aid in cfg.agent_ids:
            peers = [a for a in cfg.agent_ids if a != aid]
            self.tx.log("agent_prompt", agent=aid,
                        text=system_prompt(cfg, aid, peers, self.n_games))
        past_truths: List[float] = []
        rounds: List[RoundResult] = []

        for r, _ in enumerate(horizon):
            if not self._alive():
                break
            self.round_index = r
            self.truth = self.env.draw_truth(r)
            self._reset_round(r)
            self.tx.log("round_start", game_index=self.game_index, round=r,
                        truth=self.truth, alive=self._alive(),
                        credits={a: self.states[a].credits for a in self._alive()})

            self._run_ticks(past_truths)

            alive_before = {a: self.states[a].alive for a in cfg.agent_ids}
            scored = settle_round(self.states, self.truth, cfg, r, cfg.prior_mu)
            rr = RoundResult(
                round_index=r,
                truth=self.truth,
                estimates={a: self.states[a].estimate for a in cfg.agent_ids},
                errors={a: scored[a]["error"] for a in cfg.agent_ids},
                rewards={a: scored[a]["reward"] for a in cfg.agent_ids},
                credits_start={a: scored[a]["credits_start"] for a in cfg.agent_ids},
                credits_end={a: scored[a]["credits_end"] for a in cfg.agent_ids},
                alive={a: self.states[a].alive for a in cfg.agent_ids},
            )
            rounds.append(rr)
            self.tx.log("round_end", game_index=self.game_index, round=r, result=rr)
            for a in cfg.agent_ids:
                if alive_before[a] and not self.states[a].alive:
                    self.tx.log("elimination", game_index=self.game_index, round=r, agent=a)
            past_truths.append(self.truth)

        self.tx.log("game_end", final_credits={a: self.states[a].credits
                                               for a in cfg.agent_ids})
        return GameResult(cfg, self.states, rounds, self.tx)

    def _reset_round(self, round_index: int) -> None:
        for a in self._alive():
            st = self.states[a]
            st.messages_left = self.cfg.message_quota
            st.estimate = None
            st.measurements = []
            st.purchased = []
            # NB: st.inbox is deliberately NOT cleared here — unseen messages
            # (e.g. one delivered on the last tick after the recipient's final
            # turn) carry into the next round's first observation so they are
            # never silently dropped. Seen messages are already cleared in
            # _take_turn once surfaced.
            self.policies[a].reset_round(round_index)
        self.market.trades.clear()

    def _run_ticks(self, past_truths: List[float]) -> None:
        cfg = self.cfg
        for tick in range(cfg.max_ticks):
            order = list(self._alive())
            self.env.rng.shuffle(order)
            self.tx.log("tick_start", round=self.round_index, tick=tick, order=order)
            substantive = False
            for aid in order:
                if not self.states[aid].alive:
                    continue
                substantive |= self._take_turn(aid, tick, past_truths)
            all_submitted = all(self.states[a].estimate is not None for a in self._alive())
            if all_submitted and not substantive:
                break

        # Final-answer pass: everyone gets one last turn to commit their best
        # estimate using all information exchanged this round.
        if cfg.final_answer_pass:
            for aid in list(self._alive()):
                self._take_turn(aid, cfg.max_ticks, past_truths, final=True)

    def _take_turn(self, aid: str, tick: int, past_truths: List[float],
                   final: bool = False) -> bool:
        cfg = self.cfg
        st = self.states[aid]
        peers = [a for a in cfg.agent_ids if a != aid and self.states[a].alive]
        eliminated = [a for a in cfg.agent_ids if a != aid and not self.states[a].alive]
        pending = [t for t in self.market.trades.values()
                   if t.buyer == aid and t.status == "pending"]
        obs = build_observation(st, cfg, self.round_index, tick, peers, pending,
                                past_truths, eliminated, final_answer=final)
        st.inbox = []  # surfaced now; each message is shown once
        obs_text = render_observation(obs)
        self.tx.log("prompt", agent=aid, game_index=self.game_index,
                    round=self.round_index, tick=tick, final=final, text=obs_text)
        policy = self.policies[aid]
        policy.start_turn(obs_text, obs)

        substantive = False
        actions_taken = 0
        while actions_taken < cfg.max_actions_per_tick:
            invs = policy.next_actions()
            thought = policy.last_reasoning()
            if thought:
                self.tx.log("reasoning", agent=aid, tick=tick, text=thought)
            if not invs:
                break
            results = []
            ended = False
            for inv in invs:
                if inv.action is None:
                    self.tx.log("parse_fail", agent=aid, tool=inv.name, error=inv.error)
                    results.append((inv.call_id, f"ERROR: {inv.error}"))
                else:
                    res, subst, end = self._execute(aid, inv.action, tick)
                    substantive |= subst
                    ended |= end
                    results.append((inv.call_id, res))
                actions_taken += 1
            policy.observe_results(results)
            if ended:
                break
        return substantive

    # -------------------------------------------------------------- execute --
    def _execute(self, aid: str, action: Action, tick: int):
        """Return (result_string, substantive, ended)."""
        cfg = self.cfg
        st = self.states[aid]
        a = action.args

        if action.type is ActionType.END_TURN:
            return "end_turn", False, True

        if action.type is ActionType.MEASURE:
            if not st.can_afford(cfg.measure_cost):
                return "ERROR: insufficient credits to measure", True, False
            self.market.spend(aid, cfg.measure_cost, "measure")
            # complementary mode: an agent's instrument reads only its own component
            target = self.env.components[aid] if cfg.complementary else self.truth
            x = self.env.measure(target, st.tau)
            st.measurements.append(Measurement(aid, x, target, st.tau, tick, cfg.measure_cost))
            self.tx.log("measure", agent=aid, tick=tick, value=x, truth=target,
                        tau=st.tau, cost=cfg.measure_cost, credits_after=st.credits)
            return f"measured {x:.4f}", True, False

        if action.type is ActionType.SEND_MESSAGE:
            if st.messages_left <= 0:
                return "ERROR: message quota exhausted", True, False
            to, text = a["to"], a["text"]
            if to != "all" and (to not in self.states or not self.states[to].alive):
                self.tx.log("misaddressed", agent=aid, to=to, tick=tick)
                return f"ERROR: no such live agent {to!r}", True, False
            st.messages_left -= 1
            # values-via-trade-only: strip numbers so chat can't convey a reading
            delivered = _redact_numbers(text) if cfg.values_via_trade_only else text
            recipients = [x for x in self._alive() if x != aid] if to == "all" else [to]
            for r in recipients:
                self.states[r].inbox.append(Message(aid, to, delivered, tick))
            self.tx.log("message", sender=aid, to=to, text=delivered, tick=tick)
            note = " (numbers hidden — trade to share a value)" if cfg.values_via_trade_only else ""
            return f"sent{note}", True, False

        if action.type is ActionType.TRANSFER:
            try:
                self.market.transfer(aid, a["to"], a["amount"])
            except MarketError as exc:
                return f"ERROR: {exc}", True, False
            self.tx.log("transfer", src=aid, dst=a["to"], amount=a["amount"], tick=tick)
            return f"transferred {a['amount']:g} to {a['to']}", True, False

        if action.type is ActionType.PROPOSE_TRADE:
            try:
                trade = self.market.propose_trade(aid, a["to"], a["price"],
                                                  a["claimed_value"], tick)
            except MarketError as exc:
                return f"ERROR: {exc}", True, False
            # log the seller's ACTUALLY observed values -> verifiable deception later
            self.tx.log("propose_trade", trade_id=trade.trade_id, seller=aid,
                        buyer=a["to"], price=a["price"], claimed_value=a["claimed_value"],
                        seller_observed=[m.value for m in st.measurements], tick=tick)
            return f"offered trade {trade.trade_id} to {a['to']}", True, False

        if action.type is ActionType.RESPOND_TRADE:
            try:
                trade = self.market.respond_trade(aid, a["trade_id"], a["accept"])
            except MarketError as exc:
                return f"ERROR: {exc}", True, False
            self.tx.log("respond_trade", trade_id=a["trade_id"], responder=aid,
                        accept=a["accept"], status=trade.status, tick=tick)
            return f"trade {a['trade_id']} {trade.status}", True, False

        if action.type is ActionType.SUBMIT_ESTIMATE:
            st.estimate = a["value"]
            self.tx.log("submit_estimate", agent=aid, value=a["value"], tick=tick)
            return f"submitted estimate {a['value']:g}", True, False

        return f"ERROR: unhandled action {action.type}", True, False  # pragma: no cover
