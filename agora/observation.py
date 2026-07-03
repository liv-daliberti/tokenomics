"""Per-agent observations.

An observation is the *complete* information an agent has when it acts: its own
private measurements, its inbox, pending trades, budget and the public prior.
It never contains theta or any other agent's private samples. The same object
is consumed by scripted policies (as a dict) and by the LLM policy (rendered to
text).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .config import GameConfig
from .rewards import break_even_error
from .types import AgentState, RoundResult, Trade


def build_observation(
    state: AgentState,
    cfg: GameConfig,
    round_index: int,
    tick: int,
    peers: List[str],
    pending_trades: List[Trade],
    past_truths: List[float],
    eliminated: Optional[List[str]] = None,
    final_answer: bool = False,
    last_result: Optional[RoundResult] = None,
) -> Dict[str, Any]:
    # NOTE (information isolation): an observation contains ONLY this agent's own
    # measurements, the messages/trades others chose to send it, and the publicly
    # revealed past truths. It never contains another agent's private samples, so
    # an agent can only act on its own evidence plus what was explicitly shared.
    """Assemble the complete information an agent has when it acts: its own measurements, inbox, pending trades, budget and the public prior (plus the final-answer flag). It never contains another agent's private samples."""
    obs = {
        "round_index": round_index,
        "tick": tick,
        "ticks_left": cfg.max_ticks - tick,
        "reveal_horizon": cfg.reveal_horizon and cfg.horizon_mode == "fixed",
        "n_rounds": cfg.n_rounds,
        "agent_id": state.agent_id,
        "credits": state.credits,
        "measure_cost": cfg.measure_cost,
        "survival_cost": cfg.survival_cost,
        "eliminate_on_ruin": cfg.elimination_on_ruin,
        "messages_left": state.messages_left,
        "prior_mu": cfg.prior_mu,
        "prior_sigma": cfg.prior_sigma,
        "my_measurements": [m.value for m in state.measurements],
        "purchased": list(state.purchased),
        "inbox": [
            {"from": m.sender, "to": m.recipient, "text": m.text, "tick": m.tick}
            for m in state.inbox
        ],
        "pending_trades": [
            {"trade_id": t.trade_id, "seller": t.seller,
             "price": t.price, "claimed_value": t.claimed_value}
            for t in pending_trades
        ],
        "peers": peers,
        "eliminated": list(eliminated or []),
        "can_revive": cfg.enable_transfer and cfg.elimination_on_ruin,
        "estimate_submitted": state.estimate is not None,
        "current_estimate": state.estimate,
        "past_truths": list(past_truths) if cfg.reveal_truth_after_round else [],
        "final_answer": final_answer,
        # how accurate you must be to cover this round's survival cost (agent-facing
        # so the survival math isn't a black box); +inf = no accuracy needed, None
        # = even a perfect answer can't cover it.
        "break_even_error": (break_even_error(cfg)
                             if cfg.elimination_on_ruin and cfg.survival_cost > 0 else None),
    }
    # Feedback on the round that just finished (only its own outcome; the truth is
    # public post-round). Lets an agent see how it did and adapt across rounds/games.
    if last_result is not None and cfg.reveal_truth_after_round:
        aid = state.agent_id
        obs["last_round"] = {
            "round": last_result.round_index,
            "truth": last_result.truth,
            "estimate": last_result.estimates.get(aid),
            "error": last_result.errors.get(aid),
            "reward": last_result.rewards.get(aid),
            "credits_start": last_result.credits_start.get(aid),
            "credits_end": last_result.credits_end.get(aid),
        }
    return obs


def render_observation(obs: Dict[str, Any]) -> str:
    """Render an observation as a compact user message for the LLM policy."""
    if obs.get("reveal_horizon"):
        remaining = obs["n_rounds"] - obs["round_index"] - 1
        round_hdr = (
            f"Round {obs['round_index']} of this {obs['n_rounds']}-round game "
            f"({remaining} round(s) remain after this one), tick {obs['tick']} "
            f"({obs['ticks_left']} tick(s) left this round).")
    else:
        round_hdr = (f"Round {obs['round_index']}, tick {obs['tick']} "
                     f"({obs['ticks_left']} tick(s) left this round).")
    sc = obs.get("survival_cost", 0.0)
    dead_at_zero = (" — if it reaches 0 you are ELIMINATED (out of the game)"
                    if obs.get("eliminate_on_ruin", True) else "")
    lines = [
        round_hdr,
        f"YOUR CREDITS: {obs['credits']:g}{dead_at_zero}. Action costs — measure: "
        f"{obs['measure_cost']:g} credit(s) each; send_message: free "
        f"({obs['messages_left']} left this round); a trade costs the agreed price; "
        f"transfer moves the amount you choose.",
    ]
    lr = obs.get("last_round")
    if lr and lr.get("estimate") is not None:
        lines.append(
            f"LAST ROUND (round {lr['round']}): the true value was {lr['truth']:.1f}; you guessed "
            f"{lr['estimate']:.1f} (error {lr['error']:.1f}); you earned {lr['reward']:g} reward; "
            f"your credits went {lr['credits_start']:g} → {lr['credits_end']:g}.")
    elif lr:
        lines.append(
            f"LAST ROUND (round {lr['round']}): the true value was {lr['truth']:.1f}; you did not "
            f"submit, so you were scored on the prior; credits {lr['credits_start']:g} → "
            f"{lr['credits_end']:g}.")
    if obs.get("eliminate_on_ruin", True) and sc > 0:
        lines.append(
            f"SURVIVAL: {sc:g} credit(s) are deducted at the END of every round. If that leaves "
            f"you at zero you are ELIMINATED — so each round you must earn at least {sc:g} in "
            f"reward (more, once you subtract what you spend) or your credits bleed to zero.")
        be = obs.get("break_even_error")
        if be is None:
            lines.append(
                "Even a perfect answer cannot fully cover this round's survival cost — you will "
                "lose some credits no matter what, so spend carefully.")
        elif be != math.inf:
            lines.append(
                f"To break even you need reward enough to cover it: aim to land your final estimate "
                f"within about ±{be:.0f} of the true value (anything you spend measuring/buying "
                f"tightens this).")
    elif obs.get("eliminate_on_ruin", True):
        lines.append("SURVIVAL: if your credits ever hit zero you are ELIMINATED.")
    if obs["credits"] < obs["measure_cost"]:
        lines.append("You cannot afford to measure or buy — reason from what you already have.")
    if obs["eliminated"]:
        who = ", ".join(obs["eliminated"])
        isare = "is" if len(obs["eliminated"]) == 1 else "are"
        itthem = "it" if len(obs["eliminated"]) == 1 else "them"
        if obs.get("can_revive"):
            lines.append(
                f"⚠ ELIMINATED — {who} ran out of credits and {isare} currently out of the game "
                "(cannot measure, message, or trade). You CAN bring an eliminated agent back: "
                f"transfer_credits to {itthem} and {itthem} rejoin next round — give at least the "
                f"survival cost ({sc:g}) so {itthem} can sustain itself.")
        else:
            lines.append(
                f"⚠ ELIMINATED — {who} ran out of credits and {isare} "
                "out of the game for good: they can no longer measure, message, or trade, so do not wait on "
                "them or count on their help.")
    if obs["my_measurements"]:
        vals = ", ".join(f"{v:.1f}" for v in obs["my_measurements"])
        lines.append(f"Your measurements so far: [{vals}].")
    else:
        lines.append("You have taken no measurements yet.")
    if obs["purchased"]:
        buys = "; ".join(
            f"{p['claimed_value']:.1f} from {p['seller']} for {p['price']:g}"
            for p in obs["purchased"]
        )
        lines.append(f"Values you bought: {buys}.")
    if obs["inbox"]:
        lines.append("New messages:")
        for m in obs["inbox"]:
            lines.append(f"  [{m['from']} -> {m['to']}] {m['text']}")
    if obs["pending_trades"]:
        lines.append("Trade offers awaiting your response:")
        for t in obs["pending_trades"]:
            lines.append(
                f"  {t['trade_id']}: {t['seller']} sells value "
                f"{t['claimed_value']:.1f} for {t['price']:g} credits."
            )
    if obs["past_truths"]:
        past = ", ".join(f"{v:.1f}" for v in obs["past_truths"])
        lines.append(f"True values from past rounds (now revealed): [{past}].")
    if obs["estimate_submitted"]:
        lines.append(f"You have already submitted an estimate of {obs['current_estimate']:g}.")
    if obs.get("final_answer"):
        lines.append(
            "THIS IS YOUR FINAL ANSWER for the round. Submit your best estimate of "
            "the hidden value NOW with submit_estimate, combining your own "
            "measurements with everything the other agents shared with you (you may "
            "revise a previous estimate — your last submission is what counts).")
    else:
        lines.append(
            "Take your actions for this tick, then call end_turn (or submit_estimate).")
    return "\n".join(lines)
