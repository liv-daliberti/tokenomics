"""Per-agent observations.

An observation is the *complete* information an agent has when it acts: its own
private measurements, its inbox, pending trades, budget and the public prior.
It never contains theta or any other agent's private samples. The same object
is consumed by scripted policies (as a dict) and by the LLM policy (rendered to
text).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import GameConfig
from .types import AgentState, Trade


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
    }
    return obs


def render_observation(obs: Dict[str, Any]) -> str:
    """Render an observation as a compact user message for the LLM policy."""
    lines = [
        f"Round {obs['round_index']}, tick {obs['tick']} "
        f"({obs['ticks_left']} tick(s) left this round).",
        f"YOUR CREDITS: {obs['credits']:g}. Action costs — measure: {obs['measure_cost']:g} "
        f"credit(s) each; send_message: free ({obs['messages_left']} left this round); a trade "
        f"costs the agreed price; transfer moves the amount you choose.",
    ]
    sc = obs.get("survival_cost", 0.0)
    if obs.get("eliminate_on_ruin", True) and sc > 0:
        lines.append(
            f"SURVIVAL: {sc:g} credit(s) are deducted at the END of every round. If that leaves "
            f"you at zero you are ELIMINATED — so each round you must earn at least {sc:g} in "
            f"reward (more, once you subtract what you spend) or your credits bleed to zero.")
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
