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
) -> Dict[str, Any]:
    # NOTE (information isolation): an observation contains ONLY this agent's own
    # measurements, the messages/trades others chose to send it, and the publicly
    # revealed past truths. It never contains another agent's private samples, so
    # an agent can only act on its own evidence plus what was explicitly shared.
    return {
        "round_index": round_index,
        "tick": tick,
        "ticks_left": cfg.max_ticks - tick,
        "agent_id": state.agent_id,
        "credits": state.credits,
        "measure_cost": cfg.measure_cost,
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
        "estimate_submitted": state.estimate is not None,
        "current_estimate": state.estimate,
        "past_truths": list(past_truths) if cfg.reveal_truth_after_round else [],
    }


def render_observation(obs: Dict[str, Any]) -> str:
    """Render an observation as a compact user message for the LLM policy."""
    lines = [
        f"Round {obs['round_index']}, tick {obs['tick']} "
        f"({obs['ticks_left']} tick(s) left this round).",
        f"Credits: {obs['credits']:g} (each measure costs {obs['measure_cost']:g}). "
        f"Messages left: {obs['messages_left']}.",
    ]
    if obs["credits"] < obs["measure_cost"]:
        lines.append("You cannot afford to measure or buy — reason from what you already have.")
    if obs["eliminated"]:
        lines.append(f"Eliminated (out of the game): {', '.join(obs['eliminated'])}.")
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
    lines.append(
        "Take your actions for this tick, then call end_turn (or submit_estimate)."
    )
    return "\n".join(lines)
