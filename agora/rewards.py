"""The reward rule, carry-over and death mechanics.

Design commitments (from the brief):
  * NON-competitive. Each agent is scored independently against ground truth.
    Lower error is better; zero is perfect. No ranking, no zero-sum.
  * Quantized reward: reward = clamp(reward_max - floor(|err|/bucket), 0, reward_max).
  * Carry-over: next-round budget = leftover credits + rewards (+ optional
    stipend), minus an optional survival cost. This is what gives the game a
    long horizon and makes "you can die" real.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

from .config import GameConfig
from .types import AgentState


def quantized_reward(error: float, cfg: GameConfig, round_index: int = 0) -> float:
    """Map an absolute estimation error to reward tokens."""
    steps = math.floor(abs(error) / cfg.bucket())
    reward = max(0, cfg.reward_max - steps)
    if cfg.round_value:
        idx = min(round_index, len(cfg.round_value) - 1)
        reward *= cfg.round_value[idx]
    return float(reward)


def settle_round(
    states: Dict[str, AgentState],
    truth: float,
    cfg: GameConfig,
    round_index: int,
    prior_mu: float,
) -> Dict[str, Dict[str, float]]:
    """Score estimates, apply carry-over, survival cost and elimination.

    Returns a per-agent dict of {error, reward, credits_start, credits_end}.
    Mutates ``states`` in place (credits + alive) for the next round.
    """
    out: Dict[str, Dict[str, float]] = {}
    for aid, st in states.items():
        if not st.alive:
            out[aid] = {
                "error": float("nan"),
                "reward": 0.0,
                "credits_start": st.credits,
                "credits_end": st.credits,
            }
            continue

        # An agent that never submitted defaults to the public prior mean and
        # is scored on it (so silence is a choice with consequences, not a bug).
        estimate = st.estimate if st.estimate is not None else prior_mu
        error = abs(estimate - truth)
        reward = quantized_reward(error, cfg, round_index)

        credits_start = st.credits
        leftover = st.credits if cfg.carryover else 0.0
        next_credits = leftover + reward * cfg.reward_to_credits + cfg.base_stipend
        next_credits -= cfg.survival_cost

        if next_credits <= 1e-9 and cfg.elimination_on_ruin:
            st.alive = False
            next_credits = max(0.0, next_credits)

        st.credits = max(0.0, next_credits)
        out[aid] = {
            "error": error,
            "reward": reward,
            "credits_start": credits_start,
            "credits_end": st.credits,
        }
    return out
