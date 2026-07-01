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


def noise_floor(cfg: GameConfig) -> float:
    """Best error the group could achieve by pooling every affordable sample.

    With N agents each able to buy `budget/cost` measurements of noise `tau`,
    pooling them all gives a mean whose std is ~ tau / sqrt(N * budget). Floored
    at 1% of the prior spread so the reward stays sane if measuring is free.
    """
    budget = cfg.starting_credits / max(cfg.measure_cost, 1e-9)
    total = max(1.0, len(cfg.agent_ids) * budget)
    floor = cfg.tau / math.sqrt(total)
    return max(floor, 0.01 * cfg.prior_sigma)


def quantized_reward(error: float, cfg: GameConfig, round_index: int = 0) -> float:
    """Map an absolute estimation error to reward tokens (step function)."""
    steps = math.floor(abs(error) / cfg.bucket())
    reward = max(0, cfg.reward_max - steps)
    return float(reward)


def normalized_reward(error: float, cfg: GameConfig, round_index: int = 0) -> float:
    """Scale-free reward tied to the spread (sigma) and the noise (tau).

    frac = 1 when the error reaches the achievable noise floor (pooled optimum),
    0 when it is as bad as the prior spread sigma (submitting the prior mean),
    and interpolated on a log scale in between. reward = round(reward_max * frac).
    """
    sigma = cfg.prior_sigma
    floor = noise_floor(cfg)
    err = max(abs(error), floor)          # cannot score better than the floor
    if sigma <= floor:                    # degenerate problem: no room to improve
        frac = 1.0 if abs(error) <= floor else 0.0
    else:
        frac = (math.log(sigma) - math.log(err)) / (math.log(sigma) - math.log(floor))
    frac = min(1.0, max(0.0, frac))
    return float(int(cfg.reward_max * frac + 0.5))


def reward_for(error: float, cfg: GameConfig, round_index: int = 0) -> float:
    """Dispatch on the configured reward rule and apply any round-value scaling."""
    rule = normalized_reward if cfg.reward_rule == "normalized" else quantized_reward
    reward = rule(error, cfg, round_index)
    if cfg.round_value:
        idx = min(round_index, len(cfg.round_value) - 1)
        reward *= cfg.round_value[idx]
    return reward


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
        reward = reward_for(error, cfg, round_index)

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
