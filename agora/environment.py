"""The world: ground truth, the measurement tool, and the horizon.

The environment owns all randomness through a single seeded RNG so that a game
is fully reproducible from (config, seed). Because the environment *generates*
every measurement, it always knows the value an agent truly observed — this is
what makes deception verifiable downstream.
"""
from __future__ import annotations

import random
from typing import List

from .config import GameConfig


class Environment:
    """Owns all game randomness via one seeded RNG: draws the ground truth, the noisy measurements, and the (possibly hidden) horizon."""
    def __init__(self, cfg: GameConfig):
        """Seed the RNG from the config and initialise per-round bookkeeping."""
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self._round_truths: List[float] = []
        # paired-bias mode: per-agent instrument offsets for the current round
        self.offsets: dict = {a: 0.0 for a in cfg.agent_ids}

    def draw_truth(self, round_index: int) -> float:
        """Ground truth for a round: theta ~ Normal(prior_mu, prior_sigma^2).

        Every agent estimates this same hidden number ("pick a number"). In
        paired-bias mode (bias_sigma > 0) also draw per-agent instrument offsets
        that SUM TO ZERO, so averaging every agent's reading cancels them."""
        theta = self.rng.gauss(self.cfg.prior_mu, self.cfg.prior_sigma)
        if self.cfg.bias_sigma > 0:
            raw = {a: self.rng.gauss(0.0, self.cfg.bias_sigma) for a in self.cfg.agent_ids}
            mean = sum(raw.values()) / len(raw)
            self.offsets = {a: raw[a] - mean for a in self.cfg.agent_ids}  # sum to 0
        # keep the log dense-indexed even if rounds are drawn out of order
        while len(self._round_truths) <= round_index:
            self._round_truths.append(float("nan"))
        self._round_truths[round_index] = theta
        return theta

    def measure(self, truth: float, tau: float) -> float:
        """One noisy sample from the measurement tool: x ~ Normal(theta, tau^2)."""
        return self.rng.gauss(truth, tau)

    def horizon(self) -> List[float]:
        """Return the per-round continuation decisions.

        Fixed mode: exactly ``n_rounds`` rounds. Geometric mode: keep going
        with probability ``gamma`` after each round, capped at ``n_rounds`` so a
        game always terminates. Decided up front (seeded) so it is reproducible,
        but never revealed to agents unless ``reveal_horizon`` is set.
        """
        cfg = self.cfg
        if cfg.horizon_mode == "fixed":
            return [1.0] * cfg.n_rounds
        # geometric
        n = 1
        while n < cfg.n_rounds and self.rng.random() < cfg.gamma:
            n += 1
        return [1.0] * n
