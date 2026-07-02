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
    def __init__(self, cfg: GameConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self._round_truths: List[float] = []
        # complementary mode: per-agent component truth; theta = sum of components
        self.components: dict = {}

    def component_prior(self):
        """(mu, sigma) of a single agent's component, so the N components sum to
        the public theta prior Normal(prior_mu, prior_sigma^2)."""
        n = max(1, len(self.cfg.agent_ids))
        return self.cfg.prior_mu / n, self.cfg.prior_sigma / (n ** 0.5)

    def draw_truth(self, round_index: int) -> float:
        """Ground truth for a round.

        Scalar mode: theta ~ Normal(prior_mu, prior_sigma^2).
        Complementary mode: each agent gets a private component ~ Normal(mu/N,
        (sigma/sqrt(N))^2); theta is their sum (same marginal prior on theta)."""
        if self.cfg.complementary:
            mu_c, sig_c = self.component_prior()
            self.components = {a: self.rng.gauss(mu_c, sig_c) for a in self.cfg.agent_ids}
            theta = sum(self.components.values())
        else:
            theta = self.rng.gauss(self.cfg.prior_mu, self.cfg.prior_sigma)
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
