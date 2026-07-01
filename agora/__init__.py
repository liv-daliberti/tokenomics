"""Agora — a minimal multi-agent Measurement Market.

N agents estimate a hidden value each round using a costly, noisy measurement
tool and a communication/trade channel. Cooperation (pooling measurements) is
the welfare optimum, but sold measurements are unverifiable, so agents can lie.
Reward is the (non-competitive) distance from ground truth, carried over across
a hidden-horizon sequence of rounds. See docs/DESIGN.md.
"""
from .config import PRESETS, GameConfig, load_config
from .referee import GameResult, Referee
from .transcripts import Transcript

__all__ = [
    "GameConfig",
    "PRESETS",
    "load_config",
    "Referee",
    "GameResult",
    "Transcript",
]

__version__ = "0.1.0"
