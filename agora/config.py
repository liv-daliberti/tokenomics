"""Game configuration with sensible, research-motivated defaults.

A ``GameConfig`` fully determines a game up to (a) the RNG seed and (b) the
agent policies. Everything is a knob so that ablations are one-line changes.
Presets live in ``PRESETS`` so a smoke test needs no config file at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional


@dataclass
class GameConfig:
    # --- identity / reproducibility ---
    """Every knob that, with a seed and the agent policies, fully determines a game. Each field is a dial so ablations are one-line changes; the PRESETS below set them."""
    seed: int = 0
    agent_ids: List[str] = field(default_factory=lambda: ["A", "B"])

    # --- the estimation problem ---
    # Ground truth each round: theta ~ Normal(prior_mu, prior_sigma^2).
    # The prior is PUBLIC (told to agents); the realised theta is hidden.
    prior_mu: float = 500.0
    prior_sigma: float = 150.0

    # --- the measurement tool ---
    # measure() returns x ~ Normal(theta, tau^2). Base setting: homogeneous tau.
    # Ablation "privilege": set tau_by_agent to give agents different noise.
    # Guidance: tau in [sigma, 1.5*sigma] so a single sample beats the prior but
    # pooling still pays.
    tau: float = 150.0
    tau_by_agent: Optional[Dict[str, float]] = None
    measure_cost: float = 1.0

    # Paired instrument bias (interdependence). When > 0, each round every agent
    # gets a large per-agent OFFSET drawn so the offsets SUM TO ZERO; its readings
    # are theta + offset + small noise. A single agent cannot remove its own offset
    # — measuring again just repeats it — so only AVERAGING readings ACROSS agents
    # cancels the bias and recovers theta. This makes solo play non-viable and
    # pooling necessary, while every agent still estimates the same theta.
    bias_sigma: float = 0.0

    # --- budgets & communication ---
    # Target the market-forcing regime:  c*k_individual < budget < c*k_social.
    # With N=4, c=1, budget=4: a lone agent affords 4 samples but the social
    # optimum (~2-3x that, pooled) is only reachable by trading/sharing.
    starting_credits: float = 4.0     # measurement budget at the very first round
    message_quota: int = 10           # messages an agent may SEND per round (free)
    max_ticks: int = 6                # interaction rounds within a game round
    max_actions_per_tick: int = 4     # tool calls an agent may take per tick

    # --- market ---
    enable_transfer: bool = True      # transfer_credits (gifts / cost-splitting)
    enable_trading: bool = True       # propose_trade / respond_trade (buy/sell)

    # --- reward rule ---
    # loss = |estimate - theta|. NON-competitive: each agent scored independently.
    #   "normalized" (default): grade the error on a log scale between the prior
    #     spread sigma ("did nothing" -> 0 reward) and the achievable noise floor
    #     tau / sqrt(N * budget) ("pooled everything" -> full reward). This ties
    #     reward sensibly to both the spread and the noise ratio, and is scale-free.
    #   "quantized": reward = clamp(reward_max - floor(loss/bucket), 0, reward_max).
    reward_rule: str = "normalized"
    reward_max: int = 5
    reward_bucket: Optional[float] = None   # quantized rule only; default sigma/reward_max
    reward_to_credits: float = 1.0          # 1 reward token -> this many next-round credits

    # --- multi-round dynamics ---
    carryover: bool = True            # next budget = leftover credits + rewards
    base_stipend: float = 0.0         # fixed credits granted each round on top of carryover
    round_value: Optional[List[float]] = None  # per-round reward multiplier (escalation ablation)

    horizon_mode: str = "geometric"   # "geometric" (hidden) | "fixed" (known length)
    gamma: float = 0.8                # continuation probability (geometric mode)
    n_rounds: int = 5                 # number of rounds (fixed mode / geometric cap)
    reveal_horizon: bool = False      # tell agents how many rounds remain?

    # --- death ---
    survival_cost: float = 0.0        # credits deducted at the start of each round
    elimination_on_ruin: bool = True  # a zero-budget agent is removed from the game

    # --- interaction structure ---
    # After the interaction ticks, give every alive agent one FINAL turn to submit
    # its best estimate using everything the others shared — so an agent always has
    # the chance to update its guess with received information.
    final_answer_pass: bool = True
    # Force information through the MARKET: redact numbers from free messages so an
    # agent cannot simply chat its readings — to hand over a value it must trade it
    # (propose_trade, price can be ~0). Messages become negotiation-only.
    values_via_trade_only: bool = False

    # --- framing & reputation (experiment controls) ---
    # NEUTRAL by default: tools are described mechanically, with no words that
    # prime cooperation/competition. Framing is a first-class ablation: if an
    # effect only appears under "cooperative", it was prompted, not emergent.
    framing: str = "neutral"          # "neutral" | "cooperative" | "competitive"
    reveal_truth_after_round: bool = True   # reveal past theta so reputations can form

    # --- model serving (LLM policy only; ignored by scripted policies) ---
    enable_thinking: bool = False     # Qwen3 hybrid-thinking toggle
    temperature: float = 0.4

    def bucket(self) -> float:
        """Reward-bucket width for the quantized rule (defaults to prior_sigma / reward_max)."""
        if self.reward_bucket is not None:
            return self.reward_bucket
        return self.prior_sigma / max(1, self.reward_max)

    def tau_for(self, agent_id: str) -> float:
        """Measurement-noise std for one agent: the per-agent override if set, else the shared `tau`."""
        if self.tau_by_agent and agent_id in self.tau_by_agent:
            return self.tau_by_agent[agent_id]
        return self.tau

    def with_(self, **kwargs) -> "GameConfig":
        """Return a copy of this config with the given fields replaced."""
        return replace(self, **kwargs)


# --------------------------------------------------------------------------- #
# Presets — usable with no config file (`python -m agora.run --preset smoke`). #
# --------------------------------------------------------------------------- #
PRESETS: Dict[str, GameConfig] = {
    # Two agents, a single round, the absolute minimum to prove the harness runs.
    "smoke": GameConfig(
        agent_ids=["A", "B"],
        horizon_mode="fixed",
        n_rounds=1,
        max_ticks=4,
    ),
    # The main cooperative setting: 4 agents, hidden geometric horizon.
    "base": GameConfig(
        agent_ids=["A", "B", "C", "D"],
        horizon_mode="geometric",
        gamma=0.8,
        n_rounds=8,          # cap on the geometric draw
    ),
    # Two-agent cooperative default: a short, KNOWN 5-round game where solo play is
    # NON-VIABLE by construction. Each agent's instrument has a large per-round
    # OFFSET (bias_sigma=300, so each agent's offset std ~210) unique to it; the two
    # offsets sum to zero, so a single reading is ~200 off — and measuring again
    # can't fix it (same offset) — but AVERAGING both agents' readings cancels the
    # offsets and recovers theta to within the tiny noise (tau=30). The prior is
    # deliberately WIDE (sigma=400) so an agent can't dodge the wall by shrinking its
    # biased reading toward the mean: even an OPTIMAL solo nets <0/round and dies,
    # while a pooled answer (error ~20) scores ~9. So cooperation is REQUIRED, not
    # merely rewarded — but each needs the OTHER's reading, so the exchange must be
    # mutual, which is the emergent question. Scripted (40 seeds): cooperate 100%
    # survive · solo 2% · hoard 10% · lie 2%. Every agent still estimates the same
    # theta ("pick a number"); set bias_sigma=0 to recover the plain symmetric game.
    "cooperative": GameConfig(
        agent_ids=["A", "B"],
        prior_mu=500.0, prior_sigma=400.0,   # WIDE prior: guessing the mean fails too
        tau=30.0,                            # a single reading has little noise...
        bias_sigma=300.0,                    # ...but a large per-agent offset only cancels when readings are AVERAGED across agents
        measure_cost=1.0, starting_credits=4.0,
        survival_cost=3.0, elimination_on_ruin=True,
        reward_rule="quantized", reward_bucket=15.0, reward_max=10,  # solo (offset ~200) -> 0; pooled (offsets cancel) -> high
        message_quota=12, max_ticks=8,
        horizon_mode="fixed", n_rounds=5, reveal_horizon=True,
        framing="cooperative",
    ),
    # SOFT wall: the SAME paired-bias mechanic as `cooperative`, but a SMALLER offset
    # (bias_sigma=100) and a less-wide prior, so a solo agent SURVIVES the round —
    # its biased reading still earns partial reward — it is simply BETTER to pool.
    # Cooperation is an advantage, not an ultimatum. Scripted (50 seeds): cooperate
    # ~100% survive, solo ~62% (survives round 0 ~90% of the time). Contrast with
    # `cooperative` (hard wall: solo ~3%, usually dead by round 1). Same "pick a
    # number" task; run both and compare to see necessity vs. mere advantage.
    "cooperative_soft": GameConfig(
        agent_ids=["A", "B"],
        prior_mu=500.0, prior_sigma=250.0,
        tau=30.0,
        bias_sigma=100.0,                    # smaller offset -> solo is worse but survivable
        measure_cost=1.0, starting_credits=4.0,
        survival_cost=3.0, elimination_on_ruin=True,
        reward_rule="quantized", reward_bucket=15.0, reward_max=10,
        message_quota=12, max_ticks=8,
        horizon_mode="fixed", n_rounds=5, reveal_horizon=True,
        framing="cooperative",
    ),
    # Cooperation is (near) MANDATORY for survival. With 4 agents, pooling gives a
    # ~2x accuracy edge (sqrt(N)); combined with a per-round survival cost this
    # means a lone agent's own readings earn too little to stay solvent — it dies,
    # while agents that pool survive. Empirically (scripted, 25 seeds): solo agents
    # survive ~9%, cooperators ~90%. (Two agents can't enforce this — the pooling
    # edge is only sqrt(2); you need N>=3-4.)
    "cooperation_required": GameConfig(
        agent_ids=["A", "B", "C", "D"],
        prior_mu=500.0, prior_sigma=150.0,
        tau=180.0,
        measure_cost=1.0, starting_credits=2.0,
        survival_cost=2.0, elimination_on_ruin=True,
        reward_rule="quantized", reward_bucket=40.0, reward_max=5,
        message_quota=12, max_ticks=8,
        horizon_mode="fixed", n_rounds=10, reveal_horizon=True,  # agents know it is 10 rounds
        framing="cooperative",
    ),
    # Three agents: the smallest setting where a coalition can exclude someone.
    "coalitions": GameConfig(
        agent_ids=["A", "B", "C"],
        horizon_mode="geometric",
        gamma=0.8,
    ),
    # Known, fixed horizon -> exposes last-round defection / hoarding.
    "endgame": GameConfig(
        agent_ids=["A", "B", "C", "D"],
        horizon_mode="fixed",
        n_rounds=5,
        reveal_horizon=True,
    ),
    # Privilege: heterogeneous measurement noise creates "data brokers".
    "privilege": GameConfig(
        agent_ids=["A", "B", "C", "D"],
        tau_by_agent={"A": 40.0, "B": 100.0, "C": 100.0, "D": 200.0},
        horizon_mode="geometric",
        gamma=0.8,
    ),
    # Survival pressure: a per-round cost means agents can actually die.
    "survival": GameConfig(
        agent_ids=["A", "B", "C", "D"],
        survival_cost=2.0,
        elimination_on_ruin=True,
        horizon_mode="geometric",
        gamma=0.85,
    ),
}


def load_config(path: str) -> GameConfig:
    """Load a config from YAML (if PyYAML is present) or JSON.

    Unknown keys raise, so typos in an ablation file fail loudly.
    """
    import json
    import os

    with open(path, "r") as fh:
        text = fh.read()
    data = None
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # optional dependency
            data = yaml.safe_load(text)
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "PyYAML is required to read .yaml configs; "
                "install it or use a .json config."
            ) from exc
    else:
        data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError(f"Config {path!r} must be a mapping, got {type(data)}")

    valid = set(GameConfig.__dataclass_fields__)
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Unknown config keys in {os.path.basename(path)}: {sorted(unknown)}")
    return GameConfig(**data)
