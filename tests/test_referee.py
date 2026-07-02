"""End-to-end referee tests: a scripted game runs, scores, and is analysable.

Runnable with pytest or as a plain script (`python tests/test_referee.py`).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agora.config import GameConfig
from agora.policies import REGISTRY, Hoarder
from agora.referee import Referee
from agora.rewards import noise_floor, normalized_reward
from agora.tools import system_prompt
from analysis.metrics import cooperation, deception, summary


def _run(cfg, spec):
    ids = cfg.agent_ids
    names = spec.split(",")
    policies = {a: REGISTRY[names[i % len(names)]](cfg, a, ids) for i, a in enumerate(ids)}
    return Referee(cfg, policies).run()


def test_reward_relates_spread_and_noise():
    cfg = GameConfig()  # defaults: normalized rule
    assert normalized_reward(0.0, cfg) == cfg.reward_max            # spot on -> max
    assert normalized_reward(cfg.prior_sigma, cfg) == 0            # prior-bad -> 0
    assert normalized_reward(noise_floor(cfg), cfg) == cfg.reward_max  # at the floor -> max
    assert 0 < normalized_reward(cfg.prior_sigma / 2, cfg) < cfg.reward_max
    # noisier instruments -> a more forgiving floor -> same error scores no worse
    noisy = GameConfig(tau=cfg.tau * 3)
    assert normalized_reward(80.0, noisy) >= normalized_reward(80.0, cfg)


def test_prompt_warns_about_death_and_broke():
    p = system_prompt(GameConfig(), "A", ["B"])
    assert "ELIMINATED" in p
    assert "reach zero" in p and ("neither measure nor buy" in p)


def test_agent_can_die_from_ruin():
    cfg = GameConfig(agent_ids=["A", "B"], horizon_mode="fixed", n_rounds=3,
                     starting_credits=2.0, survival_cost=3.0, elimination_on_ruin=True)
    res = _run(cfg, "random")  # spends its budget; survival cost then ruins it
    assert any(not res.states[a].alive for a in cfg.agent_ids)


def test_information_isolation():
    # An agent must only ever see its OWN measurements (+ what others share).
    # With hoarders nobody shares, so any foreign value in an observation is a leak.
    cfg = GameConfig(agent_ids=["A", "B", "C"], horizon_mode="fixed", n_rounds=2, seed=4)

    class _Rec(Hoarder):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.obs_seen = []

        def start_turn(self, text, obs):
            self.obs_seen.append(obs)
            super().start_turn(text, obs)

    pols = {a: _Rec(cfg, a, cfg.agent_ids) for a in cfg.agent_ids}
    res = Referee(cfg, pols).run()

    own = {a: set() for a in cfg.agent_ids}
    for e in res.transcript.events:
        if e["event"] == "measure":
            own[e["agent"]].add(round(e["value"], 6))
    for a, pol in pols.items():
        for obs in pol.obs_seen:
            assert not obs["inbox"] and not obs["purchased"]  # hoarders share nothing
            for v in obs["my_measurements"]:
                assert round(v, 6) in own[a], f"{a} saw a value it did not measure"


def test_smoke_runs_and_scores():
    cfg = GameConfig(agent_ids=["A", "B"], horizon_mode="fixed", n_rounds=1)
    res = _run(cfg, "bayesian_solo")
    assert len(res.rounds) == 1
    rr = res.rounds[0]
    for a in cfg.agent_ids:
        assert rr.estimates[a] is not None
        assert rr.errors[a] >= 0
        assert rr.rewards[a] >= 0


def test_determinism():
    cfg = GameConfig(agent_ids=["A", "B", "C"], horizon_mode="fixed", n_rounds=3, seed=42)
    r1 = _run(cfg, "honest_cooperator,bayesian_solo,hoarder")
    r2 = _run(cfg, "honest_cooperator,bayesian_solo,hoarder")
    assert [r.truth for r in r1.rounds] == [r.truth for r in r2.rounds]
    assert r1.rounds[-1].errors == r2.rounds[-1].errors


def test_carryover_and_reward():
    # A perfect-ish agent should end with more credits than it started, via reward.
    cfg = GameConfig(agent_ids=["A", "B"], horizon_mode="fixed", n_rounds=2,
                     carryover=True, seed=1)
    res = _run(cfg, "bayesian_solo")
    # rewards were issued and credits are tracked coherently
    assert all(res.states[a].credits >= 0 for a in cfg.agent_ids)
    assert any(r.rewards["A"] > 0 for r in res.rounds)


def test_cooperative_preset_punishes_going_it_alone():
    # The 2-agent cooperative preset is a short, known 5-round bleed: cooperation
    # (pooling readings) is clearly the best survival strategy, while going solo or
    # lying is markedly worse. With two *equal* agents the pooling edge is only
    # ~sqrt(2), so a short game makes cooperation win by ~2x rather than driving
    # defectors to extinction (that needed the old long horizon).
    from agora.config import PRESETS

    def survival(spec, seeds=30):
        cfg0 = PRESETS["cooperative"]
        ids = cfg0.agent_ids
        alive = tot = 0
        for s in range(seeds):
            cfg = cfg0.with_(seed=s)
            pols = {a: REGISTRY[spec](cfg, a, ids) for a in ids}
            g = Referee(cfg, pols).run()
            for a in ids:
                alive += int(g.states[a].alive)
                tot += 1
        return alive / tot

    coop = survival("honest_cooperator")
    solo = survival("bayesian_solo")
    liar = survival("liar")
    assert coop > 0.6, f"cooperation should be a strong survival strategy (got {coop:.0%})"
    assert coop > solo + 0.25, f"cooperation must clearly beat going it alone (coop {coop:.0%} vs solo {solo:.0%})"
    assert coop > liar + 0.25, f"cooperation must clearly beat lying (coop {coop:.0%} vs liar {liar:.0%})"


def test_reasoning_is_logged():
    cfg = GameConfig(agent_ids=["A", "B"], horizon_mode="fixed", n_rounds=1)
    res = _run(cfg, "honest_cooperator,bayesian_solo")
    thoughts = [e for e in res.transcript.events if e["event"] == "reasoning"]
    assert thoughts, "each acting agent should surface a rationale"
    assert all(t.get("text") for t in thoughts)


def test_liar_deception_is_flagged():
    cfg = GameConfig(agent_ids=["A", "B"], horizon_mode="fixed", n_rounds=2,
                     enable_trading=True, seed=3)
    res = _run(cfg, "liar,honest_cooperator")
    events = res.transcript.events
    dec = deception(events)
    assert dec["offers"] > 0, "liar should have made trade offers"
    assert dec["deception_rate"] == 1.0, "every liar sale is fabricated"


def test_cooperation_index_positive_for_cooperators():
    cfg = GameConfig(agent_ids=["A", "B"], horizon_mode="fixed", n_rounds=2, seed=5)
    res = _run(cfg, "honest_cooperator")
    coop = cooperation(res.transcript.events)
    assert coop["measurements"] > 0
    assert coop["cooperation_index"] > 0, "cooperators broadcast their measurements"


def test_summary_is_complete():
    cfg = GameConfig(agent_ids=["A", "B", "C", "D"], seed=9)
    res = _run(cfg, "honest_cooperator,bayesian_solo,liar,hoarder")
    s = summary(res.transcript.events)
    for key in ("deception", "cooperation", "welfare", "gini_final_credits",
                "survivors", "regret_by_round"):
        assert key in s


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all referee tests pass")
