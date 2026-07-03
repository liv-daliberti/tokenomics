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
    # The 2-agent cooperative preset is a HARD wall via paired instrument bias: each
    # agent's reading carries a large fixed per-round offset (the offsets cancel only
    # when averaged across agents), and the prior is wide, so NO solo strategy — not
    # even measuring repeatedly or shrinking toward the prior — recovers theta. Only
    # agents that pool readings survive. Scripted (30 seeds): cooperate ~100%, solo
    # ~2%, hoard ~10%, lie ~2%.
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
    assert coop > 0.85, f"pooling cancels the bias -> cooperators reliably survive (got {coop:.0%})"
    assert solo < 0.30, f"a lone agent is stuck at its instrument offset and dies (got {solo:.0%})"
    assert survival("hoarder") < 0.40, "even a passive hoarder cannot free-ride to survival here"
    assert survival("liar") < 0.30, "lying cannot substitute for a real pooled reading"


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
    for key in ("deception", "cooperation", "reciprocity", "rescue", "price_stats",
                "welfare", "gini_final_credits", "survivors", "regret_by_round"):
        assert key in s


def test_reciprocity_detects_one_sided_exchange():
    from analysis.metrics import reciprocity
    # A shares its reading with B; B measures but never shares back -> one-sided.
    ev = [
        {"event": "game_start", "config": {"agent_ids": ["A", "B"]}},
        {"event": "measure", "agent": "A", "value": 500.0},
        {"event": "measure", "agent": "B", "value": 510.0},
        {"event": "message", "sender": "A", "to": "B", "text": "my reading is 500.0"},
        {"event": "message", "sender": "B", "to": "A", "text": "thanks, noted"},  # no value
    ]
    r = reciprocity(ev)
    assert r["directed"] == {"A->B": 1}
    assert r["reciprocity_index"] == 0.0 and r["one_sided_pairs"] == 1 and r["mutual_pairs"] == 0
    # B shares back -> now mutual
    ev.append({"event": "message", "sender": "B", "to": "A", "text": "mine is 510.0"})
    r2 = reciprocity(ev)
    assert r2["reciprocity_index"] == 1.0 and r2["mutual_pairs"] == 1 and r2["one_sided_pairs"] == 0


def test_paired_bias_offsets_cancel_only_when_pooled():
    # Each agent's readings carry a fixed per-round offset; the offsets sum to zero,
    # so a lone agent (even measuring many times) is stuck at its offset, but
    # averaging both agents' readings recovers theta.
    import statistics
    from agora.environment import Environment
    cfg = GameConfig(agent_ids=["A", "B"], bias_sigma=300.0, tau=30.0,
                     prior_mu=500.0, prior_sigma=400.0, seed=1)
    env = Environment(cfg)
    theta = env.draw_truth(0)
    assert abs(env.offsets["A"] + env.offsets["B"]) < 1e-9          # offsets sum to zero
    assert abs(env.offsets["A"]) > 50                               # and are large
    a = statistics.mean(env.measure(theta + env.offsets["A"], cfg.tau) for _ in range(200))
    b = statistics.mean(env.measure(theta + env.offsets["B"], cfg.tau) for _ in range(200))
    assert abs(a - theta) > 40, "a lone agent stays stuck at its offset no matter how much it measures"
    assert abs((a + b) / 2 - theta) < 15, "averaging the two agents' readings recovers theta"


def test_paired_bias_is_explained_in_the_prompt():
    from agora.tools import system_prompt
    cfg = GameConfig(agent_ids=["A", "B"], bias_sigma=300.0)
    sp = system_prompt(cfg, "A", ["B"]).lower()
    assert "offset" in sp and "average" in sp and ("cancel" in sp or "cancels" in sp)


def test_reciprocity_ignores_dead_agent_rounds():
    # A share into an already-eliminated partner must NOT count against reciprocity
    # (a dead agent cannot reciprocate).
    from analysis.metrics import reciprocity
    ev = [
        {"event": "game_start", "config": {"agent_ids": ["A", "B"]}},
        {"event": "round_start", "round": 0, "alive": ["A", "B"]},
        {"event": "measure", "agent": "A", "value": 500.0},
        {"event": "message", "sender": "A", "to": "B", "text": "reading 500.0"},   # both alive
        {"event": "round_start", "round": 1, "alive": ["A"]},                       # B eliminated
        {"event": "measure", "agent": "A", "value": 498.0},
        {"event": "message", "sender": "A", "to": "all", "text": "reading 498.0"},  # B dead -> dropped
    ]
    assert reciprocity(ev)["directed"] == {"A->B": 1}


def test_rescue_and_price_stats():
    from analysis.metrics import rescue, price_stats
    ev = [
        {"event": "transfer", "src": "A", "dst": "B", "amount": 3.0},
        {"event": "revival", "agent": "B", "credits": 3.0},
        {"event": "elimination", "agent": "B"},
        {"event": "propose_trade", "trade_id": "T1", "seller": "A", "buyer": "B",
         "price": 0.0, "claimed_value": 500.0, "seller_observed": [500.0]},
        {"event": "respond_trade", "trade_id": "T1", "responder": "B", "status": "accepted"},
        {"event": "propose_trade", "trade_id": "T2", "seller": "A", "buyer": "B",
         "price": 2.0, "claimed_value": 501.0, "seller_observed": [501.0]},
        {"event": "respond_trade", "trade_id": "T2", "responder": "B", "status": "accepted"},
    ]
    rc = rescue(ev)
    assert (rc["transfers"], rc["revivals"], rc["eliminations"]) == (1, 1, 1)
    assert rc["credits_transferred"] == 3.0
    ps = price_stats(ev)
    assert ps["settled"]["n"] == 2 and ps["settled_gifts"] == 1 and ps["settled_charged"] == 1


def test_study_runner_aggregates():
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts", "study.py")
    spec = importlib.util.spec_from_file_location("study", path)
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    m = study.run_one("cooperative", "honest_cooperator", 0, 1)
    assert 0.0 <= m["survival"] <= 1.0 and m["cooperation"] == m["cooperation"]
    mean, ci, n = study.aggregate([0.5, 0.7, 0.9])
    assert abs(mean - 0.7) < 1e-9 and n == 3 and ci > 0


def test_prompt_states_the_exact_reward_function():
    # An agent must be told the precise error->reward mapping, not just "closer is
    # better", so it can reason about how accurate it needs to be.
    cfg = GameConfig(agent_ids=["A", "B"], reward_rule="quantized",
                     reward_bucket=13.0, reward_max=10, reward_to_credits=1.0)
    sp = system_prompt(cfg, "A", ["B"])
    assert "max(0, 10 - floor(" in sp and "/ 13" in sp     # the exact step function
    assert "reward token becomes 1 credit" in sp           # reward -> credits conversion


def test_break_even_error_covers_the_survival_cost():
    import math
    from agora.rewards import break_even_error, reward_for
    cfg = GameConfig(agent_ids=["A"], reward_rule="quantized", reward_bucket=13.0,
                     reward_max=10, survival_cost=2.5, reward_to_credits=1.0,
                     base_stipend=0.0, prior_sigma=150.0)
    be = break_even_error(cfg)
    assert be not in (None, math.inf)
    # at/under the threshold you earn enough to cover survival; well past it you don't
    assert reward_for(be, cfg) >= math.ceil(2.5)
    assert reward_for(be + 3 * 13.0, cfg) < math.ceil(2.5)
    # if a perfect answer still can't cover the cost, that is signalled with None
    assert break_even_error(cfg.with_(survival_cost=999.0)) is None


def test_per_round_feedback_is_own_outcome_only_and_resets_each_game():
    from agora.observation import build_observation, render_observation
    from agora.types import AgentState, RoundResult
    cfg = GameConfig(agent_ids=["A", "B"], reveal_truth_after_round=True)
    st = AgentState(agent_id="A", credits=7.0, tau=cfg.tau, messages_left=cfg.message_quota)
    rr = RoundResult(round_index=0, truth=500.0,
                     estimates={"A": 480.0, "B": 510.0}, errors={"A": 20.0, "B": 10.0},
                     rewards={"A": 8.0, "B": 9.0}, credits_start={"A": 4.0, "B": 4.0},
                     credits_end={"A": 7.0, "B": 8.0}, alive={"A": True, "B": True})
    txt = render_observation(build_observation(st, cfg, 1, 0, ["B"], [], [500.0], last_result=rr))
    assert "LAST ROUND" in txt and "480.0" in txt and "error 20" in txt and "earned 8 reward" in txt
    assert "510" not in txt and "error 10" not in txt         # never B's private outcome
    # no feedback at a game's first round (last_result is None)
    txt0 = render_observation(build_observation(st, cfg, 0, 0, ["B"], [], [], last_result=None))
    assert "LAST ROUND" not in txt0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all referee tests pass")
