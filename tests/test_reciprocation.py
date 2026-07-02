"""Prove the reciprocation plumbing works end to end.

These pin down that an agent CAN receive another agent's message/trade and act on
it — so if a real LLM never reciprocates, that is a model choice, not a bug.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agora.config import GameConfig
from agora.observation import render_observation
from agora.policies import REGISTRY
from agora.policies.base import Policy, ToolInvocation
from agora.policies.scripted import HonestCooperator
from agora.referee import Referee
from agora.types import Action, ActionType


class Scripted(Policy):
    """Emits a fixed list of actions on each successive turn; records what it saw."""

    def __init__(self, plans):
        self.plans = plans          # list[ list[Action] ] — one per turn
        self.turn = 0
        self.observations = []
        self._q = []

    def reset_round(self, r):
        pass

    def start_turn(self, text, obs):
        self.observations.append(obs)
        acts = list(self.plans[self.turn]) if self.turn < len(self.plans) else []
        self.turn += 1
        acts.append(Action(ActionType.END_TURN))
        self._q = [ToolInvocation(f"c{i}", a.type.value, a) for i, a in enumerate(acts)]

    def next_actions(self):
        q, self._q = self._q, []
        return q

    def observe_results(self, results):
        pass

    def received_texts(self):
        return [m["text"] for o in self.observations for m in o["inbox"]]


class Accepter(Policy):
    """Accepts any trade offered to it as soon as it appears in the observation."""

    def __init__(self):
        self.observations = []
        self._q = []

    def reset_round(self, r):
        pass

    def start_turn(self, text, obs):
        self.observations.append(obs)
        acts = [Action(ActionType.RESPOND_TRADE, {"trade_id": t["trade_id"], "accept": True})
                for t in obs["pending_trades"]]
        acts.append(Action(ActionType.END_TURN))
        self._q = [ToolInvocation(f"c{i}", a.type.value, a) for i, a in enumerate(acts)]

    def next_actions(self):
        q, self._q = self._q, []
        return q

    def observe_results(self, results):
        pass


def _cfg(**kw):
    return GameConfig(agent_ids=["A", "B"], horizon_mode="fixed", n_rounds=1,
                      max_ticks=5, seed=0, **kw)


def test_messages_are_delivered_both_directions():
    # Both send on turn 0; each must SEE the other's message in a later observation.
    cfg = _cfg()
    A = Scripted([[Action(ActionType.SEND_MESSAGE, {"to": "B", "text": "from A: I measured 512"})]])
    B = Scripted([[Action(ActionType.SEND_MESSAGE, {"to": "A", "text": "from B: I measured 488"})]])
    Referee(cfg, {"A": A, "B": B}).run()
    assert any("from A" in t for t in B.received_texts()), "B never received A's message"
    assert any("from B" in t for t in A.received_texts()), "A never received B's reply"


def test_message_text_reaches_the_rendered_prompt():
    # The message must appear in the TEXT an LLM would be shown, not just the dict.
    cfg = _cfg()
    A = Scripted([[Action(ActionType.SEND_MESSAGE, {"to": "B", "text": "PLEASE POOL WITH ME"})]])
    B = Scripted([[]])
    Referee(cfg, {"A": A, "B": B}).run()
    seen = [o for o in B.observations if any("PLEASE POOL" in m["text"] for m in o["inbox"])]
    assert seen, "message not in B's observation"
    assert "PLEASE POOL WITH ME" in render_observation(seen[0]), "message not rendered into prompt text"


def test_trade_offer_is_visible_and_acceptable():
    cfg = _cfg(enable_trading=True)
    # A offers to sell a value to B for 1 credit on turn 0.
    A = Scripted([[Action(ActionType.PROPOSE_TRADE, {"to": "B", "price": 1.0, "claimed_value": 617.0})]])
    B = Accepter()
    res = Referee(cfg, {"A": A, "B": B}).run()
    # B saw the pending trade and accepted; escrow moved credits and delivered the value.
    assert any(o["pending_trades"] for o in B.observations), "B never saw the trade offer"
    bought = res.states["B"].purchased
    assert bought and abs(bought[0]["claimed_value"] - 617.0) < 1e-9, "B did not receive the sold value"
    # buyer paid, seller was paid (escrow settled)
    assert res.states["A"].credits > cfg.starting_credits - 1e-9  # A gained the price


def test_cooperator_incorporates_received_values():
    # A reciprocating policy must fold received readings into its answer.
    cfg = _cfg()
    coop = HonestCooperator(cfg, "A", ["A", "B"])
    coop.reset_round(0)
    obs = {
        "my_measurements": [400.0], "purchased": [],
        "inbox": [{"from": "B", "to": "all", "text": "MEASUREMENT 600.00", "tick": 0}],
        "pending_trades": [], "messages_left": 5, "ticks_left": 1, "credits": 2.0,
        "prior_mu": 500.0, "prior_sigma": 150.0, "eliminated": [],
    }
    coop.start_turn("", obs)                 # harvests the broadcast into memory
    assert any(abs(v - 600.0) < 1e-6 for _, v in coop._received), "did not remember the shared reading"
    est = coop._estimate(obs)
    assert 400.0 < est < 600.0, f"cooperator ignored the shared reading (est={est})"


def test_two_cooperators_pool_and_beat_two_solos():
    # End-to-end: mutual reciprocation makes both agents more accurate than solos.
    import statistics as st

    def mean_err(spec):
        errs = []
        for s in range(8):
            cfg = GameConfig(agent_ids=["A", "B"], horizon_mode="fixed", n_rounds=1,
                             max_ticks=5, seed=s)
            pols = {a: REGISTRY[spec](cfg, a, cfg.agent_ids) for a in cfg.agent_ids}
            g = Referee(cfg, pols).run()
            errs += [e for rr in g.rounds for e in rr.errors.values() if e == e]
        return st.mean(errs)

    assert mean_err("honest_cooperator") < mean_err("bayesian_solo"), \
        "pooling should beat solo — reciprocation must actually help"


def test_last_tick_message_not_dropped():
    # A message delivered on a round's final tick (after the recipient's turn)
    # must not be lost — it carries into the next round's first observation.
    cfg = GameConfig(agent_ids=["A", "B"], horizon_mode="fixed", n_rounds=2, max_ticks=1, seed=0)
    A = Scripted([[Action(ActionType.SEND_MESSAGE, {"to": "B", "text": "R0 from A"})],
                  [Action(ActionType.SEND_MESSAGE, {"to": "B", "text": "R1 from A"})]])
    B = Scripted([[], [], []])
    Referee(cfg, {"A": A, "B": B}).run()
    assert any("R0 from A" in t for t in B.received_texts()), "last-tick message was dropped"


def test_cooperation_required_preset_kills_solos():
    # In the cooperation_required preset a non-cooperating agent almost never
    # survives, while cooperators usually do.
    from agora.config import PRESETS

    def survival(spec, seeds=20):
        cfg0 = PRESETS["cooperation_required"]
        ids = cfg0.agent_ids
        alive = tot = 0
        for s in range(seeds):
            cfg = cfg0.with_(seed=s)
            pols = {x: REGISTRY[spec](cfg, x, ids) for x in ids}
            g = Referee(cfg, pols).run()
            for x in ids:
                alive += int(g.states[x].alive)
                tot += 1
        return alive / tot

    solo = survival("bayesian_solo")
    coop = survival("honest_cooperator")
    assert solo < 0.25, f"a solo strategy should almost never survive (got {solo:.0%})"
    assert coop > 0.6, f"cooperators should usually survive (got {coop:.0%})"
    assert coop > solo + 0.4, "cooperation must clearly beat going it alone"


def test_final_answer_pass_lets_agent_update_its_guess():
    # An agent submits an early guess, then the final-answer turn lets it revise;
    # the revised value is what is scored.
    class Updater(Policy):
        def reset_round(self, r):
            pass

        def start_turn(self, text, obs):
            val = 222.0 if obs.get("final_answer") else 111.0
            self._q = [
                ToolInvocation("s", "submit_estimate",
                               Action(ActionType.SUBMIT_ESTIMATE, {"value": val})),
                ToolInvocation("e", "end_turn", Action(ActionType.END_TURN)),
            ]

        def next_actions(self):
            q, self._q = self._q, []
            return q

        def observe_results(self, results):
            pass

    cfg = GameConfig(agent_ids=["A"], horizon_mode="fixed", n_rounds=1, max_ticks=2,
                     final_answer_pass=True)
    res = Referee(cfg, {"A": Updater()}).run()
    assert res.rounds[0].estimates["A"] == 222.0, "final-answer pass must allow a revision"

    cfg_off = cfg.with_(final_answer_pass=False)
    res_off = Referee(cfg_off, {"A": Updater()}).run()
    assert res_off.rounds[0].estimates["A"] == 111.0, "without the pass, no final revision"


def test_values_via_trade_only_redacts_chat_numbers():
    # In trade-only mode, a number sent in a message is hidden from the recipient.
    cfg = GameConfig(agent_ids=["A", "B"], horizon_mode="fixed", n_rounds=1,
                     max_ticks=3, values_via_trade_only=True)
    A = Scripted([[Action(ActionType.SEND_MESSAGE, {"to": "B", "text": "my reading is 617.5"})]])
    B = Scripted([[]])
    Referee(cfg, {"A": A, "B": B}).run()
    got = " ".join(B.received_texts())
    assert "617" not in got, "trade-only must hide numbers in chat"
    assert got.strip(), "the message should still be delivered (minus the number)"


def test_complementary_preset_is_a_true_two_agent_wall():
    # With complementary tools (theta = X + Y, each agent reads only its part),
    # NO solo strategy survives — not even passive hoarding — because a lone agent
    # is structurally blind to the other's part. Only cooperation survives.
    from agora.config import PRESETS

    def survival(spec, seeds=30):
        cfg0 = PRESETS["complementary"]
        ids = cfg0.agent_ids
        alive = tot = 0
        for s in range(seeds):
            cfg = cfg0.with_(seed=s)
            pols = {x: REGISTRY[spec](cfg, x, ids) for x in ids}
            g = Referee(cfg, pols).run()
            for x in ids:
                alive += int(g.states[x].alive)
                tot += 1
        return alive / tot

    coop = survival("honest_cooperator")
    solo = survival("bayesian_solo")
    hoard = survival("hoarder")
    liar = survival("liar")
    assert coop > 0.7, f"cooperators must survive (got {coop:.0%})"
    assert solo < 0.1, f"solo must be blind and die (got {solo:.0%})"
    assert hoard < 0.2, f"even a hoarder must die (got {hoard:.0%})"
    assert liar < 0.1, f"lying must die (got {liar:.0%})"


def test_complementary_measurement_returns_own_component():
    # A's measurements should center on its component (~mu/N), not on theta.
    from agora.referee import Referee
    cfg = GameConfig(agent_ids=["A", "B"], complementary=True, horizon_mode="fixed",
                     n_rounds=1, tau=1.0, prior_mu=500.0, prior_sigma=150.0, seed=0)
    res = Referee(cfg, {a: REGISTRY["bayesian_solo"](cfg, a, cfg.agent_ids)
                        for a in cfg.agent_ids}).run()
    ev = res.transcript.events
    truth = next(e["truth"] for e in ev if e["event"] == "round_start")
    a_meas = [e["value"] for e in ev if e["event"] == "measure" and e["agent"] == "A"]
    # A's readings are of its component (~250), far below the total theta (~500).
    assert a_meas and all(abs(v) < 0.9 * truth for v in a_meas), "A seems to measure the whole theta"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all reciprocation tests pass")
