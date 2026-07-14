"""Regression tests ensuring information metrics never leak across rounds/games."""
from analysis.metrics import cooperation, reciprocity


def _start(game=0):
    return {"event": "game_start", "game_index": game,
            "config": {"agent_ids": ["A", "B"]}}


def test_cooperation_does_not_match_a_future_round_measurement():
    events = [
        _start(),
        {"event": "round_start", "round": 0, "alive": ["A", "B"]},
        {"event": "message", "sender": "A", "to": "B", "text": "reading 777"},
        {"event": "round_end", "round": 0},
        {"event": "round_start", "round": 1, "alive": ["A", "B"]},
        {"event": "measure", "agent": "A", "value": 777.0},
        {"event": "round_end", "round": 1},
    ]
    assert cooperation(events) == {"measurements": 1, "shared": 0,
                                    "cooperation_index": 0.0}
    assert reciprocity(events)["transmissions"] == 0


def test_information_metrics_do_not_match_a_future_same_round_measurement():
    events = [
        _start(),
        {"event": "round_start", "round": 0, "alive": ["A", "B"]},
        {"event": "message", "sender": "A", "to": "B", "text": "reading 42"},
        {"event": "measure", "agent": "A", "value": 42.0},
        {"event": "round_end", "round": 0},
    ]
    assert cooperation(events)["shared"] == 0
    assert reciprocity(events)["transmissions"] == 0


def test_reused_trade_ids_are_scoped_to_their_game_and_round():
    events = [
        _start(0),
        {"event": "round_start", "round": 0, "alive": ["A", "B"]},
        {"event": "measure", "agent": "A", "value": 10.0},
        {"event": "propose_trade", "trade_id": "T1", "seller": "A", "buyer": "B",
         "claimed_value": 10.0},
        {"event": "respond_trade", "trade_id": "T1", "status": "accepted"},
        {"event": "round_end", "round": 0},
        _start(1),
        {"event": "round_start", "round": 0, "alive": ["A", "B"]},
        {"event": "measure", "agent": "B", "value": 20.0},
        {"event": "propose_trade", "trade_id": "T1", "seller": "B", "buyer": "A",
         "claimed_value": 20.0},
        {"event": "round_end", "round": 0},
    ]
    assert cooperation(events)["shared"] == 1
    assert reciprocity(events)["directed"] == {"A->B": 1}
