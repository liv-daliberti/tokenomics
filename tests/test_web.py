"""The Flask app runs a scripted game end-to-end and surfaces errors gracefully.

Uses Flask's in-process test client (no live server / port needed). Skipped
cleanly if Flask is not installed.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Point the app at a throwaway runs dir BEFORE importing it (RUNS is read at import).
os.environ["AGORA_RUNS"] = tempfile.mkdtemp(prefix="agora_web_")

try:
    from web.app import app
    _HAVE_FLASK = True
except ImportError:  # pragma: no cover - depends on env
    _HAVE_FLASK = False


def _wait_done(client, job_id, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        st = client.get(f"/status/{job_id}").get_json()["status"]
        if st in ("done", "error"):
            return st
        time.sleep(0.1)
    return "timeout"


def test_index_loads():
    if not _HAVE_FLASK:
        print("skip: flask not installed"); return
    c = app.test_client()
    r = c.get("/")
    assert r.status_code == 200 and b"Run new game" in r.data


def test_scripted_game_runs_and_renders():
    if not _HAVE_FLASK:
        print("skip: flask not installed"); return
    c = app.test_client()
    r = c.post("/new", data={"preset": "base", "backend": "scripted",
                             "policies": "honest_cooperator,bayesian_solo,liar,hoarder",
                             "seed": "7"})
    assert r.status_code == 302
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]
    assert _wait_done(c, job_id) == "done"
    page = c.get(f"/game/{job_id}")           # default = simple view
    assert page.status_code == 200
    assert b"Round 0" in page.data
    assert b"what each agent did" in page.data
    assert b"System prompt" in page.data
    # detailed view still available via the toggle
    assert b"deception rate" in c.get(f"/game/{job_id}?view=detailed").data
    # it now appears on the index
    assert job_id.encode() in c.get("/").data


def test_form_overrides_agents_and_noise():
    if not _HAVE_FLASK:
        print("skip: flask not installed"); return
    c = app.test_client()
    r = c.post("/new", data={"preset": "base", "backend": "scripted",
                             "policies": "honest_cooperator,bayesian_solo",
                             "agents": "3", "tau": "42", "framing": "cooperative",
                             "horizon": "fixed", "n_rounds": "2", "seed": "1"})
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]
    assert _wait_done(c, job_id) == "done"
    import json as _json
    meta = _json.load(open(os.path.join(os.environ["AGORA_RUNS"], f"{job_id}.json")))
    assert meta["n_agents"] == 3 and meta["tau"] == 42.0 and meta["framing"] == "cooperative"
    # the prompt shown reflects the overridden framing + noise
    page = c.get(f"/game/{job_id}").data
    assert b"assisting a different user" in page  # cooperative framing preamble


def test_build_config_preserves_privilege_and_switches_agents():
    if not _HAVE_FLASK:
        print("skip: flask not installed"); return
    from web.app import build_config
    keep = build_config({"preset": "privilege", "seed": 0, "overrides": {}})
    assert keep.tau_by_agent is not None                 # per-agent noise preserved
    switched = build_config({"preset": "privilege", "seed": 0, "overrides": {"agents": 2}})
    assert switched.tau_by_agent is None and len(switched.agent_ids) == 2


def test_cooperative_is_default_framing():
    if not _HAVE_FLASK:
        print("skip: flask not installed"); return
    c = app.test_client()
    r = c.post("/new", data={"preset": "base", "backend": "scripted",
                             "policies": "bayesian_solo", "seed": "2"})  # no framing sent
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]
    assert _wait_done(c, job_id) == "done"
    import json as _json
    meta = _json.load(open(os.path.join(os.environ["AGORA_RUNS"], f"{job_id}.json")))
    assert meta["framing"] == "cooperative"


def test_new_rejects_unknown_backend_and_provider():
    if not _HAVE_FLASK:
        print("skip: flask not installed"); return
    c = app.test_client()
    base = {"preset": "smoke", "policies": "bayesian_solo", "seed": "3"}
    assert c.post("/new", data={**base, "backend": "bogus"}).status_code == 400
    assert c.post("/new", data={**base, "backend": "scripted",
                                "provider": "bogus"}).status_code == 400
    # a key posted with a scripted run is dropped, not parked in memory forever
    from web.app import _JOB_KEYS
    r = c.post("/new", data={**base, "backend": "scripted", "api_key": "sk-oops"})
    assert r.status_code == 302 and not _JOB_KEYS


def test_trade_only_knobs_flow_through_the_form():
    if not _HAVE_FLASK:
        print("skip: flask not installed"); return
    from web.app import parse_overrides
    ov = parse_overrides({"values_via_trade_only": "1", "require_paid_trades": "1",
                          "memory": "markdown"})
    assert ov["values_via_trade_only"] is True
    assert ov["require_paid_trades"] is True
    assert ov["memory"] == "markdown"
    assert "values_via_trade_only" not in parse_overrides({"values_via_trade_only": ""})
    assert "require_paid_trades" not in parse_overrides({"require_paid_trades": ""})
    assert parse_overrides({"require_paid_trades": "0"})["require_paid_trades"] is False


def test_missing_game_gets_friendly_gone_page():
    if not _HAVE_FLASK:
        print("skip: flask not installed"); return
    c = app.test_client()
    r = c.get("/game/never-existed")
    assert r.status_code == 404 and b"temporary storage" in r.data


def test_done_game_with_missing_or_corrupt_transcript_is_handled():
    if not _HAVE_FLASK:
        print("skip: flask not installed"); return
    from web.app import _write_meta
    c = app.test_client()
    base = {"status": "done", "created": 0, "title": "broken", "backend": "scripted"}
    _write_meta({"id": "missingtx", **base})
    assert c.get("/game/missingtx").status_code == 404
    _write_meta({"id": "badtx", **base})
    with open(os.path.join(os.environ["AGORA_RUNS"], "badtx.jsonl"), "w") as fh:
        fh.write("not json\n")
    response = c.get("/game/badtx")
    assert response.status_code == 500 and b"transcript is unreadable" in response.data


def test_deleting_running_job_prevents_worker_from_republishing(monkeypatch):
    if not _HAVE_FLASK:
        print("skip: flask not installed"); return
    import web.app as webapp
    entered, release = threading.Event(), threading.Event()

    def slow_match(cfg, policies, n_games, tx):
        entered.set()
        release.wait(3)
        return None

    monkeypatch.setattr(webapp, "run_match", slow_match)
    c = app.test_client()
    response = c.post("/new", data={"preset": "smoke", "backend": "scripted",
                                           "policies": "bayesian_solo", "seed": "9"})
    jid = response.headers["Location"].rstrip("/").split("/")[-1]
    assert entered.wait(2)
    c.post(f"/delete/{jid}")
    release.set()
    time.sleep(0.1)
    assert not os.path.exists(os.path.join(os.environ["AGORA_RUNS"], jid + ".json"))


def test_orphaned_running_meta_marked_interrupted():
    if not _HAVE_FLASK:
        print("skip: flask not installed"); return
    import json as _json
    from web.app import _mark_interrupted, _write_meta
    _write_meta({"id": "orphan1", "status": "running", "created": 0,
                 "title": "t", "backend": "scripted"})
    _mark_interrupted()
    meta = _json.load(open(os.path.join(os.environ["AGORA_RUNS"], "orphan1.json")))
    assert meta["status"] == "error" and "restarted" in meta["error"]


def test_nonnumeric_knobs_do_not_crash():
    if not _HAVE_FLASK:
        print("skip: flask not installed"); return
    c = app.test_client()
    # garbage in numeric fields + a multi-dash seed must NOT 500; they fall back.
    r = c.post("/new", data={"preset": "base", "backend": "scripted",
                             "policies": "bayesian_solo",
                             "tau": "abc", "gamma": "??", "agents": "xyz", "seed": "--5"})
    assert r.status_code == 302
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]
    assert _wait_done(c, job_id) == "done"
    import json as _json
    meta = _json.load(open(os.path.join(os.environ["AGORA_RUNS"], f"{job_id}.json")))
    assert meta["tau"] == 150.0  # bad tau ignored -> base preset value


def test_blank_policies_falls_back_to_default():
    if not _HAVE_FLASK:
        print("skip: flask not installed"); return
    c = app.test_client()
    r = c.post("/new", data={"preset": "smoke", "backend": "scripted",
                             "policies": "   ", "seed": "1"})
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]
    assert _wait_done(c, job_id) == "done"  # not "error" (no ZeroDivisionError)


def test_raw_transcript_is_downloadable():
    if not _HAVE_FLASK:
        print("skip: flask not installed"); return
    c = app.test_client()
    r = c.post("/new", data={"preset": "smoke", "backend": "scripted",
                             "policies": "honest_cooperator", "seed": "1"})
    jid = r.headers["Location"].rstrip("/").split("/")[-1]
    assert _wait_done(c, jid) == "done"
    t = c.get(f"/transcript/{jid}")
    assert t.status_code == 200
    assert b"round_end" in t.data and b"message" in t.data  # full events present


def test_running_page_shows_partial_transcript():
    if not _HAVE_FLASK:
        print("skip: flask not installed"); return
    import json as _json
    from web.app import JOBS, _write_meta
    jid = "liveprogress1"
    meta = {"id": jid, "status": "running", "created": time.time(),
            "title": "Live match", "backend": "llm", "games": 5}
    _write_meta(meta)
    events = [
        {"event": "match_start", "n_games": 5},
        {"event": "game_start", "game_index": 0, "n_rounds_actual": 2,
         "config": {"agent_ids": ["A", "B"], "tau": 30, "prior_mu": 500,
                    "prior_sigma": 400, "n_rounds": 2, "reveal_horizon": True,
                    "framing": "cooperative"}},
        {"event": "round_start", "round": 0, "truth": 510, "alive": ["A", "B"],
         "credits": {"A": 4, "B": 4}},
        {"event": "message", "round": 0, "sender": "A", "to": "B",
         "text": "working on it"},
    ]
    with open(os.path.join(os.environ["AGORA_RUNS"], jid + ".jsonl"), "w") as fh:
        for event in events:
            fh.write(_json.dumps(event) + "\n")
    JOBS[jid] = {"status": "running", "error": None}
    page = app.test_client().get(f"/game/{jid}")
    assert page.status_code == 200
    assert b"Game 1 of 5" in page.data and b"Round 0" in page.data
    assert b"working on it" in page.data and b"4 events logged" in page.data
    JOBS.pop(jid, None)


def test_create_tab_has_the_form_and_index_does_not():
    if not _HAVE_FLASK:
        print("skip: flask not installed"); return
    c = app.test_client()
    assert b'action="/new"' in c.get("/create").data       # form lives on its own tab
    assert b'action="/new"' not in c.get("/").data          # main page is gallery-only


def test_delete_all_clears_the_gallery():
    if not _HAVE_FLASK:
        print("skip: flask not installed"); return
    c = app.test_client()
    r = c.post("/new", data={"preset": "smoke", "backend": "scripted",
                             "policies": "bayesian_solo", "seed": "1"})
    jid = r.headers["Location"].rstrip("/").split("/")[-1]
    assert _wait_done(c, jid) == "done"
    assert jid.encode() in c.get("/").data
    c.post("/delete_all")
    assert jid.encode() not in c.get("/").data


def test_bad_policy_surfaces_error():
    if not _HAVE_FLASK:
        print("skip: flask not installed"); return
    c = app.test_client()
    r = c.post("/new", data={"preset": "smoke", "backend": "scripted",
                             "policies": "not_a_real_policy", "seed": "1"})
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]
    assert _wait_done(c, job_id) == "error"
    page = c.get(f"/game/{job_id}")
    assert b"failed to run" in page.data and b"unknown scripted policies" in page.data


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all web tests pass")
