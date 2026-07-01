"""The Flask app runs a scripted game end-to-end and surfaces errors gracefully.

Uses Flask's in-process test client (no live server / port needed). Skipped
cleanly if Flask is not installed.
"""
from __future__ import annotations

import os
import sys
import tempfile
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
    assert b"The prompt each agent is given" in page.data
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
    assert b"working together" in page  # cooperative framing preamble


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
