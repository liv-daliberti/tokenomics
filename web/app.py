"""A small Flask site for running and browsing Agora games.

Click a button to run a new game and read it as a report. Two backends:
  * scripted  — programmatic baseline agents; instant, needs no GPU.
  * llm       — real local Qwen agents via a vLLM endpoint; slower, runs in the
                background (the game page auto-refreshes until it finishes).

Run:
    pip install flask            # or: pip install -e '.[web]'
    python -m web.app            # -> http://127.0.0.1:5000
    HOST=0.0.0.0 PORT=8080 python -m web.app
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
import uuid
from typing import Dict

from flask import (Flask, abort, jsonify, redirect, render_template_string,
                   request, url_for)

from agora.config import PRESETS
from agora.policies import REGISTRY, LLMPolicy
from agora.referee import Referee
from agora.transcripts import Transcript
from analysis.metrics import load_events, summary as metric_summary
from analysis.viz import _CSS, render_body

RUNS = os.environ.get("AGORA_RUNS",
                      os.path.join(os.path.dirname(__file__), "..", "runs", "web"))
os.makedirs(RUNS, exist_ok=True)

app = Flask(__name__)
JOBS: Dict[str, dict] = {}          # id -> {status, error} for in-flight runs
LOCK = threading.Lock()

DEFAULT_POLICIES = "honest_cooperator,bayesian_solo,liar,hoarder"


# --------------------------------------------------------------------------- #
# Job execution                                                               #
# --------------------------------------------------------------------------- #
def _meta_path(job_id: str) -> str:
    return os.path.join(RUNS, f"{job_id}.json")


def _write_meta(meta: dict) -> None:
    # Atomic: write to a temp file then rename, so a concurrent reader (the
    # status poll on another thread) never sees a half-written file.
    path = _meta_path(meta["id"])
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(meta, fh)
    os.replace(tmp, path)


def _load_meta(job_id: str) -> dict:
    with open(_meta_path(job_id)) as fh:
        return json.load(fh)


def _all_games() -> list:
    games = []
    for f in os.listdir(RUNS):
        if f.endswith(".json"):
            try:
                games.append(_load_meta(f[:-5]))
            except (OSError, ValueError):
                continue
    return sorted(games, key=lambda m: m.get("created", 0), reverse=True)


def _start_job(job_id: str, params: dict) -> None:
    """Create the job record synchronously (so /status finds it immediately),
    then run it on a background thread."""
    meta = {"id": job_id, "status": "running", "created": time.time(), **params}
    _write_meta(meta)
    with LOCK:
        JOBS[job_id] = {"status": "running", "error": None}
    threading.Thread(target=_run_job, args=(job_id, meta), daemon=True).start()


def _run_job(job_id: str, meta: dict) -> None:
    params = meta
    try:
        seed = params["seed"]
        cfg = PRESETS[params["preset"]].with_(seed=seed)
        ids = cfg.agent_ids

        if params["backend"] == "llm":
            from agora.backends import OpenAIBackend
            be = OpenAIBackend(model=params["model"], base_url=params["base_url"])
            policies = {a: LLMPolicy(be, cfg, a, [p for p in ids if p != a]) for a in ids}
        else:
            names = [n.strip() for n in params["policies"].split(",") if n.strip()]
            bad = [n for n in names if n not in REGISTRY]
            if bad:
                raise ValueError(f"unknown scripted policies: {bad}; "
                                 f"choose from {sorted(REGISTRY)}")
            policies = {a: REGISTRY[names[i % len(names)]](cfg, a, ids)
                        for i, a in enumerate(ids)}

        tx = Transcript(os.path.join(RUNS, f"{job_id}.jsonl"))
        Referee(cfg, policies, tx).run()
        tx.close()

        s = metric_summary(tx.events)
        meta.update(
            status="done",
            rounds=len([e for e in tx.events if e["event"] == "round_end"]),
            deception_rate=s["deception"]["deception_rate"],
            cooperation=s["cooperation"]["cooperation_index"],
            survivors=s["survivors"], n_agents=s["n_agents"],
            gini=s["gini_final_credits"], welfare=s["welfare"],
            parse_fail_rate=s["diagnostics"]["parse_fail_rate"],
        )
        _write_meta(meta)
        with LOCK:
            JOBS[job_id] = {"status": "done", "error": None}
    except Exception as exc:  # surface any failure (e.g. no LLM endpoint) to the UI
        meta.update(status="error", error=f"{type(exc).__name__}: {exc}")
        _write_meta(meta)
        with LOCK:
            JOBS[job_id] = {"status": "error", "error": meta["error"]}


# --------------------------------------------------------------------------- #
# Routes                                                                       #
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template_string(INDEX, css=_CSS, presets=sorted(PRESETS),
                                  policies_list=sorted(REGISTRY),
                                  default_policies=DEFAULT_POLICIES,
                                  games=_all_games())


@app.route("/new", methods=["POST"])
def new_game():
    f = request.form
    seed = f.get("seed", "").strip()
    params = {
        "preset": f.get("preset", "base"),
        "seed": int(seed) if seed else random.randint(0, 999_999),
        "backend": f.get("backend", "scripted"),
        "policies": f.get("policies", DEFAULT_POLICIES),
        "model": f.get("model", "qwen3-32b"),
        "base_url": f.get("base_url", "http://localhost:8000/v1"),
        "title": f.get("title", "").strip(),
    }
    if params["preset"] not in PRESETS:
        abort(400, "unknown preset")
    job_id = uuid.uuid4().hex[:10]
    if not params["title"]:
        params["title"] = f"{params['preset']} · {params['backend']} · seed {params['seed']}"
    _start_job(job_id, params)
    return redirect(url_for("game", job_id=job_id))


@app.route("/game/<job_id>")
def game(job_id: str):
    if not os.path.exists(_meta_path(job_id)):
        abort(404)
    meta = _load_meta(job_id)
    status = JOBS.get(job_id, {}).get("status", meta.get("status", "done"))

    if status == "running":
        return render_template_string(WAIT, css=_CSS, meta=meta)
    if status == "error":
        return render_template_string(ERROR, css=_CSS, meta=meta)

    events = load_events(os.path.join(RUNS, f"{job_id}.jsonl"))
    body = render_body(events, meta.get("title", "Agora game"))
    return render_template_string(GAME, css=_CSS, body=body, meta=meta)


@app.route("/status/<job_id>")
def status(job_id: str):
    if not os.path.exists(_meta_path(job_id)):
        abort(404)
    st = JOBS.get(job_id, {}).get("status") or _load_meta(job_id).get("status", "done")
    return jsonify({"status": st})


@app.route("/delete/<job_id>", methods=["POST"])
def delete(job_id: str):
    for ext in (".json", ".jsonl"):
        p = os.path.join(RUNS, f"{job_id}{ext}")
        if os.path.exists(p):
            os.remove(p)
    JOBS.pop(job_id, None)
    return redirect(url_for("index"))


# --------------------------------------------------------------------------- #
# Templates (inline; the report CSS is shared with the standalone viewer)      #
# --------------------------------------------------------------------------- #
_SHELL = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agora</title><style>{{ css|safe }}
.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin:18px 0;}
label{display:block;color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.4px;margin:12px 0 4px;}
input,select{width:100%;padding:9px 11px;background:#0f1115;color:var(--fg);border:1px solid var(--line);border-radius:9px;font:inherit;}
.row{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
button{background:var(--blue);color:#04121f;border:0;border-radius:10px;padding:11px 18px;font:600 15px/1 inherit;cursor:pointer;margin-top:16px;}
button.ghost{background:#20242e;color:var(--fg);}
.games{list-style:none;padding:0;margin:0;}
.games li{display:flex;align-items:center;gap:12px;padding:11px 0;border-bottom:1px solid var(--line);}
.games a{flex:1;text-decoration:none;color:var(--fg);font-weight:600;}
.games .m{color:var(--mut);font-size:13px;}
.pill{font-size:11px;padding:2px 8px;border-radius:20px;background:#20242e;color:var(--mut);}
.pill.bad{background:#3a1c1a;color:var(--red);} .pill.ok{background:#1c3a24;color:var(--green);}
.back{color:var(--blue);text-decoration:none;font-size:14px;}
.hide{display:none;}
</style></head><body><div class="wrap">{{ inner|safe }}</div>
<script>
function toggleBackend(){var b=document.querySelector('input[name=backend]:checked').value;
 document.getElementById('scripted-opts').classList.toggle('hide', b!=='scripted');
 document.getElementById('llm-opts').classList.toggle('hide', b!=='llm');}
document.addEventListener('DOMContentLoaded',function(){var r=document.querySelectorAll('input[name=backend]');
 r.forEach(function(x){x.addEventListener('change',toggleBackend);});toggleBackend&&toggleBackend();});
</script></body></html>"""

INDEX = _SHELL.replace("{{ inner|safe }}", """
<h1>Agora — the Measurement Market</h1>
<p class="sub">Run a game of agents estimating a hidden value; watch cooperation, fraud, and ruin.</p>

<form class="panel" method="post" action="/new">
  <label>Setting (preset)</label>
  <select name="preset">
    {% for p in presets %}<option value="{{p}}" {% if p=='base' %}selected{% endif %}>{{p}}</option>{% endfor %}
  </select>

  <label>Agents</label>
  <div>
    <label style="text-transform:none;letter-spacing:0;display:inline;margin-right:16px;">
      <input type="radio" name="backend" value="scripted" checked style="width:auto"> scripted (instant, no GPU)</label>
    <label style="text-transform:none;letter-spacing:0;display:inline;">
      <input type="radio" name="backend" value="llm" style="width:auto"> Qwen via vLLM (slower)</label>
  </div>

  <div id="scripted-opts">
    <label>Scripted policies (cycled over agents)</label>
    <input name="policies" value="{{ default_policies }}">
    <p class="m" style="color:var(--mut);font-size:12px;margin:6px 0 0">
      available: {{ policies_list|join(', ') }}</p>
  </div>

  <div id="llm-opts" class="hide">
    <div class="row">
      <div><label>Model (served name)</label><input name="model" value="qwen3-32b"></div>
      <div><label>vLLM base URL</label><input name="base_url" value="http://localhost:8000/v1"></div>
    </div>
    <p class="m" style="color:var(--mut);font-size:12px;margin:6px 0 0">
      Start the server first: <code>scripts/serve_qwen.sh</code>. Runs in the background.</p>
  </div>

  <div class="row">
    <div><label>Seed (blank = random)</label><input name="seed" placeholder="random"></div>
    <div><label>Title (optional)</label><input name="title" placeholder="auto"></div>
  </div>

  <button type="submit">▶ Run new game</button>
</form>

<h1 style="font-size:20px;margin-top:34px">Games</h1>
{% if not games %}<p class="sub">No games yet — run one above.</p>{% endif %}
<ul class="games">
{% for g in games %}
  <li>
    <a href="/game/{{ g.id }}">{{ g.title }}</a>
    {% if g.status == 'done' %}
      <span class="m">{{ g.rounds }} rounds</span>
      {% if g.deception_rate is defined and g.deception_rate == g.deception_rate %}
        <span class="pill {{ 'bad' if g.deception_rate>0 else 'ok' }}">deception {{ '%.2f'|format(g.deception_rate) }}</span>{% endif %}
      <span class="pill">survivors {{ g.survivors }}/{{ g.n_agents }}</span>
    {% elif g.status == 'running' %}<span class="pill">running…</span>
    {% else %}<span class="pill bad">error</span>{% endif %}
    <form method="post" action="/delete/{{ g.id }}" onsubmit="return confirm('Delete this game?')">
      <button class="ghost" style="margin:0;padding:5px 10px;font-size:12px">✕</button></form>
  </li>
{% endfor %}
</ul>
""")

WAIT = _SHELL.replace("{{ inner|safe }}", """
<meta http-equiv="refresh" content="2">
<a class="back" href="/">← all games</a>
<h1>{{ meta.title }}</h1>
<div class="panel"><p class="sub" style="margin:0">⏳ Running the agents… this page refreshes automatically.
{% if meta.backend == 'llm' %} Real Qwen games take a while (sequential turns, many model calls).{% endif %}</p></div>
""")

ERROR = _SHELL.replace("{{ inner|safe }}", """
<a class="back" href="/">← all games</a>
<h1>{{ meta.title }}</h1>
<div class="panel"><p class="sub" style="margin:0 0 8px">This game failed to run:</p>
<pre style="color:var(--red);white-space:pre-wrap;margin:0">{{ meta.error }}</pre>
{% if meta.backend == 'llm' %}<p class="m" style="color:var(--mut);margin-top:12px">
Is the vLLM server up at <code>{{ meta.base_url }}</code>? Start it with <code>scripts/serve_qwen.sh</code>,
and install the client with <code>pip install openai</code>.</p>{% endif %}</div>
""")

GAME = _SHELL.replace("{{ inner|safe }}", """
<a class="back" href="/">← all games</a>
{{ body|safe }}
""")


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    # Seed the gallery with one instant game so a first visit isn't empty.
    if not _all_games():
        jid = uuid.uuid4().hex[:10]
        _start_job(jid, {"preset": "base", "seed": 7, "backend": "scripted",
                         "policies": DEFAULT_POLICIES, "model": "qwen3-32b",
                         "base_url": "http://localhost:8000/v1",
                         "title": "base · scripted · seed 7 (example)"})
    print(f"Agora web UI on http://{host}:{port}")
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
