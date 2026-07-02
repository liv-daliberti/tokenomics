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
import re
import shutil
import string
import threading
import time
import uuid
from typing import Dict

from flask import (Flask, abort, jsonify, redirect, render_template_string,
                   request, url_for)

from agora.config import PRESETS, GameConfig
from agora.policies import REGISTRY, LLMPolicy
from agora.referee import run_match
from agora.transcripts import Transcript
from analysis.metrics import load_events, summary as metric_summary
from analysis.viz import _CSS, render_body, render_simple

RUNS = os.environ.get("AGORA_RUNS",
                      os.path.join(os.path.dirname(__file__), "..", "runs", "web"))
os.makedirs(RUNS, exist_ok=True)

app = Flask(__name__)
JOBS: Dict[str, dict] = {}          # id -> {status, error} for in-flight runs
LOCK = threading.Lock()

DEFAULT_POLICIES = "honest_cooperator,bayesian_solo,liar,hoarder"

# Simulator knobs the form may override (blank = keep the preset's value).
_INT_KNOBS = {"agents": (2, 12), "message_quota": (0, 50), "max_ticks": (1, 20),
              "n_rounds": (1, 30), "reward_max": (1, 20)}
_FLOAT_KNOBS = {"tau": (0.0, 1e6), "prior_sigma": (0.0, 1e6), "prior_mu": (-1e9, 1e9),
                "measure_cost": (0.0, 1e6), "starting_credits": (0.0, 1e6),
                "gamma": (0.0, 0.99), "survival_cost": (0.0, 1e6)}


def _clamp(v, lo, hi):
    """Clamp a value to the [lo, hi] range."""
    return max(lo, min(hi, v))


def parse_overrides(form) -> dict:
    """Read the (all-optional) simulator knobs from the form, clamped.

    Non-numeric input is ignored (falls back to the preset value) rather than
    crashing the request.
    """
    ov: dict = {}
    for k, (lo, hi) in _INT_KNOBS.items():
        raw = form.get(k, "").strip()
        if raw:
            try:
                ov[k] = int(_clamp(int(float(raw)), lo, hi))
            except ValueError:
                pass
    for k, (lo, hi) in _FLOAT_KNOBS.items():
        raw = form.get(k, "").strip()
        if raw:
            try:
                ov[k] = _clamp(float(raw), lo, hi)
            except ValueError:
                pass
    for k in ("framing", "horizon"):
        raw = form.get(k, "").strip()
        if raw:
            ov[k] = raw
    return ov


def build_config(params: dict) -> GameConfig:
    """Start from the chosen preset, then apply the form's overrides."""
    preset = PRESETS[params["preset"]]
    cfg = preset
    ov = dict(params.get("overrides", {}))

    agents = ov.pop("agents", None)
    if agents and agents != len(preset.agent_ids):
        # Only rebuild the roster when the count actually changes; this preserves
        # a preset's per-agent noise (e.g. privilege) when N is left as-is.
        ids = list(string.ascii_uppercase[:agents])
        cfg = cfg.with_(agent_ids=ids, tau_by_agent=None)

    horizon = ov.pop("horizon", None)
    if horizon == "fixed":
        cfg = cfg.with_(horizon_mode="fixed", reveal_horizon=True)
    elif horizon == "geometric":
        cfg = cfg.with_(horizon_mode="geometric", reveal_horizon=False)

    return cfg.with_(seed=params["seed"], **ov)


# --------------------------------------------------------------------------- #
# Job execution                                                               #
# --------------------------------------------------------------------------- #
def _meta_path(job_id: str) -> str:
    """Path to a job's metadata JSON in the runs dir."""
    return os.path.join(RUNS, f"{job_id}.json")


def _write_meta(meta: dict) -> None:
    # Atomic: write to a temp file then rename, so a concurrent reader (the
    # status poll on another thread) never sees a half-written file.
    """Atomically write a job's metadata (temp file + rename) so a concurrent reader never sees a partial file."""
    path = _meta_path(meta["id"])
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(meta, fh)
    os.replace(tmp, path)


def _load_meta(job_id: str) -> dict:
    """Read a job's metadata JSON."""
    with open(_meta_path(job_id)) as fh:
        return json.load(fh)


def _all_games() -> list:
    """Load every game's metadata, newest first."""
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


_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SAMPLE_RUNS = [
    ("docs/samples/qwen3-32b_5games.jsonl", "Qwen3-32B vs Qwen3-32B — 5 games, shared memory (co-evolution)"),
]


def seed_samples() -> None:
    """Populate the gallery from the committed sample transcripts, ONCE.

    A `.seeded` marker means we have seeded before, so a deliberate "Delete all"
    is not resurrected on restart. A fresh, ephemeral deploy (e.g. Render) has no
    marker and gets the curated samples."""
    marker = os.path.join(RUNS, ".seeded")
    if os.path.exists(marker):
        return
    for rel, title in _SAMPLE_RUNS:
        src = os.path.join(_REPO, rel)
        jid = "sample-" + re.sub(r"[^a-zA-Z0-9]", "-", os.path.basename(rel).replace(".jsonl", ""))
        if not os.path.exists(src) or os.path.exists(_meta_path(jid)):
            continue
        try:
            ev = load_events(src)
            s = metric_summary(ev)
            cfg = next(e for e in ev if e["event"] == "game_start")["config"]
            meta = {
                "id": jid, "status": "done", "title": title,
                "created": os.path.getmtime(src), "backend": "llm", "preset": "(sample)",
                "n_agents": len(cfg.get("agent_ids", [])), "tau": cfg.get("tau"),
                "framing": cfg.get("framing"), "n_games": s.get("n_games"),
                "rounds": len([e for e in ev if e["event"] == "round_end"]),
                "deception_rate": s["deception"]["deception_rate"],
                "cooperation": s["cooperation"]["cooperation_index"],
                "survivors": s["survivors"], "gini": s["gini_final_credits"],
                "welfare": s["welfare"], "parse_fail_rate": s["diagnostics"]["parse_fail_rate"],
            }
            shutil.copy(src, os.path.join(RUNS, jid + ".jsonl"))
            _write_meta(meta)
        except Exception:  # a bad sample must never take down the app
            continue
    try:
        open(marker, "w").close()
    except OSError:
        pass


def _run_job(job_id: str, meta: dict) -> None:
    """Run one match (build config + policies, play it, write the transcript) and update the job's metadata/status; any failure is surfaced to the UI."""
    params = meta
    try:
        cfg = build_config(params)
        ids = cfg.agent_ids
        n_games = int(params.get("games", 1))
        meta.update(n_agents=len(ids), tau=cfg.tau, framing=cfg.framing,
                    survival_cost=cfg.survival_cost, n_games=n_games,
                    horizon=("known %d-round" % cfg.n_rounds) if cfg.reveal_horizon
                             else "hidden (γ=%.2f)" % cfg.gamma)

        if params["backend"] == "llm":
            from agora.backends import OpenAIBackend
            be = OpenAIBackend(model=params["model"], base_url=params["base_url"])
            policies = {a: LLMPolicy(be, cfg, a, [p for p in ids if p != a], n_games=n_games)
                        for a in ids}
        else:
            names = [n.strip() for n in params["policies"].split(",") if n.strip()]
            if not names:
                raise ValueError(f"no scripted policies given; choose from {sorted(REGISTRY)}")
            bad = [n for n in names if n not in REGISTRY]
            if bad:
                raise ValueError(f"unknown scripted policies: {bad}; "
                                 f"choose from {sorted(REGISTRY)}")
            policies = {a: REGISTRY[names[i % len(names)]](cfg, a, ids)
                        for i, a in enumerate(ids)}

        # A match of n_games played back-to-back with the SAME (persistent)
        # policy objects, so an LLM agent keeps its memory across games.
        tx = Transcript(os.path.join(RUNS, f"{job_id}.jsonl"))
        run_match(cfg, policies, n_games, tx)
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
def _preset_data() -> dict:
    """Preset values used to pre-fill the form fields client-side."""
    out = {}
    for name, c in PRESETS.items():
        out[name] = {
            "agents": len(c.agent_ids), "tau": c.tau,
            "prior_mu": c.prior_mu, "prior_sigma": c.prior_sigma,
            "survival_cost": c.survival_cost, "n_rounds": c.n_rounds,
            "measure_cost": c.measure_cost, "starting_credits": c.starting_credits,
            "message_quota": c.message_quota, "max_ticks": c.max_ticks, "gamma": c.gamma,
            "horizon": "fixed" if c.horizon_mode == "fixed" else "geometric",
        }
    return out


def _page_ctx(page: str) -> dict:
    """Shared template context (presets, policy names, preset-fill data) for the Games / Create tabs."""
    return dict(css=_CSS, page=page, presets=sorted(PRESETS),
                policies_list=sorted(REGISTRY), default_policies=DEFAULT_POLICIES,
                preset_data=json.dumps(_preset_data()))


@app.route("/")
def index():
    """The Games tab: the About panel and the gallery of past runs."""
    return render_template_string(INDEX, games=_all_games(), **_page_ctx("games"))


@app.route("/create")
def create():
    """The Run-new-game tab: the configuration form."""
    return render_template_string(INDEX, games=[], **_page_ctx("create"))


@app.route("/delete_all", methods=["POST"])
def delete_all():
    """Clear the whole gallery (does not resurrect on restart — see seed marker)."""
    for f in os.listdir(RUNS):
        if (f.endswith(".json") or f.endswith(".jsonl")) and not f.startswith("."):
            try:
                os.remove(os.path.join(RUNS, f))
            except OSError:
                pass
    JOBS.clear()
    return redirect(url_for("index"))


@app.route("/new", methods=["POST"])
def new_game():
    """Parse the form, start a background job, and redirect to its game page."""
    f = request.form
    try:
        seed = int(f.get("seed", "").strip())
    except ValueError:
        seed = random.randint(0, 999_999)
    try:
        games = max(1, min(20, int(float(f.get("games", "5")))))
    except (ValueError, TypeError):
        games = 5
    params = {
        "preset": f.get("preset", "cooperative"),
        "seed": seed,
        "games": games,
        "backend": f.get("backend", "llm"),
        "policies": (f.get("policies") or "").strip() or DEFAULT_POLICIES,
        "model": f.get("model", "qwen3-32b"),
        "base_url": f.get("base_url", "http://localhost:8000/v1"),
        "title": f.get("title", "").strip(),
        "overrides": parse_overrides(f),
    }
    params["overrides"].setdefault("framing", "cooperative")  # cooperative by default
    if params["preset"] not in PRESETS:
        abort(400, "unknown preset")
    job_id = uuid.uuid4().hex[:10]
    if not params["title"]:
        gtag = f"{params['games']} games · " if params["games"] > 1 else ""
        params["title"] = f"{params['preset']} · {params['backend']} · {gtag}seed {params['seed']}"
    _start_job(job_id, params)
    return redirect(url_for("game", job_id=job_id))


@app.route("/game/<job_id>")
def game(job_id: str):
    """Render a finished game (simple or detailed view), or a running/error placeholder."""
    if not os.path.exists(_meta_path(job_id)):
        abort(404)
    meta = _load_meta(job_id)
    status = JOBS.get(job_id, {}).get("status", meta.get("status", "done"))

    if status == "running":
        return render_template_string(WAIT, css=_CSS, meta=meta)
    if status == "error":
        return render_template_string(ERROR, css=_CSS, meta=meta)

    view = request.args.get("view", "simple")
    events = load_events(os.path.join(RUNS, f"{job_id}.jsonl"))
    title = meta.get("title", "Agora game")
    body = render_body(events, title) if view == "detailed" else render_simple(events, title)
    return render_template_string(GAME, css=_CSS, body=body, meta=meta, view=view)


@app.route("/status/<job_id>")
def status(job_id: str):
    """JSON status of a job, for the game page's auto-refresh poll."""
    if not os.path.exists(_meta_path(job_id)):
        abort(404)
    st = JOBS.get(job_id, {}).get("status") or _load_meta(job_id).get("status", "done")
    return jsonify({"status": st})


@app.route("/transcript/<job_id>")
def transcript(job_id: str):
    """Serve the raw JSONL transcript — every event, verbatim, for full inspection."""
    path = os.path.abspath(os.path.join(RUNS, f"{job_id}.jsonl"))
    if not os.path.exists(path):
        abort(404)
    from flask import send_file
    return send_file(path, mimetype="application/x-ndjson", as_attachment=True,
                     download_name=f"agora_{job_id}.jsonl")


@app.route("/delete/<job_id>", methods=["POST"])
def delete(job_id: str):
    """Delete one game's files and drop it from the gallery."""
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
.nav{display:flex;gap:4px;margin:0 0 22px;border-bottom:1px solid var(--line);}
.nav a{padding:11px 18px;text-decoration:none;color:var(--mut);font-weight:600;border-bottom:2px solid transparent;margin-bottom:-1px;}
.nav a.active{color:var(--fg);border-bottom-color:var(--blue);}
</style></head><body><div class="wrap">{{ inner|safe }}</div>
<script>
function toggleBackend(){var el=document.querySelector('input[name=backend]:checked'); if(!el) return;
 var b=el.value, s=document.getElementById('scripted-opts'), l=document.getElementById('llm-opts');
 if(s) s.classList.toggle('hide', b!=='scripted'); if(l) l.classList.toggle('hide', b!=='llm');}
document.addEventListener('DOMContentLoaded',function(){var r=document.querySelectorAll('input[name=backend]');
 r.forEach(function(x){x.addEventListener('change',toggleBackend);});toggleBackend();});
</script></body></html>"""

INDEX = _SHELL.replace("{{ inner|safe }}", """
<div class="nav">
  <a href="/" class="{{ 'active' if page != 'create' else '' }}">Games</a>
  <a href="/create" class="{{ 'active' if page == 'create' else '' }}">＋ Run new game</a>
</div>

{% if page != 'create' %}
<h1>Agora — the Measurement Market</h1>
<p class="sub">Watch AI agents cooperate, deceive, hoard, and go bust while estimating a hidden number.</p>

<details class="panel" open>
  <summary style="cursor:pointer;font-weight:600">What am I looking at?</summary>
  <div style="color:var(--mut);font-size:14px;line-height:1.65;margin-top:12px">
    <p style="margin:0 0 10px"><b>Agora</b> is a small testbed for how AI agents behave when they must
    cooperate under scarcity. Each round a hidden number is drawn (think “cars per hour on a highway”),
    and every agent tries to <b>estimate it</b>. An agent can spend <b>credits</b> to take a noisy
    <b>measurement</b>, <b>message</b> the other agents, and <b>buy/sell</b> measurements from them.</p>
    <p style="margin:0 0 10px">The twist: budgets are tight, so no agent can nail the answer alone — they have
    to <b>pool their readings</b> to do well. But a shared number <b>can’t be verified</b>, so an agent could
    lie. Score is how close the guess is to the truth; good scores earn credits to keep playing, and
    <b>running out means elimination</b>. We watch whether they cooperate, deceive, hoard, or go bust.</p>
    <p style="margin:0 0 10px"><b>How to read a game:</b> it is split into rounds. Per agent you see its
    💭 <b>reasoning</b>, the <b>measurements</b> it took, the <b>messages</b> it sent, and any <b>trades</b> —
    then the <b>outcome</b> (true value, each agent’s answer, its error, reward, and credit balance). A
    <span class="tag lie">FABRICATED</span> tag marks a sold value that doesn’t match what the seller actually
    measured. The <b>Scoreboard</b> at the top of a run gives a quick who-won / who-survived summary.</p>
    <p style="margin:0 0 10px"><b>Each agent has its own private memory.</b> Agent A only ever sees
    <b>A’s</b> own conversation — its system prompt, its own measurements, the messages sent to it, and
    its own reasoning — and B only sees <b>B’s</b>. They never read each other’s thoughts or memory; the
    <i>only</i> way anything crosses between them is when an agent <b>chooses</b> to message or trade.
    Across a multi-game match each agent keeps <i>its own</i> growing memory, so “co-evolution” is each
    agent learning privately, never a shared mind.</p>
    <p style="margin:0"><b>Every run here is Qwen-3-32B vs Qwen-3-32B</b> — two copies of the same open model.
    Hit <a href="/create">Run new game</a> to pit two of them against each other (this needs a local
    vLLM endpoint serving the model).</p>
  </div>
</details>

<div style="display:flex;justify-content:space-between;align-items:center;margin-top:28px">
  <h2 style="font-size:20px;margin:0">Games</h2>
  {% if games %}<form method="post" action="/delete_all" style="margin:0"
      onsubmit="return confirm('Delete ALL games? This cannot be undone.')">
    <button class="ghost" style="margin:0;padding:6px 12px;font-size:12px">Delete all</button></form>{% endif %}
</div>
{% if not games %}<p class="sub">No games yet — <a href="/create">run one</a>.</p>{% endif %}
<ul class="games">
{% for g in games %}
  <li>
    <a href="/game/{{ g.id }}">{{ g.title }}</a>
    {% if g.status == 'done' %}
      <span class="m">{{ g.n_agents }} agents · {% if g.n_games and g.n_games > 1 %}{{ g.n_games }} games · {% endif %}{{ g.rounds }} rounds{% if g.tau is defined %} · τ={{ g.tau }}{% endif %}</span>
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

{% else %}
<h1>Run a new game</h1>
<form class="panel" method="post" action="/new">
  <label>Setting (preset)</label>
  <select name="preset">
    {% for p in presets %}<option value="{{p}}" {% if p=='cooperative' %}selected{% endif %}>{{p}}</option>{% endfor %}
  </select>

  <input type="hidden" name="backend" value="llm">
  <label>Agents — Qwen-3-32B via a local vLLM endpoint</label>
  <div class="row">
    <div><label>Model (served name)</label><input name="model" value="qwen3-32b"></div>
    <div><label>vLLM base URL</label><input name="base_url" value="http://localhost:8000/v1"></div>
  </div>
  <p class="m" style="color:var(--mut);font-size:12px;margin:6px 0 0">
    Serve the model first (<code>scripts/serve_qwen.sh</code>); the match runs in the background.</p>

  <label>Games in a row — played back-to-back; the agents keep their memory across all of them</label>
  <input name="games" type="number" min="1" max="20" value="5">

  <div style="border-top:1px solid var(--line);margin:20px 0 4px;padding-top:14px">
    <b>Simulator variables</b> <span class="m" style="color:var(--mut);font-size:12px">— prefilled from the preset; edit any value</span>
  </div>
  <div class="row">
    <div><label>Number of agents</label><input name="agents" type="number" min="2" max="12" value="2"></div>
    <div><label>Measurement noise τ — how far readings stray from the true value</label>
      <input name="tau" placeholder="preset"></div>
  </div>
  <div class="row">
    <div><label>Prior mean μ — the average true value</label><input name="prior_mu" placeholder="preset"></div>
    <div><label>Prior spread σ — how much the true value varies</label><input name="prior_sigma" placeholder="preset"></div>
  </div>
  <div class="row">
    <div><label>Survival cost / round — 0 = nobody dies</label><input name="survival_cost" placeholder="preset"></div>
    <div><label>Measurement cost (credits each)</label><input name="measure_cost" placeholder="preset"></div>
  </div>
  <div class="row">
    <div><label>Horizon</label>
      <select name="horizon">
        <option value="">preset</option>
        <option value="fixed">fixed — agents know the number of rounds</option>
        <option value="geometric">hidden — random end</option>
      </select></div>
    <div><label>Rounds (max)</label><input name="n_rounds" type="number" min="1" max="30" placeholder="preset"></div>
  </div>
  <div class="row">
    <div><label>Framing (default cooperative)</label>
      <select name="framing">
        <option value="cooperative" selected>cooperative</option>
        <option value="neutral">neutral</option>
        <option value="competitive">competitive</option>
      </select></div>
    <div></div>
  </div>
  <details style="margin-top:8px"><summary style="cursor:pointer;color:var(--mut);font-size:13px">more knobs…</summary>
    <div class="row">
      <div><label>Starting credits</label><input name="starting_credits" placeholder="preset"></div>
      <div><label>Messages / round</label><input name="message_quota" type="number" placeholder="preset"></div>
    </div>
    <div class="row">
      <div><label>Interaction ticks / round</label><input name="max_ticks" type="number" placeholder="preset"></div>
      <div><label>Continuation prob γ (hidden horizon)</label><input name="gamma" placeholder="preset"></div>
    </div>
  </details>

  <div class="row" style="margin-top:12px">
    <div><label>Seed (blank = random)</label><input name="seed" placeholder="random"></div>
    <div><label>Title (optional)</label><input name="title" placeholder="auto"></div>
  </div>

  <button type="submit">▶ Run new game</button>
</form>

<script>
const AGORA_PRESETS = {{ preset_data|safe }};
function fillPreset(){
  var sel = document.querySelector('select[name=preset]'); if(!sel) return;
  var p = AGORA_PRESETS[sel.value]; if(!p) return;
  ['tau','prior_mu','prior_sigma','survival_cost','n_rounds','measure_cost',
   'starting_credits','message_quota','max_ticks','gamma'].forEach(function(k){
    var el = document.querySelector('[name="'+k+'"]');
    if(el && p[k]!==undefined && p[k]!==null) el.value = p[k];
  });
  var hz = document.querySelector('select[name="horizon"]'); if(hz && p.horizon) hz.value = p.horizon;
}
document.addEventListener('DOMContentLoaded', function(){
  var sel = document.querySelector('select[name=preset]');
  if(sel){ sel.addEventListener('change', fillPreset); fillPreset(); }
});
</script>
{% endif %}
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
<div style="display:flex;justify-content:space-between;align-items:center">
  <a class="back" href="/">← all games</a>
  <div class="m" style="color:var(--mut);font-size:13px">view:
    <a href="?view=simple" style="{{ 'font-weight:700;text-decoration:none' if view=='simple' else '' }}">simple</a> ·
    <a href="?view=detailed" style="{{ 'font-weight:700;text-decoration:none' if view=='detailed' else '' }}">detailed</a>
    · <a href="/transcript/{{ meta.id }}">raw transcript ⬇</a></div>
</div>
<p class="sub" style="font-size:13px;margin:10px 0 4px">Each round shows what each agent did — its
💭 reasoning, measurements, messages, and trades — then the outcome. The <b>Scoreboard</b> summarizes
who won and who survived. <a href="/">What is this?</a></p>
{{ body|safe }}
""")


# Populate the gallery at import so a fresh (ephemeral) deploy is never empty.
seed_samples()


def main() -> None:
    """Launch the dev server (seeding the sample gallery first)."""
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    print(f"Agora web UI on http://{host}:{port}")
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
