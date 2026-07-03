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
from analysis.viz import _CSS, render_body, render_comparison, render_simple

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

# Reuse the dose-response renderer (a script) to serve /gradient in-app.
import importlib.util as _ilu  # noqa: E402
_gspec = _ilu.spec_from_file_location("gradient_report",
                                      os.path.join(_REPO, "scripts", "gradient_report.py"))
_gradient = _ilu.module_from_spec(_gspec)
_gspec.loader.exec_module(_gradient)

_SAMPLE_RUNS = [
    ("docs/samples/qwen3-32b_5games.jsonl", "Qwen3-32B vs Qwen3-32B — HARD wall (paired bias; solo is fatal)"),
    ("docs/samples/qwen3-32b_5games_medium.jsonl", "Qwen3-32B vs Qwen3-32B — MEDIUM wall (solo usually dies)"),
    ("docs/samples/qwen3-32b_5games_soft.jsonl", "Qwen3-32B vs Qwen3-32B — SOFT wall (solo survives, worse)"),
]


def seed_samples() -> None:
    """Keep the gallery's curated samples in sync with `_SAMPLE_RUNS`, idempotently.

    A `.seeded` file records which sample ids we have added. On start we (1) drop
    any auto-seeded `sample-*` game that is no longer curated — e.g. an old sample
    for a mechanic we removed — (2) leave user-run games and user-deleted samples
    alone, and (3) add any current sample that isn't present yet. A fresh deploy
    with no file gets the full current set."""
    marker = os.path.join(RUNS, ".seeded")
    try:
        seeded = {ln.strip() for ln in open(marker) if ln.strip()}
    except OSError:
        seeded = set()

    current = {}  # jid -> (rel, title)
    for rel, title in _SAMPLE_RUNS:
        jid = "sample-" + re.sub(r"[^a-zA-Z0-9]", "-", os.path.basename(rel).replace(".jsonl", ""))
        current[jid] = (rel, title)

    # (1) remove stale curated samples: any sample-* game not in the current set
    for f in os.listdir(RUNS):
        if f.startswith("sample-") and f.endswith(".json"):
            jid = f[:-5]
            if jid not in current:
                for ext in (".json", ".jsonl"):
                    p = os.path.join(RUNS, jid + ext)
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except OSError:
                            pass
                seeded.discard(jid)

    # (2)+(3) add any current sample we have not added and the user has not deleted
    for jid, (rel, title) in current.items():
        src = os.path.join(_REPO, rel)
        if jid in seeded or os.path.exists(_meta_path(jid)) or not os.path.exists(src):
            seeded.add(jid)
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
            seeded.add(jid)
        except Exception:  # a bad sample must never take down the app
            continue
    try:
        with open(marker, "w") as fh:
            fh.write("\n".join(sorted(seeded)))
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


def _gradient_charts() -> tuple:
    """(embeddable dose-response charts HTML, its CSS) from the gradient data, or
    ('', '') if there is none — used to show the graphs on the home story page."""
    try:
        for base in ("docs/samples/gradient", "runs/qwen"):
            rows = _gradient.collect(os.path.join(_REPO, base, "grad_b*.jsonl"))
            if rows:
                return _gradient.charts_block(rows), _gradient.CHART_CSS
    except Exception:
        pass
    return "", ""


def _page_ctx(page: str) -> dict:
    """Shared template context (presets, policy names, preset-fill data, and the
    embedded dose-response charts) for the Games / Create tabs."""
    charts, chart_css = _gradient_charts() if page != "create" else ("", "")
    return dict(css=_CSS, page=page, presets=sorted(PRESETS),
                policies_list=sorted(REGISTRY), default_policies=DEFAULT_POLICIES,
                preset_data=json.dumps(_preset_data()),
                gradient_charts=charts, chart_css=chart_css)


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


@app.route("/compare")
def compare():
    """Side-by-side metrics table for the finished games (hard wall vs soft wall, etc.)."""
    runs = []
    for g in _all_games():
        if g.get("status") not in (None, "done"):
            continue
        path = os.path.join(RUNS, f"{g['id']}.jsonl")
        if os.path.exists(path):
            runs.append((g.get("title") or g["id"], load_events(path)))
    body = (render_comparison(runs) if runs else
            '<h1>Compare runs</h1><p class="sub">No finished games to compare yet.</p>')
    return render_template_string(COMPARE, css=_CSS, body=body)


@app.route("/gradient")
def gradient():
    """The interdependence dose-response report: offset (bias σ) vs cooperation,
    reciprocity, survival, and messaging, as small-multiple charts."""
    rows = []
    for base in ("docs/samples/gradient", "runs/qwen"):  # committed data, then local runs
        rows = _gradient.collect(os.path.join(_REPO, base, "grad_b*.jsonl"))
        if rows:
            break
    if not rows:
        return render_template_string(
            COMPARE, css=_CSS,
            body='<h1>Interdependence gradient</h1><p class="sub">No gradient runs yet — '
                 'run the offset sweep (<code>scripts/agora_qwen_gradient.slurm</code>) to populate it.</p>')
    nav = ('<a href="/" style="position:fixed;top:12px;left:14px;z-index:99;'
           'font:13px/1 system-ui,sans-serif;color:#3987e5;text-decoration:none;'
           'background:rgba(20,22,30,.72);padding:7px 11px;border-radius:9px">← all games</a>')
    return nav + _gradient.render(rows)


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
<style>
  {{ chart_css|safe }}
  :root{--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;}
  .hero{padding:8px 0 4px;}
  .eyebrow{font:600 12px/1 var(--mono);letter-spacing:.18em;text-transform:uppercase;color:var(--blue);margin:0 0 16px;}
  .lead{font-size:clamp(30px,5.2vw,52px);line-height:1.04;letter-spacing:-.022em;font-weight:750;margin:0 0 18px;text-wrap:balance;}
  .lead em{font-style:normal;color:var(--blue);}
  .dek{font-size:19px;line-height:1.5;color:var(--mut);max-width:62ch;margin:0;}
  .dek em{font-style:normal;color:var(--fg);font-weight:500;}
  .sec{margin:52px 0 0;scroll-margin-top:20px;}
  .sec-eyebrow{font:600 12px/1 var(--mono);letter-spacing:.16em;text-transform:uppercase;color:var(--blue);margin:0 0 8px;}
  .sec-h{font-size:clamp(22px,3vw,29px);font-weight:730;letter-spacing:-.02em;line-height:1.12;margin:0 0 14px;text-wrap:balance;}
  .prose{font-size:16px;line-height:1.64;color:var(--mut);max-width:64ch;}
  .prose p{margin:0 0 13px;} .prose b{color:var(--fg);font-weight:600;} .prose:last-child{margin-bottom:0;}
  .steps{display:flex;flex-direction:column;gap:12px;margin:6px 0 0;}
  .step{display:flex;gap:16px;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:17px 20px;}
  .step-n{flex:none;width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;font:700 14px/1 var(--mono);color:var(--blue);border:1px solid var(--line);background:var(--bg);}
  .step h4{font-size:16px;margin:3px 0 6px;letter-spacing:-.01em;}
  .step p{color:var(--mut);font-size:14px;line-height:1.6;margin:0;} .step b{color:var(--fg);font-weight:640;}
  .mechanic{margin:14px 0 2px;background:var(--bg);border:1px solid var(--line);border-radius:12px;padding:15px 18px;}
  .mech-truth{font-size:13px;color:var(--mut);margin:0 0 8px;} .mech-truth b{color:var(--fg);font-variant-numeric:tabular-nums;}
  .mech-row{display:flex;align-items:center;gap:12px;padding:5px 0;font-size:14px;color:var(--mut);}
  .mech-row .tg{font:700 10px/1 var(--mono);letter-spacing:.05em;text-transform:uppercase;padding:4px 7px;border-radius:5px;white-space:nowrap;}
  .tg.a{color:var(--blue);background:rgba(90,169,230,.14);} .tg.b{color:var(--amber);background:rgba(230,179,90,.14);} .tg.ok{color:var(--green);background:rgba(90,209,154,.14);}
  .mech-row .num{font-size:19px;font-weight:700;color:var(--fg);font-variant-numeric:tabular-nums;min-width:50px;}
  .mech-row.avg{border-top:1px solid var(--line);margin-top:5px;padding-top:11px;}
  .cta{font:600 14px/1 inherit;color:var(--blue);text-decoration:none;white-space:nowrap;}
  .cta:hover{text-decoration:underline;}
  .stat-row{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:16px 0 0;}
  @media(max-width:640px){.stat-row{grid-template-columns:1fr;}}
  .stat{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;position:relative;overflow:hidden;}
  .stat::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;}
  .stat.soft::before{background:var(--red);} .stat.med::before{background:var(--amber);} .stat.hard::before{background:var(--green);}
  .stat .k{font:600 11px/1 var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--mut);}
  .stat .v{font-size:40px;font-weight:750;letter-spacing:-.02em;margin:9px 0 3px;font-variant-numeric:tabular-nums;}
  .stat.soft .v{color:var(--red);} .stat.med .v{color:var(--amber);} .stat.hard .v{color:var(--green);}
  .stat .d{font-size:12.5px;color:var(--mut);line-height:1.45;}
  .note{font-size:14px;color:var(--mut);max-width:66ch;margin:14px 0 0;line-height:1.55;} .note b{color:var(--fg);} .note a{color:var(--blue);}
  .meaning{border-left:3px solid var(--blue);padding:2px 0 2px 20px;margin:6px 0 0;}
  .meaning p{font-size:21px;line-height:1.4;color:var(--fg);font-weight:520;margin:0 0 14px;max-width:60ch;letter-spacing:-.01em;}
  .meaning p b{color:var(--blue);}
  .meaning .sub{font-size:15px;color:var(--mut);font-weight:400;line-height:1.55;} .meaning .sub b{color:var(--fg);}
  .feat{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:16px 0 0;}
  @media(max-width:640px){.feat{grid-template-columns:1fr;}}
  .fcard{display:block;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;text-decoration:none;color:var(--fg);transition:border-color .12s,transform .12s;}
  .fcard:hover{border-color:var(--blue);transform:translateY(-1px);}
  .fcard .ft{font:700 11px/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;}
  .fcard.hard .ft{color:var(--green);} .fcard.soft .ft{color:var(--red);}
  .fcard h4{font-size:16px;margin:9px 0 6px;letter-spacing:-.01em;}
  .fcard p{color:var(--mut);font-size:13px;line-height:1.55;margin:0;}
  .explore-links{display:flex;gap:18px;flex-wrap:wrap;margin:16px 0 0;}
  .gallery-head{display:flex;justify-content:space-between;align-items:baseline;margin:34px 0 2px;}
  .gallery-head h2{font-size:18px;margin:0;letter-spacing:-.01em;color:var(--mut);font-weight:600;}
</style>

<header class="hero">
  <p class="eyebrow">Agora · a multi-agent LLM study</p>
  <h1 class="lead">Do language-model agents cooperate?<br><em>Only when they have to.</em></h1>
  <p class="dek">We built a small world where two AI agents can either go it alone or help each other — and
    a dial that controls <em>how much they need to</em>. This is what we did, what we found, and what it means.</p>
</header>

<section class="sec">
  <p class="sec-eyebrow">Why we did this</p>
  <h2 class="sec-h">Multi-agent AI is coming. Does it actually collaborate?</h2>
  <div class="prose">
    <p>Agents that negotiate, delegate, and split work are arriving fast — but a basic question is unanswered:
      when cooperation <b>would help</b>, do language-model agents take it, or go it alone? We built a small,
      controlled world where cooperation is <b>measurable</b> and deception is <b>verifiable</b>, and watched
      two copies of the same model play it out.</p>
  </div>
</section>

<section class="sec">
  <p class="sec-eyebrow">What we built · Agora</p>
  <h2 class="sec-h">Two agents, one hidden number, tight budgets</h2>
  <div class="prose">
    <p>Each round a hidden number is drawn. Two identical <b>Qwen-3-32B</b> agents each try to estimate it.
      An agent can <b>measure</b> (a noisy reading that costs credits), <b>message</b> the other, <b>trade</b>
      readings, or <b>give</b> credits. Scoring is non-competitive — you're judged only on your <b>own</b>
      accuracy — but a survival cost bleeds you each round, so a bad estimate eventually means
      <b>elimination</b>. The referee knows the true value, so we can see exactly who shared, who lied, and
      who survived.</p>
  </div>
</section>

<section class="sec">
  <p class="sec-eyebrow">What happened</p>
  <h2 class="sec-h">Incentives failed. Structure worked.</h2>
  <div class="steps">
    <div class="step"><div class="step-n">1</div><div>
      <h4>Left alone, they don't cooperate</h4>
      <p>When solo play works, the agents just solve it alone. In one match, <b>0 of 196</b> reasoning steps
        even mentioned the other agent.</p></div></div>
    <div class="step"><div class="step-n">2</div><div>
      <h4>Rewarding accuracy backfired</h4>
      <p>We tried to fix it with payoffs — bigger rewards for good estimates, and we told each agent exactly
        how close it had to get. But <b>accuracy isn't cooperation</b>: handed a clear target, each agent just
        measured harder <b>on its own</b> to hit it instead of pooling with the other. Measuring costs credits,
        so they burned through their budgets and <b>died even faster</b>.</p></div></div>
    <div class="step"><div class="step-n">3</div><div>
      <h4>So we made cooperation the only way to win</h4>
      <p>Each agent's instrument gets a hidden <b>offset</b> it can't remove — measuring again just repeats
        it. Only <b>averaging both agents' readings</b> cancels the offsets and recovers the truth:</p>
      <div class="mechanic">
        <div class="mech-truth">Hidden truth <b>θ = 480</b></div>
        <div class="mech-row"><span class="tg a">You read</span><span class="num">720</span>your instrument runs high</div>
        <div class="mech-row"><span class="tg b">Partner reads</span><span class="num">240</span>theirs runs low</div>
        <div class="mech-row avg"><span class="tg ok">Average</span><span class="num">480</span>the offsets cancel — the truth, recovered together</div>
      </div></div></div>
    <div class="step"><div class="step-n">4</div><div>
      <h4>Cooperation switched on</h4>
      <p>They immediately began modelling each other, messaging, and pooling — the exact behavior that never
        appeared before.</p></div></div>
  </div>
</section>

<section class="sec">
  <p class="sec-eyebrow">What we found</p>
  <h2 class="sec-h">The more they need each other, the more they give back</h2>
  <div class="prose"><p>We turned interdependence into a dial and measured <b>reciprocity</b> — whether the
    exchange is mutual or one-sided. Across three clean settings it climbs from free-riding to near-perfect
    give-and-take:</p></div>
  <div class="stat-row" title="Reciprocity index — how mutual the exchange is: 1 = both agents share equally, ~0 = one gives while the other only takes.">
    <div class="stat soft"><div class="k">Soft wall · solo survives</div><div class="v">0.28</div>
      <div class="d">one-sided — one gives, the other just takes</div></div>
    <div class="stat med"><div class="k">Medium wall</div><div class="v">0.45</div>
      <div class="d">the exchange starts to balance out</div></div>
    <div class="stat hard"><div class="k">Hard wall · solo is fatal</div><div class="v">0.97</div>
      <div class="d">near-perfect, mutual give-and-take</div></div>
  </div>
  {% if gradient_charts %}
  <p class="note" style="margin-top:22px"><b>The full sweep</b> — dialing the offset 0 → 500. Hover any point
    or caption for detail:</p>
  {{ gradient_charts|safe }}
  <p class="note"><b>Read the trend, not the points:</b> each point is a single match, so the middle is noisy
    (a multi-seed run is underway). The far right shows a real limit — push the wall too hard and agents die
    before they can establish cooperation. Full report &amp; table on the
    <a class="cta" href="/gradient" style="font-size:inherit">gradient page</a>.</p>
  {% endif %}
</section>

<section class="sec">
  <p class="sec-eyebrow">What it means</p>
  <div class="meaning">
    <p>Language-model agents don't cooperate just because it would help. <b>They cooperate when they must</b> —
      and mutual give-and-take, not one-sided free-riding, scales with how much they need each other.</p>
    <p class="sub">Design implication: to get cooperative multi-agent AI, build <b>interdependence into the
      task itself</b>. Tuning incentives isn't enough.</p>
  </div>
</section>

<section class="sec">
  <p class="sec-eyebrow">See for yourself</p>
  <h2 class="sec-h">Explore the runs</h2>
  <div class="prose"><p>Every run is Qwen-3-32B against itself. Open one to watch each agent reason, measure,
    message, and trade — tick by tick — then who survived. The two ends of the dial:</p></div>
  <div class="feat">
    <a class="fcard hard" href="/game/sample-qwen3-32b-5games"><div class="ft">Hard wall</div>
      <h4>Solo is impossible → they reciprocate</h4>
      <p>Both agents realize they need the other's reading, message back and forth, and pool to recover the
        true value.</p></a>
    <a class="fcard soft" href="/game/sample-qwen3-32b-5games-soft"><div class="ft">Soft wall</div>
      <h4>Solo half-works → it stays one-sided</h4>
      <p>One agent shares generously; the other takes and rarely gives back. Cooperation is optional, so it
        frays.</p></a>
  </div>
  <div class="explore-links">
    <a class="cta" href="/compare">⇄ Compare every run side by side →</a>
    <a class="cta" href="/gradient">📈 The full dose–response →</a>
  </div>
</section>

<div class="gallery-head">
  <h2>All runs</h2>
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


COMPARE = _SHELL.replace("{{ inner|safe }}", """
<a class="back" href="/">← all games</a>
<p class="sub" style="font-size:13px;margin:10px 0 14px">Every finished game, side by side.
Use it to contrast setups — e.g. the <b>hard wall</b> (solo is fatal) vs the <b>soft wall</b>
(solo survives but cooperation pays), and whether the agents actually reciprocate.</p>
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
