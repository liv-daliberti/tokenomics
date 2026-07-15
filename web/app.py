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
from agora.policies import REGISTRY
from agora.referee import run_match
from agora.transcripts import Transcript
from analysis.metrics import load_events, summary as metric_summary
from analysis.viz import _CSS, render_body, render_comparison, render_simple

RUNS = os.environ.get("AGORA_RUNS",
                      os.path.join(os.path.dirname(__file__), "..", "runs", "web"))
os.makedirs(RUNS, exist_ok=True)

app = Flask(__name__)
JOBS: Dict[str, dict] = {}          # id -> {status, error} for in-flight runs
CANCELLED = set()                   # deleted jobs whose workers must not publish results
LOCK = threading.Lock()

# API keys for hosted endpoints (Azure/OpenAI) live in memory ONLY — they are
# never written into the job metadata JSON (which is on disk and re-served).
# Blank-field reuse is scoped to the exact (provider, base_url) the key was
# entered for — otherwise a second visitor could aim base_url at their own
# server and receive the stored key as a Bearer header — and is enabled only
# on a private loopback server (not on Render or a 0.0.0.0 bind, where other
# clients share the process; AGORA_ALLOW_KEY_REUSE=1 overrides).
_JOB_KEYS: Dict[str, str] = {}        # job_id -> key for that run
_LAST_KEYS: Dict[tuple, str] = {}     # (provider, base_url) -> last key entered
_ALLOW_KEY_REUSE = bool(os.environ.get("AGORA_ALLOW_KEY_REUSE")) or (
    not os.environ.get("RENDER")
    and os.environ.get("HOST", "127.0.0.1") in ("127.0.0.1", "localhost", "::1"))

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


def _parse_keys(raw: str) -> Dict[str, str]:
    """Parse the API-key field: one bare key ({'*': key}, applied to every
    cloud endpoint) or space/comma-separated host=key pairs for a mixed-model
    run ({host: key}). Pairs win only if EVERY token parses as host=key with a
    dotted host, so a bare key that happens to contain '=' (Azure keys can end
    in '=') is never misread as pairs."""
    raw = (raw or "").strip()
    if not raw:
        return {}
    toks = [t for t in re.split(r"[,\s;]+", raw) if t]
    pairs = {}
    for t in toks:
        h, sep, k = t.partition("=")
        if sep and h and k and ("." in h or h == "localhost"):
            pairs[h] = k
    if pairs and len(pairs) == len(toks):
        return pairs
    return {"*": raw}


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
    if form.get("memory", "").strip() in ("context", "markdown"):
        ov["memory"] = form.get("memory").strip()
    # tri-state selects: "" = preset, "1"/"0" = force on/off
    for k in ("values_via_trade_only", "require_paid_trades"):
        if form.get(k, "").strip() in ("0", "1"):
            ov[k] = form.get(k).strip() == "1"
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


def _running_progress(job_id: str, meta: dict) -> tuple[list, dict]:
    """Load the append-only transcript and summarize an in-flight match.

    Transcript writes are flushed one complete JSON object at a time, so this
    lets the running page show work immediately instead of waiting for the
    match-level metrics pass at the end.  Be deliberately forgiving here: a
    status refresh should still render if it races the very first write.
    """
    path = os.path.join(RUNS, f"{job_id}.jsonl")
    events = []
    if os.path.exists(path):
        try:
            events = load_events(path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            events = []

    starts = [e for e in events if e.get("event") == "game_start"]
    finished_games = sum(e.get("event") == "game_end" for e in events)
    finished_rounds = sum(e.get("event") == "round_end" for e in events)
    current_game = starts[-1].get("game_index", len(starts) - 1) + 1 if starts else 0
    current_round = next((e.get("round") for e in reversed(events)
                          if e.get("round") is not None), None)
    progress = {
        "games_done": finished_games,
        "current_game": current_game,
        "games_total": int(meta.get("games", meta.get("n_games", 1))),
        "rounds_done": finished_rounds,
        "current_round": current_round,
        "events": len(events),
    }
    return events, progress


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

# The de-confounding comparison charts (prompted vs neutral/no-hint) reuse the same
# chart engine; loaded the same way so /gradient and the home can embed them.
_dspec = _ilu.spec_from_file_location("deconfound_report",
                                      os.path.join(_REPO, "scripts", "deconfound_report.py"))
_deconf = _ilu.module_from_spec(_dspec)
_dspec.loader.exec_module(_deconf)

# Per-match progress for the GPT-5.4 replication browser (/gpt54) comes from
# the same logic the CLI progress bars use.
_pspec = _ilu.spec_from_file_location("gpt54_progress",
                                      os.path.join(_REPO, "scripts", "gpt54_progress.py"))
_progress = _ilu.module_from_spec(_pspec)
_pspec.loader.exec_module(_progress)

# Where GPT-5.4 transcripts live: the live runs dir first (this machine),
# then committed samples (so the deployed site can show curated matches).
_GPT54_DIRS = [os.path.join(_REPO, "runs", "gpt54"),
               os.path.join(_REPO, "docs", "samples", "gpt54")]
_SAFE_RUN = re.compile(r"^[A-Za-z0-9._-]+$")


def _gpt54_files() -> dict:
    """{match name: transcript path}, live runs shadowing committed samples."""
    out: dict = {}
    for base in _GPT54_DIRS:
        if not os.path.isdir(base):
            continue
        for f in sorted(os.listdir(base)):
            if f.endswith(".jsonl"):
                out.setdefault(f[:-6], os.path.join(base, f))
    return out


def _gpt54_manifest() -> dict:
    """The driver's manifest (status + token usage per match), if present."""
    for base in _GPT54_DIRS:
        p = os.path.join(base, "manifest.json")
        if os.path.exists(p):
            try:
                return json.load(open(p))
            except (OSError, ValueError):
                pass
    return {}


def _load_events_partial(path: str) -> list:
    """Load a transcript that may be mid-write: skip any torn trailing line."""
    events = []
    try:
        with open(path) as fh:
            for line in fh:
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return events


_STATS_CACHE: Dict[tuple, dict] = {}   # (path, mtime) -> behavioural stats


def _match_stats(path: str) -> dict:
    """Cheap per-match behaviour stats for the /gpt54 dashboard, cached by
    mtime so a page refresh doesn't re-read transcripts that haven't moved."""
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return {}
    if key in _STATS_CACHE:
        return _STATS_CACHE[key]
    ev = _load_events_partial(path)
    offers = [e for e in ev if e.get("event") == "propose_trade"]
    prices = sorted(e.get("price", 0) for e in offers)
    st = {
        "msgs": sum(1 for e in ev if e.get("event") == "message"),
        "censored": sum(1 for e in ev if e.get("event") == "message"
                        and "[#]" in (e.get("text") or "")),
        "offers": len(offers),
        "settled": sum(1 for e in ev if e.get("event") == "respond_trade"
                       and e.get("status") == "accepted"),
        "median_price": prices[len(prices) // 2] if prices else None,
        "alive": next((len(e.get("alive", [])) for e in reversed(ev)
                       if e.get("event") == "round_start"), None),
        "notes": sum(1 for e in ev if e.get("event") == "notes"),
    }
    if len(_STATS_CACHE) > 600:            # crude bound; entries are tiny
        _STATS_CACHE.clear()
    _STATS_CACHE[key] = st
    return st

# One browsable match PER POINT on the sweep dial (not separate hand-run matches):
# for each offset, the first seed in which at least one agent survives (else the
# first complete seed), so every point on the dose-response curve is openable.
# Order here = order shown in the gallery (see `seed_samples`).
_SAMPLE_RUNS = [
    ("docs/samples/sweep_off000.jsonl", "Qwen3-32B ×2 — offset σ=0 (no wall: solo is viable)"),
    ("docs/samples/sweep_off050.jsonl", "Qwen3-32B ×2 — offset σ=50 (soft wall)"),
    ("docs/samples/sweep_off100.jsonl", "Qwen3-32B ×2 — offset σ=100 (soft wall)"),
    ("docs/samples/sweep_off150.jsonl", "Qwen3-32B ×2 — offset σ=150 (mid wall)"),
    ("docs/samples/sweep_off200.jsonl", "Qwen3-32B ×2 — offset σ=200 (mid wall)"),
    ("docs/samples/sweep_off250.jsonl", "Qwen3-32B ×2 — offset σ=250 (mid-hard wall)"),
    ("docs/samples/sweep_off300.jsonl", "Qwen3-32B ×2 — offset σ=300 (hard wall: solo often fatal)"),
    ("docs/samples/sweep_off350.jsonl", "Qwen3-32B ×2 — offset σ=350 (hard wall)"),
    ("docs/samples/sweep_off400.jsonl", "Qwen3-32B ×2 — offset σ=400 (very hard wall)"),
    ("docs/samples/sweep_off500.jsonl", "Qwen3-32B ×2 — offset σ=500 (hardest: solo hopeless)"),
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

    # (2)+(3) add any current sample we have not added and the user has not deleted.
    # `created` is shifted by the sample's position so the gallery (sorted newest
    # first) preserves _SAMPLE_RUNS order — git checkouts flatten file mtimes, so
    # we can't rely on them for ordering.
    n_samples = len(current)
    for idx, (jid, (rel, title)) in enumerate(current.items()):
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
                "created": os.path.getmtime(src) + (n_samples - idx), "backend": "llm", "preset": "(sample)",
                "n_agents": len(cfg.get("agent_ids", [])), "tau": cfg.get("tau"),
                "framing": cfg.get("framing"), "n_games": s.get("n_games"),
                "rounds": len([e for e in ev if e["event"] == "round_end"]),
                "n_rounds": cfg.get("n_rounds"),  # rounds PER game (the told horizon)
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
    # Pop the keys up front so EVERY exit path (scripted backend, error before
    # the LLM branch) evicts them from memory, and so the except block below
    # can redact them from any error text that gets persisted.
    keys = _JOB_KEYS.pop(job_id, None) or {}
    bare = keys.get("*")                                   # one key for all endpoints
    pairs = {h: k for h, k in keys.items() if h != "*"} or None   # per-host keys
    # Without a visitor-supplied key, do NOT let the backend fall back to the
    # server's own env keys: base_url is form-controlled, so lending env keys
    # would let a visitor bounce them to an arbitrary host. A private operator
    # can opt in with AGORA_ALLOW_ENV_KEYS=1.
    if bare is None and not os.environ.get("AGORA_ALLOW_ENV_KEYS"):
        bare = "EMPTY"
    try:
        with LOCK:
            if job_id in CANCELLED:
                return
        cfg = build_config(params)
        ids = cfg.agent_ids
        n_games = int(params.get("games", 1))
        meta.update(n_agents=len(ids), tau=cfg.tau, framing=cfg.framing,
                    survival_cost=cfg.survival_cost, n_games=n_games,
                    horizon=("known %d-round" % cfg.n_rounds) if cfg.reveal_horizon
                             else "hidden (γ=%.2f)" % cfg.gamma)

        if params["backend"] == "llm":
            from agora.run import build_policies
            # A model mix ("m1@url1, m2@url2#provider", cycled over seats) pits
            # different models against each other in the identical game; blank
            # mix = every seat on the single endpoint from the form.
            spec = (params.get("models") or "").strip() or "llm"
            policies = build_policies(cfg, spec, params["model"], params["base_url"],
                                      n_games=n_games, api_key=bare,
                                      provider=params.get("provider") or None,
                                      api_keys=pairs)
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
            n_rounds=cfg.n_rounds,  # rounds PER game (the told horizon)
            deception_rate=s["deception"]["deception_rate"],
            cooperation=s["cooperation"]["cooperation_index"],
            survivors=s["survivors"], n_agents=s["n_agents"],
            gini=s["gini_final_credits"], welfare=s["welfare"],
            parse_fail_rate=s["diagnostics"]["parse_fail_rate"],
        )
        with LOCK:
            if job_id in CANCELLED:
                return
            _write_meta(meta)
            JOBS[job_id] = {"status": "done", "error": None}
    except Exception as exc:  # surface any failure (e.g. no LLM endpoint) to the UI
        # meta is persisted to disk and rendered on the error page, so scrub any
        # key material from provider error bodies (some echo a masked key, and a
        # user-pointed proxy could echo the whole Authorization header).
        err = f"{type(exc).__name__}: {exc}"
        for k in list(keys.values()) + ([bare] if bare else []):
            if k and k != "EMPTY":
                err = err.replace(k, "***")
        err = re.sub(r"\bsk-[A-Za-z0-9_\-*.]{4,}", "sk-***", err)
        meta.update(status="error", error=err)
        with LOCK:
            if job_id in CANCELLED:
                return
            _write_meta(meta)
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
    """(charts HTML, chart CSS, source label). Prefers the multi-seed aggregate
    (mean ± CI over every finished seed) over the single-seed points; ('','','')
    if there is no gradient data."""
    try:
        for base in ("docs/samples/gradient", "runs/qwen"):
            bdir = os.path.join(_REPO, base)
            rows, label = _gradient.load_rows(bdir)
            if rows:
                return (_gradient.charts_block(rows, _gradient._load_anchors(bdir)),
                        _gradient.CHART_CSS, label)
    except Exception:
        pass
    return "", "", ""


def _deconf_charts() -> str:
    """De-confounding comparison charts (cooperation + survival, prompted vs
    neutral/no-hint) for the home page. Reads the COMMITTED aggregates so it works
    on the deployed site without the raw transcripts; '' if the data isn't there."""
    try:
        base = os.path.join(_REPO, "docs", "samples", "gradient")
        dpath = os.path.join(base, "deconf_aggregate.json")
        if not os.path.exists(dpath):
            return ""
        conf = json.load(open(os.path.join(base, "gradient_aggregate.json")))["rows"]
        dec = json.load(open(dpath))["rows"]
        apath = os.path.join(base, "gradient_anchors.json")
        anc = json.load(open(apath))["specs"] if os.path.exists(apath) else {}
        cc = _deconf.comparison_charts(conf, dec, anc)
        # Survival first: it is the robust, tight-CI result; cooperation (noisier) second.
        return (f'<div class="grad"><div class="card hero">{cc["surv"]}</div>'
                f'<div class="card hero">{cc["coop"]}</div></div>')
    except Exception:
        return ""


def _page_ctx(page: str) -> dict:
    """Shared template context (presets, policy names, preset-fill data, the
    shared study CSS, and the nav's active tab) for the Home / Create views.
    The dose-response charts now live on /baseline, not here."""
    return dict(css=_CSS, page=page, presets=sorted(PRESETS),
                policies_list=sorted(REGISTRY), default_policies=DEFAULT_POLICIES,
                preset_data=json.dumps(_preset_data()),
                study_css=_STUDY_CSS, nav=("create" if page == "create" else "home"))


@app.route("/")
def index():
    """Home: the GPT-5.4 study (one clean scroll) + the run gallery."""
    return render_template_string(INDEX, games=_all_games(), **_page_ctx("games"))


@app.route("/create")
def create():
    """The Run-new-game view: the configuration form."""
    return render_template_string(INDEX, games=[], **_page_ctx("create"))


@app.route("/baseline")
def baseline():
    """The completed Qwen3-32B baseline study, in full — the narrative and the
    dose-response / de-confounding charts that used to live on the home page."""
    charts, chart_css, _ = _gradient_charts()
    return render_template_string(
        BASELINE, css=_CSS, study_css=_STUDY_CSS, nav="baseline",
        chart_css=chart_css, gradient_charts=charts, deconf_charts=_deconf_charts())


@app.route("/delete_all", methods=["POST"])
def delete_all():
    """Clear the whole gallery (does not resurrect on restart — see seed marker)."""
    with LOCK:
        CANCELLED.update(JOBS)
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
        "provider": f.get("provider", "vllm"),   # 'vllm' | 'openai' (Azure etc.)
        "policies": (f.get("policies") or "").strip() or DEFAULT_POLICIES,
        "model": f.get("model", "qwen3-32b"),
        "base_url": f.get("base_url", "http://localhost:8000/v1"),
        "models": (f.get("models") or "").strip(),   # optional per-seat mix

        "title": f.get("title", "").strip(),
        "overrides": parse_overrides(f),
    }
    params["overrides"].setdefault("framing", "cooperative")  # cooperative by default
    if params["preset"] not in PRESETS:
        abort(400, "unknown preset")
    if params["backend"] not in ("llm", "scripted"):
        abort(400, "unknown backend")
    if params["provider"] not in ("vllm", "openai"):
        abort(400, "unknown provider")
    mix_models = []
    if params["models"]:
        toks = [t.strip() for t in params["models"].split(",") if t.strip()]
        if any("@" not in t for t in toks):
            abort(400, "model mix must be comma-separated model@base_url[#provider] tokens")
        mix_models = [t.partition("@")[0].strip() for t in toks]
    job_id = uuid.uuid4().hex[:10]
    # Keys go into the in-memory store only — never into params/meta, which
    # are persisted to disk — and only for LLM runs, whose _run_job pops them
    # on every path. The field holds one key, or host=key pairs for a mix. A
    # blank field reuses the last keys entered for this exact endpoint/mix
    # (so you type them once per server session).
    keys = _parse_keys(f.get("api_key")) if params["backend"] == "llm" else {}
    key_slot = (params["provider"], params["models"] or params["base_url"])
    if keys and _ALLOW_KEY_REUSE:
        _LAST_KEYS[key_slot] = keys
    elif not keys and _ALLOW_KEY_REUSE:
        keys = _LAST_KEYS.get(key_slot, {})
    if keys and params["backend"] == "llm":
        _JOB_KEYS[job_id] = keys
    if not params["title"]:
        gtag = f"{params['games']} games · " if params["games"] > 1 else ""
        if params["backend"] != "llm":
            mtag = params["backend"]
        elif mix_models:
            mtag = " vs ".join(dict.fromkeys(mix_models))
        else:
            mtag = params["model"]
        params["title"] = f"{params['preset']} · {mtag} · {gtag}seed {params['seed']}"
    _start_job(job_id, params)
    return redirect(url_for("game", job_id=job_id))


@app.route("/game/<job_id>")
def game(job_id: str):
    """Render a finished game (simple or detailed view), or a running/error placeholder."""
    if not os.path.exists(_meta_path(job_id)):
        # On the hosted free tier the runs dir is ephemeral: a redeploy or idle
        # restart clears user-run games (samples re-seed). Say so instead of a
        # bare 404 that reads like a broken link.
        return render_template_string(GONE, css=_CSS, job_id=job_id), 404
    meta = _load_meta(job_id)
    status = JOBS.get(job_id, {}).get("status", meta.get("status", "done"))

    if status == "running":
        events, progress = _running_progress(job_id, meta)
        body = render_simple(events, meta.get("title", "Agora game")) if events else ""
        return render_template_string(WAIT, css=_CSS, meta=meta, body=body,
                                      progress=progress)
    if status == "error":
        return render_template_string(ERROR, css=_CSS, meta=meta)

    view = request.args.get("view", "simple")
    transcript_path = os.path.join(RUNS, f"{job_id}.jsonl")
    if not os.path.exists(transcript_path):
        return render_template_string(GONE, css=_CSS, job_id=job_id), 404
    try:
        events = load_events(transcript_path)
    except (OSError, ValueError, KeyError) as exc:
        broken = dict(meta, error=f"The saved transcript is unreadable: {type(exc).__name__}: {exc}")
        return render_template_string(ERROR, css=_CSS, meta=broken), 500
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


_FIG_LOCK = threading.Lock()
# figures that regenerate from live run data on request (throttled); others are
# served as the committed static file.
_LIVE_FIGS = {"cost_error.png": "plot_cost_error.py",
              "err_lines.png": "plot_error_views.py",
              "err_heatmap.png": "plot_error_views.py",
              "err_scatter.png": "plot_error_views.py"}


def _maybe_regen(name: str, path: str) -> None:
    """Regenerate a live figure from current runs, at most every 30s."""
    script = _LIVE_FIGS.get(name)
    if not script:
        return
    import sys
    import subprocess
    with _FIG_LOCK:
        age = time.time() - os.path.getmtime(path) if os.path.exists(path) else 1e9
        if age <= 30:
            return
        try:
            subprocess.run([sys.executable, os.path.join(_REPO, "scripts", script),
                            "--out", os.path.join(_REPO, "paper", "fig",
                                                  name.replace(".png", ".pdf"))],
                           cwd=_REPO, timeout=90, capture_output=True)
        except Exception:      # a plot failure must not break the page
            pass


@app.route("/fig/<name>")
def figure(name: str):
    """Serve a paper figure (PNG). Live figures regenerate from current run
    data (throttled); the rest are served as the committed static file."""
    if not _SAFE_RUN.match(name) or not name.endswith(".png"):
        abort(404)
    path = os.path.join(_REPO, "paper", "fig", name)
    _maybe_regen(name, path)
    if not os.path.exists(path):
        abort(404)
    from flask import send_file
    resp = send_file(path, mimetype="image/png")
    resp.headers["Cache-Control"] = "no-cache"      # so a reload shows the latest
    return resp


@app.route("/views")
def views():
    """Three candidate presentations of the difficulty x deception x model error
    result, side by side, live-updating — pick one."""
    return render_template_string(VIEWS, css=_CSS)


@app.route("/economics")
def economics():
    """The interactive cost–utility explorer: the raw price tradeoff (measure
    more vs buy the partner's reading) under the REAL reward rule, with sliders
    for the knobs so you can watch the variables trade off. Pure client-side
    math — exact closed forms, no simulation, no external libraries."""
    c = PRESETS["cooperative"]
    defaults = {
        "tau": c.tau, "prior_sigma": c.prior_sigma, "measure_cost": c.measure_cost,
        "reward_max": c.reward_max, "bucket": c.bucket(), "rtc": c.reward_to_credits,
        "survival": c.survival_cost, "bias_sigma": c.bias_sigma,
        "starting_credits": c.starting_credits,
    }
    return render_template_string(ECON, css=_CSS, defaults=json.dumps(defaults))


@app.route("/gpt54")
def gpt54_index():
    """The GPT-5.4 results dashboard: live aggregate stats (tokens, cache
    rate, market behaviour, survival) plus every match with its condition,
    progress, and a link to the full per-match report."""
    files = _gpt54_files()
    manifest = _gpt54_manifest()
    rows = []
    tot = {"in": 0, "cached": 0, "out": 0, "offers": 0, "settled": 0,
           "censored": 0, "finished": 0, "alive": 0, "seats": 0}
    prices = []
    for name, path in files.items():
        done, exp, ended = _progress.match_progress(path)
        usage = (manifest.get(name, {}) or {}).get("usage") or {}
        st = _match_stats(path)
        market = "open" if name.endswith("_open") else "paid"
        rows.append({
            "name": name, "pct": round(100 * done / exp) if exp else 0,
            "ended": ended, "rounds": f"{done}/{exp}", "market": market,
            "tokens": (f"{usage.get('prompt_tokens', 0)/1e6:.1f}M in"
                       if usage.get("prompt_tokens") else "—"),
            "trades": f"{st.get('settled', 0)}/{st.get('offers', 0)}",
            "censored": st.get("censored", 0),
            "mtime": os.path.getmtime(path),
        })
        tot["in"] += usage.get("prompt_tokens", 0)
        tot["cached"] += usage.get("cached_tokens", 0)
        tot["out"] += usage.get("completion_tokens", 0)
        tot["offers"] += st.get("offers", 0)
        tot["settled"] += st.get("settled", 0)
        tot["censored"] += st.get("censored", 0)
        if st.get("median_price") is not None:
            prices.append(st["median_price"])
        if ended:
            tot["finished"] += 1
            if st.get("alive") is not None:
                tot["alive"] += st["alive"]
                tot["seats"] += 2
    rows.sort(key=lambda r: (r["ended"], -r["mtime"]))   # running first, newest next
    tiles = {
        "matches": f"{tot['finished']}/{len(rows)}",
        "tokens": f"{tot['in']/1e6:,.0f}M",
        "cache": (f"{100*tot['cached']/tot['in']:.0f}%" if tot["cached"] else "—"),
        "trades": f"{tot['settled']}/{tot['offers']}",
        "price": (f"{sorted(prices)[len(prices)//2]:g} cr" if prices else "—"),
        "censored": tot["censored"],
        "survival": (f"{100*tot['alive']/tot['seats']:.0f}%" if tot["seats"] else "—"),
    }
    return render_template_string(GPT54_LIST, css=_CSS, rows=rows, tiles=tiles,
                                  any_running=any(not r["ended"] for r in rows))


@app.route("/gpt54/<name>")
def gpt54_game(name: str):
    """Render one GPT-5.4 match as the standard game report; a match still in
    flight renders its partial transcript and auto-refreshes."""
    if not _SAFE_RUN.match(name):
        abort(404)
    path = _gpt54_files().get(name)
    if path is None:
        abort(404)
    events = _load_events_partial(path)
    if not any(e.get("event") == "game_start" for e in events):
        return render_template_string(GPT54_WAIT, css=_CSS, name=name)
    done, exp, ended = _progress.match_progress(path)
    try:
        body = render_simple(events, f"GPT-5.4 · {name}")
    except Exception:                     # a partial tail mid-write must never 500
        return render_template_string(GPT54_WAIT, css=_CSS, name=name)
    return render_template_string(GPT54_GAME, css=_CSS, body=body, name=name,
                                  ended=ended, pct=round(100 * done / exp) if exp else 0)


@app.route("/gradient")
def gradient():
    """The interdependence dose-response report: offset (bias σ) vs cooperation,
    reciprocity, survival, and messaging, as small-multiple charts."""
    rows, label = [], ""
    anchors = None
    for base in ("docs/samples/gradient", "runs/qwen"):  # committed data, then local runs
        bdir = os.path.join(_REPO, base)
        rows, label = _gradient.load_rows(bdir)
        if rows:
            anchors = _gradient._load_anchors(bdir)
            break
    if not rows:
        return render_template_string(
            COMPARE, css=_CSS,
            body='<h1>Interdependence gradient</h1><p class="sub">No gradient runs yet — '
                 'run the offset sweep (<code>scripts/agora_qwen_gradient.slurm</code>) to populate it.</p>')
    nav = ('<a href="/" style="position:fixed;top:12px;left:14px;z-index:99;'
           'font:13px/1 system-ui,sans-serif;color:#3987e5;text-decoration:none;'
           'background:rgba(20,22,30,.72);padding:7px 11px;border-radius:9px">← all games</a>')
    return nav + _gradient.render(rows, label, anchors)


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
    with LOCK:
        CANCELLED.add(job_id)
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

# One nav bar on every narrative/tool page, so the site reads as a single
# thing instead of a stack of bolted-on pages. `nav` marks the active tab.
_NAV = """<nav class="topnav">
  <a class="brand" href="/">Agora</a>
  <a href="/" class="{{ 'on' if nav=='home' else '' }}">GPT-5.4 study</a>
  <a href="/gpt54" class="{{ 'on' if nav=='dash' else '' }}">Live dashboard</a>
  <a href="/baseline" class="{{ 'on' if nav=='baseline' else '' }}">Qwen baseline</a>
  <a href="/gradient" class="{{ 'on' if nav=='gradient' else '' }}">Dose–response</a>
  <a href="/economics" class="{{ 'on' if nav=='econ' else '' }}">Cost–utility</a>
  <span class="spacer"></span>
  <a class="run {{ 'on' if nav=='create' else '' }}" href="/create">＋ Run a game</a>
</nav>"""

# Shared "study" CSS — used by the home (GPT-5.4) and the /baseline (Qwen)
# pages so both speak one visual language. Chart CSS is added per-page.
_STUDY_CSS = """
  :root{--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;}
  .topnav{display:flex;gap:2px;align-items:center;flex-wrap:wrap;margin:0 0 22px;padding:0 0 10px;border-bottom:1px solid var(--line);}
  .topnav a{padding:8px 11px;text-decoration:none;color:var(--mut);font-size:13.5px;font-weight:600;border-radius:8px;white-space:nowrap;}
  .topnav a:hover{color:var(--fg);background:var(--card);}
  .topnav a.on{color:var(--fg);}
  .topnav a.brand{color:var(--fg);font-weight:800;letter-spacing:-.02em;font-size:16px;margin-right:6px;padding-left:0;}
  .topnav a.run{color:var(--blue);}
  .topnav .spacer{flex:1;}
  .toolbar{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:14px 0 0;}
  .toolbar a{display:block;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:13px 15px;text-decoration:none;color:var(--fg);font-weight:600;font-size:14px;transition:border-color .12s;}
  .toolbar a:hover{border-color:var(--blue);}
  .toolbar a span{display:block;color:var(--mut);font-size:12px;font-weight:400;margin-top:3px;}
  .bl-link{display:flex;justify-content:space-between;align-items:center;gap:14px;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px;margin:18px 0 0;text-decoration:none;color:var(--fg);transition:border-color .12s;}
  .bl-link:hover{border-color:var(--blue);} .bl-link .sub2{color:var(--mut);font-size:13px;margin-top:3px;font-weight:400;}
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
  .conditions{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:18px 0 4px;}
  @media(max-width:640px){.conditions{grid-template-columns:1fr;}}
  .cond{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;position:relative;overflow:hidden;}
  .cond::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;}
  .cond.prompted::before{background:var(--red);} .cond.neutral::before{background:var(--blue);}
  .cond .ct{display:flex;align-items:center;gap:8px;font:700 12px/1 var(--mono);letter-spacing:.03em;text-transform:uppercase;margin:0 0 8px;}
  .cond .ct .sw{width:17px;height:0;border-top:3px solid;border-radius:2px;flex:none;}
  .cond.prompted .ct{color:var(--red);} .cond.prompted .ct .sw{border-color:var(--red);}
  .cond.neutral .ct{color:var(--blue);} .cond.neutral .ct .sw{border-color:var(--blue);border-top-width:2px;border-top-style:dashed;}
  .cond p{font-size:13.5px;color:var(--mut);line-height:1.5;margin:0;} .cond b{color:var(--fg);font-weight:600;}
  .held{font-size:13.5px;color:var(--mut);line-height:1.6;max-width:66ch;margin:12px 0 2px;} .held b{color:var(--fg);}
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
"""

# ---- HOME (/) — the GPT-5.4 study, one clean scroll ----
INDEX = _SHELL.replace("{{ inner|safe }}", _NAV + """
{% if page != 'create' %}
<style>{{ study_css|safe }}</style>

<header class="hero">
  <p class="eyebrow">Agora · a multi-agent LLM study</p>
  <h1 class="lead">Make AI agents <em>buy</em> information instead of saying it — and a market appears.</h1>
  <p class="dek">We're running the study on <em>GPT-5.4</em> under a stricter rule — a <em>paid market</em>:
    numbers are censored from chat and every trade must cost more than zero, so a reading can only change
    hands as a <em>paid</em> trade. The early pilot is already a sharp, controlled contrast. Under the old
    open rules GPT-5.4 <em>never trades</em> — it just tells its partner the number. Close that channel and
    the market comes alive: <em>264 of 266</em> offers settle — every one priced at the <em>floor</em> the
    rules allow. And with trading now the only way to share a value, we can finally ask the question that
    matters: when a model can <em>tell</em> its partner is lying, does it stop buying? <em>It doesn't</em> —
    the headline finding below.</p>
</header>

<section class="sec" style="margin-top:38px">
  <p class="sec-eyebrow">The study now · GPT-5.4</p>
  <h2 class="sec-h">Same game, a paid market — and a different social animal.</h2>
  <div class="prose">
    <p>The baseline below ran on Qwen3-32B. We're now rerunning the whole thing on <b>GPT-5.4</b> with one
      deliberate change to the rules. In the baseline an agent could simply <b>say</b> a reading in chat, so
      the trading market barely mattered. Now the market is the <b>only</b> channel for a value: numbers
      (digits <i>and</i> words) are <b>censored from messages</b>, and every trade must be priced <b>above
      zero</b>. If agents want to pool, they have to buy and sell — and we can watch the price they put on
      information. The full offset sweep is running live; the <b>pilot</b> below is an early, small-n look.</p>
  </div>
  <div class="stat-row">
    <div class="stat hard"><div class="k">trades settled · paid market</div><div class="v">264/266</div>
      <div class="d">Under the old open rules: <b>0 offers</b> in every match — GPT-5.4 just tells its
        partner the number in chat.</div></div>
    <div class="stat med"><div class="k">price of a reading</div><div class="v">0.01–0.1<span style="font-size:16px"> cr</span></div>
      <div class="d">The floor <b>binds</b>: forced to charge <i>something</i>, agents settle a symmetric
        swap at the smallest price they can name — a penny, not the reading's ~6-credit value.</div></div>
    <div class="stat hard"><div class="k">chat leaks blocked</div><div class="v">84</div>
      <div class="d">Messages where an agent tried to speak a number and the censor replaced it with
        <code>[#]</code> — then it fell back to trading.</div></div>
    <div class="stat med"><div class="k">markdown memory cost</div><div class="v">~4× less</div>
      <div class="d">Journaling each round + context reset used <b>8M</b> input tokens (82% cached) vs
        <b>30M</b> for one growing conversation — same 2/2 survival and full cooperation.</div></div>
  </div>
  <p class="note"><b>Pilot, read with care:</b> these are the first GPT-5.4 matches (n is small, one seed per
    cell), enough to show the market <i>mechanism</i> turns on but not yet a dose–response. The full sweep is
    running now. <a class="cta" href="/gpt54" style="font-size:inherit">Watch it live on the GPT-5.4
    dashboard →</a></p>
</section>

<section class="sec">
  <p class="sec-eyebrow">The headline finding · the knowing–doing gap</p>
  <h2 class="sec-h">They can tell it's a lie — and buy it anyway.</h2>
  <div class="prose">
    <p>Here is what the paid market lets us do that nothing else can. We sit an agent across from a
      <b>scripted liar</b> that fabricates every reading it sells, and — because the referee saw what the
      seller actually measured — we label each offer honest or fake <b>mechanically, with no judge model</b>.
      Then we measure two things: does the model <b>flag</b> the fabrication when asked, and does it
      <b>decline to buy</b> it in the game? The two come apart, badly.</p>
  </div>
  <div class="stat-row">
    <div class="stat soft"><div class="k">GPT-5.4 · buys the liar's fakes</div><div class="v">95%</div>
      <div class="d">…that, as a judge, it flags as fabricated with probability <b>0.81–0.88</b>. It knows,
        and pays anyway — judging <i>its own</i> games.</div></div>
    <div class="stat soft"><div class="k">Qwen3-32B · buys the liar's fakes</div><div class="v">100%</div>
      <div class="d">Same picture, worse: every fabricated reading bought, though a judge flags them at
        <b>0.82–0.85</b>.</div></div>
    <div class="stat med"><div class="k">the gap holds under pressure</div><div class="v">~0.8</div>
      <div class="d">buys − (1 − flags). It doesn't shrink at the hard wall, where the agent <i>needs</i> the
        reading to survive — it buys the lie just as readily.</div></div>
  </div>
  <p class="note">The label is <b>judge-free</b> (the referee's own record of what the seller saw), so this
    isn't one model grading another — which is what makes it a benchmark. <b>Pilot, 2 seeds/cell:</b>
    GPT-5.4's numbers are a true single-model gap (it judges its own play); Qwen's use GPT-5.4 as an oracle
    detector for now. And a characterised caveat: at the hard wall an honest partner's readings are genuinely
    ~110 off the truth (its instrument offset), which confuses the <i>judge prompt</i> — the ground-truth
    label is unaffected.</p>

  <div class="prose" style="margin-top:24px">
    <p><b>Does making the lie expensive fix it?</b> Partly — and the way it fails is the sharpest part of the
      story. We raise the price of every trade and watch what GPT-5.4 pays for. It prices <b>honest</b>
      information about right: it buys a real reading (worth ~6 credits) almost always up to a price of 8, then
      refuses once it's clearly overpriced. But it never prices the <b>fabrication</b>: it buys the liar's fake
      at every cheap price, and even at <b>32 credits</b> — five times what a real reading is worth, for
      something worth nothing — it still buys it <b>41%</b> of the time, <i>more often than it buys the
      fairly-priced honest reading at that same price</i>. The model can value real information; it just won't
      discount a value for being a lie. The gap is, underneath, a <b>pricing failure</b>.</p>
  </div>
  <img src="/fig/price_sweep.png"
       alt="GPT-5.4 buy-rate vs price: the honest reading is refused when overpriced, but the fabrication stays bought"
       style="width:100%;max-width:580px;display:block;margin:16px auto 0;border-radius:12px;background:#fff;padding:10px">

  <div class="prose" style="margin-top:24px">
    <p><b>The whole design space at once.</b> Below: the LLM's estimation <b>error</b> (how wrong its final
      answer is) against the <b>price of information</b>, for every combination of <b>difficulty</b> (blue =
      easy, red = the hard wall) and <b>model</b> (● GPT-5.4, ▲ Qwen3-32B), split by whether the partner
      <b>always lies</b>, <b>sometimes lies</b>, or <b>never lies</b>. Dashed lines are the rational optimum.
      This figure is <b>rendering live as the runs land</b> — reload to watch it fill in.</p>
  </div>
  <img src="/fig/cost_error.png"
       alt="Estimation error vs price of information, by difficulty and model, for lying/mixed/honest partners"
       style="width:100%;max-width:900px;display:block;margin:14px auto 0;border-radius:12px;background:#fff;padding:10px">
</section>

<section class="sec">
  <p class="sec-eyebrow">What's next · running now</p>
  <h2 class="sec-h">The full offset sweep, live on GPT-5.4.</h2>
  <div class="prose">
    <p>The pilot shows the paid market <i>turns on</i>; the full run measures how it behaves as we dial up how
      badly the agents need each other — the same offset sweep from <b>0</b> (solo is fine) to <b>500</b> (solo
      is hopeless) that the baseline used, now under the paid-market rules. Every match streams to the live
      dashboard as it finishes, with its trades, censored messages, tokens, and survival.</p>
  </div>
  <div class="toolbar">
    <a href="/gpt54">Live dashboard<span>every GPT-5.4 match, updating in real time</span></a>
    <a href="/economics">Cost–utility explorer<span>the price tradeoff, from the reward rule</span></a>
    <a href="/create">Run a game<span>drive the market yourself, any model</span></a>
    <a href="/compare">Compare runs<span>side by side across the dial</span></a>
  </div>
</section>

<a class="bl-link" href="/baseline">
  <div><b>The Qwen3-32B baseline study, in full →</b>
    <div class="sub2">Where this started: cooperation is instructed not discovered, honest but blind to a
      liar — the offset dial, the two controls, and the dose–response charts.</div></div>
  <span class="cta">Read the baseline →</span>
</a>

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
      <span class="m">{{ g.n_agents }} agents · {% if g.n_games and g.n_games > 1 %}{{ g.n_games }} games × {{ g.n_rounds or (g.rounds // g.n_games) }} rounds{% else %}{{ g.n_rounds or g.rounds }} rounds{% endif %}{% if g.tau is defined %} · τ={{ g.tau }}{% endif %}</span>
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
  <label>Agents — model endpoint</label>
  <select name="provider" id="provider">
    <option value="vllm" selected>local vLLM — Qwen-3-32B (scripts/serve_qwen.sh)</option>
    <option value="openai">cloud — Azure / OpenAI-compatible (e.g. gpt-5.4)</option>
  </select>
  <div class="row">
    <div><label>Model (served / deployment name)</label><input name="model" id="model" value="qwen3-32b"></div>
    <div><label>Base URL</label><input name="base_url" id="base_url" value="http://localhost:8000/v1"></div>
  </div>
  <div id="key-row" class="hide">
    <label>API key — kept in memory only, never written to disk</label>
    <input name="api_key" id="api_key" type="password" autocomplete="off"
           placeholder="one key, or host=key pairs for a model mix; blank = reuse this session's keys">
  </div>
  <label>Model mix — optional: different models play each other</label>
  <input name="models" id="models" autocomplete="off"
         placeholder="e.g. gpt-5.4@https://liv.services.ai.azure.com/openai/v1, qwen3-32b@http://localhost:8000/v1">
  <p class="m" style="color:var(--mut);font-size:12px;margin:6px 0 0">
    Comma-separated <code>model@base_url[#provider]</code>, cycled over the seats — leave blank to run
    every seat on the single endpoint above. Each seat gets the identical game; only the model differs.
    Cloud endpoints resolve keys per host: paste <code>host=key</code> pairs (space-separated) in the
    API-key field, e.g. <code>liv.services.ai.azure.com=KEY1 api.anthropic.com=KEY2</code>.</p>
  <p class="m" style="color:var(--mut);font-size:12px;margin:6px 0 0">
    Local: serve the model first (<code>scripts/serve_qwen.sh</code>). Cloud: paste your Azure AI Foundry
    key and use the resource's OpenAI-compatible URL, e.g.
    <code>https://liv.services.ai.azure.com/openai/v1</code> with deployment <code>gpt-5.4</code>.
    The match runs in the background either way.</p>

  <label>Games in a row — played back-to-back</label>
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
    <div><label>Memory across games (LLM agents)</label>
      <select name="memory">
        <option value="context" selected>full context — one growing conversation</option>
        <option value="markdown">markdown notes — journal each round, context reset per game</option>
      </select></div>
  </div>
  <div class="row">
    <div><label>Value exchange</label>
      <select name="values_via_trade_only">
        <option value="">preset</option>
        <option value="1">trade-only — numbers censored from chat; values move only via trades</option>
        <option value="0">open — numbers allowed in messages</option>
      </select></div>
    <div><label>Paid trades</label>
      <select name="require_paid_trades">
        <option value="">preset</option>
        <option value="1">required — price must be &gt; 0 (any amount, never free)</option>
        <option value="0">off — free offers (price 0) allowed</option>
      </select></div>
  </div>
  <p class="m" style="color:var(--mut);font-size:12px;margin:6px 0 0">
    To make <b>paid trading the only way to share a value</b>: set Value exchange to
    <b>trade-only</b> (chat censors digits <i>and</i> spelled-out numbers) and Paid trades to
    <b>required</b> (the market rejects price-0 offers — 0.25 credits is fine, free is not).</p>
  <p class="m" style="color:var(--mut);font-size:12px;margin:6px 0 0">
    <b>Markdown notes</b>: after every round each agent writes what happened to its own notebook;
    between games its conversation is cleared and only the notebook comes back. The notebooks
    appear on the game page (📝 per round, plus a full per-agent document at the bottom).</p>
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
const PROVIDER_DEFAULTS = {
  vllm:   {model: 'qwen3-32b', base_url: 'http://localhost:8000/v1'},
  openai: {model: 'gpt-5.4',   base_url: 'https://liv.services.ai.azure.com/openai/v1'}
};
function syncKeyRow(){
  var sel = document.getElementById('provider');
  var k = document.getElementById('key-row');
  var mix = document.getElementById('models');
  // the key field matters for the cloud provider OR whenever a model mix is
  // set (a mix may include cloud endpoints regardless of the provider select)
  var need = (sel && sel.value !== 'vllm') || (mix && mix.value.trim() !== '');
  if(k) k.classList.toggle('hide', !need);
}
function fillProvider(){
  var sel = document.getElementById('provider'); if(!sel) return;
  var d = PROVIDER_DEFAULTS[sel.value]; if(!d) return;
  document.getElementById('model').value = d.model;
  document.getElementById('base_url').value = d.base_url;
  // a key typed for the other provider must not ride along hidden in the POST
  document.getElementById('api_key').value = '';
  syncKeyRow();
}
document.addEventListener('DOMContentLoaded', function(){
  var sel = document.querySelector('select[name=preset]');
  if(sel){ sel.addEventListener('change', fillPreset); fillPreset(); }
  var pv = document.getElementById('provider');
  // sync visibility on load too (browser back / autofill can restore the
  // select without firing change) — but don't overwrite restored field values
  if(pv){ pv.addEventListener('change', fillProvider); syncKeyRow(); }
  var mix = document.getElementById('models');
  if(mix){ mix.addEventListener('input', syncKeyRow); }
});
</script>
{% endif %}
""")

# ---- BASELINE (/baseline) — the completed Qwen3-32B study, moved off the home ----
BASELINE = _SHELL.replace("{{ inner|safe }}", _NAV + """
<style>{{ chart_css|safe }}{{ study_css|safe }}</style>

<header class="hero">
  <p class="eyebrow">Agora · the baseline study</p>
  <h1 class="lead">Two AI agents cooperate — only because we told them to.</h1>
  <p class="dek">The completed <em>Qwen3-32B</em> study that set up the current GPT-5.4 run. Two identical
    agents must pool their work to survive, and we dialed up how badly they need each other. They cooperate
    — but <em>only because the prompt tells them to, and how</em>. Strip that away and the cooperation
    <em>vanishes</em>: they'd sooner die alone than work out that pooling saves them. And sat across from a
    partner that lied every single round, they <em>never learned to stop trusting it</em>.</p>
</header>

<section class="sec">
  <p class="sec-eyebrow">Why we did this</p>
  <h2 class="sec-h">Multi-agent AI is coming. Does it actually collaborate?</h2>
  <div class="prose">
    <p>Agents that negotiate, delegate, and split work are arriving fast — but a basic question is unanswered:
      when they can help each other, do language-model agents share <b>fairly</b>, or does one carry the other
      while it takes? We built a small, controlled world where cooperation is <b>measurable</b> and deception
      is <b>verifiable</b>, and watched two copies of the same model play it out.</p>
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
  <p class="sec-eyebrow">How we tested it</p>
  <h2 class="sec-h">Make cooperation necessary — then dial exactly how much.</h2>
  <div class="steps">
    <div class="step"><div class="step-n">1</div><div>
      <h4>Two things we can measure directly</h4>
      <p>When agents <i>can</i> help each other, do they <b>cooperate at all</b> — share readings, talk — or
        just ignore each other? And when they do, is the give-and-take <b>mutual</b> or <b>one-sided</b> (one
        gives, the other only takes)? A referee that knows the true value lets us score both. Then we vary how
        badly they need each other and watch what moves.</p></div></div>
    <div class="step"><div class="step-n">2</div><div>
      <h4>One dial: how badly they need each other</h4>
      <p>Each agent's instrument gets a hidden <b>offset</b> it can't remove — measuring again just repeats
        it. Only <b>averaging both agents' readings</b> cancels it and recovers the truth. We turn that offset
        from <b>0</b> (solo is fine) up to <b>500</b> (solo is hopeless):</p>
      <div class="mechanic">
        <div class="mech-truth">Hidden truth <b>θ = 480</b></div>
        <div class="mech-row"><span class="tg a">You read</span><span class="num">720</span>your instrument runs high</div>
        <div class="mech-row"><span class="tg b">Partner reads</span><span class="num">240</span>theirs runs low</div>
        <div class="mech-row avg"><span class="tg ok">Average</span><span class="num">480</span>the offsets cancel — the truth, recovered together</div>
      </div></div></div>
    <div class="step"><div class="step-n">3</div><div>
      <h4>~140 matches across the dial — then two controls</h4>
      <p>We run the same game at <b>ten settings</b> of that offset — plus <b>four finer steps</b> between 0 and
        50 to pin down exactly where cooperation turns on — <b>ten seeds</b> each, ~140 Qwen-vs-Qwen matches in
        all. Each match is <b>10 games × 5 rounds</b> in one growing conversation, so the agents keep the
        <b>full context</b> of everything before. Then <b>two controls</b> to ask whether the behaviour is real
        or prompted: the whole sweep <b>rerun with neutral wording and the how-to hint removed</b>, and a
        separate <b>probe</b> pitting one agent against a bot whose honesty we control.</p></div></div>
  </div>
</section>

<section class="sec">
  <p class="sec-eyebrow">What we found</p>
  <h2 class="sec-h">They cooperate only when told — and can't spot a liar.</h2>
  <div class="prose">
    <p>We dialed the wall from <b>0</b> (solo is fine) to <b>500</b> (solo is hopeless), <b>ten seeds</b> at each
      setting. In the charts below the <b>solid line</b> is the original run; each <b>dashed line</b> is the
      <i>same game</i> with the prompt's help taken away — and, on survival, two scripted strategy ceilings in
      the identical game: the best possible all-cooperate pair and the best possible all-solo pair. Error bars
      are 95% CIs; the whole story is the gap between the lines. <b>Survival is on top because it is the robust result</b> — tight intervals, a clean trend.
      Cooperation (below it) points the same way but is noisier, so read its <i>shape</i>, not any single point.</p>
  </div>
  <div class="conditions">
    <div class="cond prompted">
      <div class="ct"><span class="sw"></span>Prompted run</div>
      <p>Cooperative framing — <b>"you and your partner are a team"</b> — plus the hint:
        <b>"average your two readings to cancel the error."</b></p>
    </div>
    <div class="cond neutral">
      <div class="ct"><span class="sw"></span>Neutral run · the control</div>
      <p>No team framing, <b>no hint</b> — the agents must work out pooling themselves. Everything else is
        identical.</p>
    </div>
  </div>
  <p class="held"><b>Held fixed in both:</b> the task, measurement noise, budget, survival cost, horizon and
    prior — only the wording changes, and the offset dial runs 0→500 in each. The two faint dashed lines on the
    survival chart are <b>scripted, non-LLM</b> reference agents in the same game, each the <b>ceiling of its
    strategy</b>: the best possible always-pooling pair, and the best possible never-sharing solo pair.
    <b>What we did not do:</b> script the LLMs' choices, change the task, or tell the neutral agents the
    trick.</p>
  {% if deconf_charts %}{{ deconf_charts|safe }}{% elif gradient_charts %}{{ gradient_charts|safe }}{% endif %}
  <div class="prose" style="margin-top:22px">
    <p><b>Cooperation is instructed, not discovered.</b> With the cooperative prompt, sharing jumps to the norm
      at the <b>very first notch</b> of the wall and stays there — it looks like two agents realising they can't
      do it alone. But that's the prompt talking. Strip the "team" framing and the "average your readings" hint
      (the <b style="color:var(--blue)">blue</b> points), and cooperation has <b>no response to the wall at
      all</b> — sharing sits near <b>15%</b> whether the wall is off (14% at offset 0) or lethal. And without
      pooling they <b>die</b>: neutral survival lands exactly on the <b>ceiling for solo play</b>
      (<b>63%</b> → <b>24%</b> → <b>3%</b> as the wall hardens, n=10 each) — solo played as well as solo can
      be played, and no better — never near the <b>ceiling for cooperation</b>. The pooling was the
      instruction, not the agents working out that they need each other.</p>
    <p><b>Honest — but blind to a liar.</b> They barely cheat: with the referee checking every sold value
      against what the seller actually knew, just <b>1 value in 1,360</b> genuinely contradicted what the seller
      knew — faced with an unverifiable channel they route around it (<b>~85%</b> of matches settle zero trades)
      rather than exploit it. But the other half of social reasoning fails. Against a bot that <b>fabricates ~9
      of every 10</b> readings it sells, truth revealed each round, Qwen <b>never stops buying</b>: matched over
      the first six games (5 seeds), it accepts the liar <b>99%</b> of the time versus an honest partner's
      <b>57%</b>, flat across all ten of the liar's games. In one match the true value was <b>205</b> and the
      liar sold a fabricated <b>905</b> — <b>4× too high</b> — and Qwen paid a credit and bought it.</p>
  </div>
  <p class="note">Full per-offset breakdown — cooperation, survival, reciprocity, fabrication, and the scripted
    baselines — on the <a class="cta" href="/gradient" style="font-size:inherit">gradient page</a>.
    {% if not deconf_charts and not gradient_charts %}<br>The sweep is running — the charts appear here as runs finish.{% endif %}</p>
</section>

<section class="sec">
  <p class="sec-eyebrow">What it means</p>
  <div class="meaning">
    <p>Put together, it's one picture: the social behaviour this game rewards is <b>not something these agents
      bring on their own</b>. They cooperate when the prompt tells them to and how — take the instruction away
      and they'd sooner <b>die alone</b> than work out that pooling saves them. And they don't do the other half
      of social reasoning either: they <b>never stop trusting a partner that lies</b> to them every round, even
      with the truth handed to them each time.</p>
    <p>The one-line version: <b>cooperation between these LLM agents has to be instructed, its absence is fatal,
      and they don't learn to distrust a proven liar.</b></p>
    <p class="sub">This is the baseline the current GPT-5.4 run builds on — same game, but now the market is the
      only channel for a value, and every trade must cost. <a class="cta" href="/">See the GPT-5.4 study →</a></p>
  </div>
</section>

<section class="sec">
  <p class="sec-eyebrow">See for yourself</p>
  <h2 class="sec-h">Explore the baseline runs</h2>
  <div class="prose"><p>Every run is Qwen-3-32B against itself — a single match at one point on the dial. Open
    one to watch each agent reason, measure, message, and trade tick by tick, then who survived. The two sides
    of the switch:</p></div>
  <div class="feat">
    <a class="fcard soft" href="/game/sample-sweep-off000"><div class="ft">offset σ = 0 · coin flip</div>
      <h4>No wall — cooperation is optional</h4>
      <p>Either agent can hit the target alone, so cooperating is a coin flip — about half the runs they barely
        talk (like this one), half they pool anyway. Everyone survives regardless.</p></a>
    <a class="fcard hard" href="/game/sample-sweep-off050"><div class="ft">offset σ = 50 · the norm</div>
      <h4>A wall appears — cooperation kicks in</h4>
      <p>Now going solo is penalized. Within the first game or two the agents start messaging and pooling
        readings — the same model, one notch of the dial later.</p></a>
  </div>
  <div class="toolbar" style="margin-top:16px">
    <a href="/gradient">The full dose–response<span>every offset, all four metrics</span></a>
    <a href="/compare">Compare runs<span>side by side across the dial</span></a>
    <a href="/economics">Cost–utility explorer<span>the price tradeoff</span></a>
  </div>
</section>
""")

WAIT = _SHELL.replace("{{ inner|safe }}", """
<meta http-equiv="refresh" content="2">
<a class="back" href="/">← all games</a>
{% if not body %}<h1>{{ meta.title }}</h1>{% endif %}
<div class="panel" style="margin-bottom:16px"><p class="sub" style="margin:0">
⏳ Running the agents… showing live progress and refreshing every 2 seconds.
{% if progress.current_game %} Game {{ progress.current_game }} of {{ progress.games_total }}
{% if progress.current_round is not none %} · round {{ progress.current_round }}{% endif %}
· {{ progress.rounds_done }} round(s) finished · {{ progress.events }} events logged.
{% else %} The agents are being initialized.{% endif %}
{% if meta.backend == 'llm' %} Model turns are sequential, so pauses between updates are normal.{% endif %}
</p></div>
{{ body|safe }}
""")

ERROR = _SHELL.replace("{{ inner|safe }}", """
<a class="back" href="/">← all games</a>
<h1>{{ meta.title }}</h1>
<div class="panel"><p class="sub" style="margin:0 0 8px">This game failed to run:</p>
<pre style="color:var(--red);white-space:pre-wrap;margin:0">{{ meta.error }}</pre>
{% if meta.backend == 'llm' %}<p class="m" style="color:var(--mut);margin-top:12px">
{% if meta.provider == 'openai' %}Check the endpoint URL <code>{{ meta.base_url }}</code>, the deployment
name <code>{{ meta.model }}</code>, and the API key (keys live in memory only — re-enter it on the
Run-new-game form after a server restart).{% else %}Is the vLLM server up at
<code>{{ meta.base_url }}</code>? Start it with <code>scripts/serve_qwen.sh</code>,
and install the client with <code>pip install openai</code>.{% endif %}</p>{% endif %}</div>
""")

GPT54_LIST = _SHELL.replace("{{ inner|safe }}", """
{% if any_running %}<meta http-equiv="refresh" content="30">{% endif %}
<style>
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0 18px;}
@media(max-width:720px){.tiles{grid-template-columns:1fr 1fr;}}
.tile{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:11px 13px;}
.tile .k{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.5px;}
.tile .v{font-size:23px;font-weight:700;font-variant-numeric:tabular-nums;margin-top:2px;}
.tile .d{color:var(--mut);font-size:11px;margin-top:1px;}
.mkt{font-size:10px;padding:2px 7px;border-radius:20px;}
.mkt.paid{background:#1c3a24;color:var(--green);} .mkt.open{background:#20242e;color:var(--mut);}
</style>
<a class="back" href="/">← all games</a>
<h1 style="margin:10px 0 4px">GPT-5.4 · the replication, live</h1>
<p class="sub" style="margin:0 0 4px">The Qwen study rerun on <b>gpt-5.4</b> under stricter rules —
the <b>paid market</b>: numbers are censored from chat and every trade must cost more than 0, so
values can only move through the escrow. Running matches update live (auto-refresh 30s). The Qwen
results stay on the <a href="/">home page</a> as the baseline.</p>
<div class="tiles">
  <div class="tile"><div class="k">matches finished</div><div class="v">{{ tiles.matches }}</div>
    <div class="d">pilot now; full program next</div></div>
  <div class="tile"><div class="k">input tokens · cached</div><div class="v">{{ tiles.tokens }} · {{ tiles.cache }}</div>
    <div class="d">cached input bills at a deep discount</div></div>
  <div class="tile"><div class="k">trades settled / offered</div><div class="v">{{ tiles.trades }}</div>
    <div class="d">open-market pilot had ZERO offers</div></div>
  <div class="tile"><div class="k">median trade price</div><div class="v">{{ tiles.price }}</div>
    <div class="d">the price floor binds: info trades at ε</div></div>
  <div class="tile"><div class="k">censored messages</div><div class="v">{{ tiles.censored }}</div>
    <div class="d">number-leak attempts blocked → [#]</div></div>
  <div class="tile"><div class="k">survival (finished)</div><div class="v">{{ tiles.survival }}</div>
    <div class="d">agents alive at match end</div></div>
</div>
{% if not rows %}<p class="sub">No GPT-5.4 matches yet — start the driver:
<code>python scripts/gpt54_program.py --stage pilot</code></p>{% endif %}
<ul class="games">
{% for r in rows %}
  <li>
    <a href="/gpt54/{{ r.name }}">{{ r.name }}</a>
    <span class="mkt {{ r.market }}">{{ r.market }}</span>
    <span class="m">{{ r.rounds }} rounds · {{ r.tokens }} · trades {{ r.trades }}{% if r.censored %} · {{ r.censored }} censored{% endif %}</span>
    {% if r.ended %}<span class="pill ok">done</span>
    {% else %}<span class="pill">running · {{ r.pct }}%</span>{% endif %}
  </li>
{% endfor %}
</ul>
""")

GPT54_WAIT = _SHELL.replace("{{ inner|safe }}", """
<meta http-equiv="refresh" content="15">
<a class="back" href="/gpt54">← all GPT-5.4 runs</a>
<h1>{{ name }}</h1>
<div class="panel"><p class="sub" style="margin:0">⏳ This match is just starting — first events
haven't landed yet. The page refreshes automatically.</p></div>
""")

GPT54_GAME = _SHELL.replace("{{ inner|safe }}", """
{% if not ended %}<meta http-equiv="refresh" content="30">{% endif %}
<div style="display:flex;justify-content:space-between;align-items:center">
  <a class="back" href="/gpt54">← all GPT-5.4 runs</a>
  {% if not ended %}<span class="pill">running · {{ pct }}% — refreshes every 30s</span>
  {% else %}<span class="pill ok">complete</span>{% endif %}
</div>
{{ body|safe }}
""")

VIEWS = _SHELL.replace("{{ inner|safe }}", """
<meta http-equiv="refresh" content="45">
<style>
  .vwrap{max-width:1000px;}
  .vfig{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin:18px 0;}
  .vfig h3{margin:0 0 2px;font-size:17px;} .vfig .d{color:var(--mut);font-size:13.5px;margin:0 0 12px;line-height:1.5;}
  .vfig img{width:100%;display:block;border-radius:10px;background:#fff;padding:10px;}
  .vfig .rec{display:inline-block;font:600 11px/1 ui-monospace,monospace;color:var(--green);
    border:1px solid var(--green);border-radius:20px;padding:3px 8px;margin-left:8px;vertical-align:middle;}
</style>
<a class="back" href="/" style="color:var(--blue);text-decoration:none;font-size:14px">← home</a>
<h1 style="margin:10px 0 2px">Pick a presentation</h1>
<p class="sub" style="margin:0 0 6px">Three ways to tell the same result — estimation error by
difficulty, partner honesty, and model. Filling in live as the grid runs (page refreshes every 45s).
Tell me the letter you want and I'll make it the paper/site figure.</p>

<div class="vfig">
  <h3>A · Deception → error lines <span class="rec">my pick</span></h3>
  <p class="d">x = how much the partner lies, y = error, one line per model, easy vs hard panel.
    Price dropped (it barely moves error). Reads left-to-right as a story: the more the partner lies,
    the worse you do — and the GPT/Qwen gap is the whole point.</p>
  <img src="/fig/err_lines.png" alt="deception to error lines">
</div>
<div class="vfig">
  <h3>B · Heatmap</h3>
  <p class="d">rows = model × difficulty, columns = partner honesty, colour = error (darker = more wrong).
    Whole story at a glance; the difficulty×deception interaction is the top-left-light to
    bottom-right-dark gradient.</p>
  <img src="/fig/err_heatmap.png" alt="error heatmap">
</div>
<div class="vfig">
  <h3>C · Price scatter (cleaned)</h3>
  <p class="d">the original idea, tidied: three partner panels, price on x, error on y, seeds joined per
    difficulty (dark = hard, light = easy). Keeps price on-axis even though it barely moves error.</p>
  <img src="/fig/err_scatter.png" alt="price scatter">
</div>
""")

GONE = _SHELL.replace("{{ inner|safe }}", """
<a class="back" href="/">← all games</a>
<h1>This game is gone</h1>
<div class="panel"><p class="sub" style="margin:0 0 8px">
No game <code>{{ job_id }}</code> exists on this server anymore. Games run here are stored in
<b>temporary storage</b>: when the server redeploys or wakes from an idle sleep (it runs on a free
tier), user-run games are cleared. The curated sample runs are restored automatically.</p>
<p class="sub" style="margin:0"><a href="/create">▶ Run a new game</a> · <a href="/">browse the samples</a></p></div>
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
<p class="sub" style="font-size:13px;margin:10px 0 14px">Every finished game, side by side, sorted by
interdependence (offset σ). Use it to contrast points on the dial — e.g. <b>σ=0</b> (solo is viable) vs
<b>σ=300</b> (solo is often fatal) — and see how noisy reciprocity is match to match.</p>
{{ body|safe }}
""")


def _mark_interrupted() -> None:
    """Games run on daemon threads inside this process, so any meta still
    'running' at process start was killed by a restart/redeploy — flag it so
    its page stops waiting forever."""
    for f in os.listdir(RUNS):
        if not f.endswith(".json"):
            continue
        try:
            meta = _load_meta(f[:-5])
        except (OSError, ValueError):
            continue
        if meta.get("status") == "running":
            meta.update(status="error",
                        error="The server restarted while this game was running, "
                              "so the run was interrupted. Start it again.")
            _write_meta(meta)


ECON = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agora · cost–utility explorer</title><style>{{ css|safe }}
.wrap{max-width:980px;}
.lede{color:var(--mut);max-width:70ch;line-height:1.6;margin:6px 0 18px;}
.lede b{color:var(--fg);}
.controls{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;background:var(--card);
  border:1px solid var(--line);border-radius:14px;padding:14px 18px;margin:0 0 16px;}
@media(max-width:760px){.controls{grid-template-columns:1fr 1fr;}}
.ctl label{display:block;color:var(--mut);font-size:11px;text-transform:uppercase;
  letter-spacing:.5px;margin:0 0 4px;}
.ctl output{float:right;color:var(--fg);font-weight:600;font-variant-numeric:tabular-nums;}
.ctl input[type=range]{width:100%;accent-color:#5aa9e6;}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:0 0 16px;}
@media(max-width:760px){.stats{grid-template-columns:1fr 1fr;}}
.tile{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;}
.tile .k{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.5px;}
.tile .v{font-size:26px;font-weight:700;font-variant-numeric:tabular-nums;margin-top:3px;}
.tile .d{color:var(--mut);font-size:11.5px;margin-top:2px;line-height:1.4;}
.chart-card{background:var(--card);border:1px solid var(--line);border-radius:14px;
  padding:16px 18px;margin:0 0 16px;}
.chart-card h3{margin:0;font-size:15.5px;}
.chart-card .sub2{color:var(--mut);font-size:12.5px;margin:2px 0 10px;}
.lg{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--mut);margin:0 0 6px;}
.lg .sw{display:inline-block;width:16px;height:0;border-top:3px solid;border-radius:2px;
  vertical-align:middle;margin-right:5px;}
.lg .sw.dash{border-top-style:dashed;border-top-width:2px;}
svg text{font:11px system-ui,sans-serif;fill:var(--mut);}
svg .dl{font-weight:600;font-size:11.5px;}
.tt{position:fixed;pointer-events:none;background:#0c0e13;border:1px solid var(--line);
  border-radius:8px;padding:7px 10px;font-size:12px;color:var(--fg);display:none;z-index:50;
  box-shadow:0 4px 14px rgba(0,0,0,.4);}
.tt .r{display:flex;gap:8px;justify-content:space-between;}
.tt .r span:first-child{color:var(--mut);}
details.tbl{margin-top:8px;} details.tbl summary{cursor:pointer;color:var(--mut);font-size:12px;}
details.tbl table{margin-top:6px;} details.tbl td,details.tbl th{font-size:12px;padding:3px 10px;}
.note2{color:var(--mut);font-size:12.5px;line-height:1.55;max-width:74ch;}
.note2 b{color:var(--fg);}
</style></head><body><div class="wrap">
<a href="/" style="color:#5aa9e6;text-decoration:none;font-size:14px">← all games</a>
<h1 style="margin:10px 0 2px">Cost–utility explorer</h1>
<p class="lede">The raw price tradeoff, computed exactly from the game's reward rule
(<b>reward = max(0, R − ⌊error/bucket⌋)</b>, in credits). Two strategies each round:
<b style="color:#c98200">measure only</b> (your instrument's offset never averages out) vs
<b style="color:#1fa768">measure, then buy your partner's reading</b> at the asking price
(the paired offsets cancel exactly). Drag the sliders and watch the variables trade.</p>

<div class="controls">
  <div class="ctl"><label>Instrument offset σ<sub>b</sub> <output id="o-sb"></output></label>
    <input type="range" id="s-sb" min="0" max="500" step="25"></div>
  <div class="ctl"><label>Measurement noise τ <output id="o-tau"></output></label>
    <input type="range" id="s-tau" min="5" max="200" step="5"></div>
  <div class="ctl"><label>Measure cost (credits) <output id="o-c"></output></label>
    <input type="range" id="s-c" min="0" max="5" step="0.25"></div>
  <div class="ctl"><label>Asking price (credits) <output id="o-p"></output></label>
    <input type="range" id="s-p" min="0" max="10" step="0.25"></div>
</div>

<div class="stats">
  <div class="tile"><div class="k">partner's reading is worth</div><div class="v" id="t-wtp">–</div>
    <div class="d">max price a rational buyer pays (Δ expected reward, in credits)</div></div>
  <div class="tile"><div class="k">best solo net / round</div><div class="v" id="t-solo">–</div>
    <div class="d">expected reward − measuring − survival, at the best k</div></div>
  <div class="tile"><div class="k">best trade net / round</div><div class="v" id="t-trade">–</div>
    <div class="d">same, buying one partner reading at the asking price</div></div>
  <div class="tile"><div class="k">verdict at this price</div><div class="v" id="t-verdict">–</div>
    <div class="d" id="t-verdict-d">trade minus solo, best k each</div></div>
</div>

<div class="chart-card">
  <h3>Net utility per round vs measurements taken</h3>
  <p class="sub2">Expected reward (credits) minus everything spent. k = 0 means submitting
    the prior mean (solo) or relying on the bought reading alone (trade).</p>
  <div class="lg"><span><span class="sw" style="border-color:#c98200"></span>measure only</span>
    <span><span class="sw" style="border-color:#1fa768"></span>measure + buy at asking price</span></div>
  <svg id="ch1" viewBox="0 0 900 300" role="img" aria-label="Net utility versus measurements"></svg>
  <details class="tbl"><summary>table view</summary><div id="tb1"></div></details>
</div>

<div class="chart-card">
  <h3>What is the partner's reading worth as the offset grows?</h3>
  <p class="sub2">Willingness to pay (credits) at each offset: each strategy's best expected
    reward net of measuring, subtracted. Where the curve is above the dashed asking price,
    trading is rational at that price.</p>
  <div class="lg"><span><span class="sw" style="border-color:#1fa768"></span>value of the partner's reading</span>
    <span><span class="sw dash" style="border-color:#9aa4b2"></span>asking price</span>
    <span><span class="sw dash" style="border-color:#5f6a7a"></span>measure cost</span></div>
  <svg id="ch2" viewBox="0 0 900 300" role="img" aria-label="Willingness to pay versus offset"></svg>
  <details class="tbl"><summary>table view</summary><div id="tb2"></div></details>
</div>

<p class="note2"><b>Fixed here:</b> <span id="rule-note"></span> Offsets are drawn to
<b>sum to zero</b> across the pair (each agent's effective offset is σ<sub>b</sub>/√2), so a plain
average cancels them exactly — that is the entire value of the trade. <b>Symmetric swap:</b> if both
agents buy from each other at the same price the payments cancel, and both walk away with the
accuracy gain — the price only redistributes credits; what it really prices is <b>access</b>.
Survival cost shifts both curves equally and never changes the verdict.</p>

<div class="tt" id="tt"></div>
<script>
const D = {{ defaults|safe }};
const SOLO = "#c98200", TRADE = "#1fa768", REF = "#9aa4b2", REF2 = "#5f6a7a";
const KMAX = 12;
function erf(x){ const s = x < 0 ? -1 : 1; x = Math.abs(x);
  const t = 1/(1+0.3275911*x);
  const y = 1-((((( 1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x);
  return s*y; }
const pLess = (t,s) => s <= 0 ? 1 : erf(t/(s*Math.SQRT2));
function expReward(s){ let e = 0;
  for (let m = 1; m <= D.reward_max; m++) e += pLess(m*D.bucket, s);
  return e; }
const sSolo = (k,sb,tau) => k === 0 ? D.prior_sigma : Math.sqrt(sb*sb/2 + tau*tau/k);
const sPool = (k,sb,tau) => k === 0 ? Math.sqrt(sb*sb/2 + tau*tau)
                                    : 0.5*Math.sqrt(tau*tau/k + tau*tau);
const P = { sb: D.bias_sigma, tau: D.tau, c: D.measure_cost, p: 1.0 };
const uSolo  = (k,sb) => D.rtc*expReward(sSolo(k,sb,P.tau)) - k*P.c - D.survival;
const uTrade = (k,sb) => D.rtc*expReward(sPool(k,sb,P.tau)) - k*P.c - P.p - D.survival;
// Willingness to pay = (best pooled expected reward net of measuring) minus
// (best solo ditto), each maximised over its OWN k — the largest price at
// which trading still weakly beats going alone. Independent of the current
// asking price, so the dashed price line is a fair comparison.
function wtpMax(sb){ let pool = -1e9, solo = -1e9;
  for (let k = 0; k <= KMAX; k++){
    pool = Math.max(pool, D.rtc*expReward(sPool(k,sb,P.tau)) - k*P.c);
    solo = Math.max(solo, D.rtc*expReward(sSolo(k,sb,P.tau)) - k*P.c);
  }
  return Math.max(0, pool - solo); }
const fmt = v => (Math.round(v*100)/100).toFixed(2).replace(/\\.?0+$/,"");

// ---- generic line chart into an <svg>, with crosshair hover + table ----
function draw(svgId, tbId, xs, series, xlabel, refs){
  const svg = document.getElementById(svgId);
  const W = 900, H = 300, L = 52, R = 120, T = 14, B = 34;
  let lo = 0, hi = -1e9;
  series.forEach(s => s.ys.forEach(v => { lo = Math.min(lo,v); hi = Math.max(hi,v); }));
  (refs||[]).forEach(r => { lo = Math.min(lo,r.y); hi = Math.max(hi,r.y); });
  if (hi <= lo) hi = lo + 1;
  const pad = (hi-lo)*0.08; lo -= pad; hi += pad;
  const X = i => L + (W-L-R) * (xs[i]-xs[0]) / (xs[xs.length-1]-xs[0]);
  const Y = v => T + (H-T-B) * (1 - (v-lo)/(hi-lo));
  let g = "";
  const steps = 5;
  for (let i = 0; i <= steps; i++){
    const v = lo + (hi-lo)*i/steps, y = Y(v);
    g += `<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" stroke="var(--line)" stroke-width="1"/>`
       + `<text x="${L-8}" y="${y+4}" text-anchor="end">${fmt(v)}</text>`;
  }
  if (lo < 0 && hi > 0)
    g += `<line x1="${L}" x2="${W-R}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--mut)" stroke-width="1" opacity=".55"/>`;
  const tick = Math.max(1, Math.round(xs.length/8));
  xs.forEach((x,i) => { if (i % tick === 0)
    g += `<text x="${X(i)}" y="${H-10}" text-anchor="middle">${x}</text>`; });
  g += `<text x="${W-R+8}" y="${H-10}">${xlabel}</text>`;
  (refs||[]).forEach(r => {
    g += `<line x1="${L}" x2="${W-R}" y1="${Y(r.y)}" y2="${Y(r.y)}" stroke="${r.color}"`
       + ` stroke-width="1.6" stroke-dasharray="5 4"/>`
       + `<text class="dl" x="${W-R+8}" y="${Y(r.y)+4}" fill="${r.color}">${r.name} ${fmt(r.y)}</text>`;
  });
  series.forEach(s => {
    const d = s.ys.map((v,i) => (i ? "L" : "M") + X(i).toFixed(1) + " " + Y(v).toFixed(1)).join(" ");
    g += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2.2" stroke-linejoin="round"/>`;
    const li = s.ys.length-1;
    g += `<text class="dl" x="${W-R+8}" y="${Y(s.ys[li])+4}" fill="${s.color}">${s.name}</text>`;
    g += s.ys.map((v,i) =>
      `<circle class="hv" data-i="${i}" cx="${X(i)}" cy="${Y(v)}" r="4.5" fill="${s.color}"`
      + ` stroke="var(--card)" stroke-width="2" opacity="0"/>`).join("");
  });
  g += `<line id="${svgId}-x" y1="${T}" y2="${H-B}" stroke="var(--mut)" stroke-width="1"`
     + ` opacity="0"/><rect x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}" fill="transparent"/>`;
  svg.innerHTML = g;

  const tt = document.getElementById("tt");
  svg.onmousemove = ev => {
    const pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
    const m = pt.matrixTransform(svg.getScreenCTM().inverse());
    let bi = 0, bd = 1e9;
    xs.forEach((x,i) => { const d = Math.abs(X(i)-m.x); if (d < bd){ bd = d; bi = i; } });
    const xl = document.getElementById(svgId + "-x");
    xl.setAttribute("x1", X(bi)); xl.setAttribute("x2", X(bi)); xl.setAttribute("opacity", ".5");
    svg.querySelectorAll(".hv").forEach(c =>
      c.setAttribute("opacity", +c.dataset.i === bi ? "1" : "0"));
    tt.style.display = "block";
    tt.style.left = Math.min(ev.clientX + 14, window.innerWidth - 190) + "px";
    tt.style.top = (ev.clientY + 12) + "px";
    tt.innerHTML = `<div class="r"><span>${xlabel}</span><b>${xs[bi]}</b></div>`
      + series.map(s => `<div class="r"><span style="color:${s.color}">${s.name}</span>`
        + `<b>${fmt(s.ys[bi])}</b></div>`).join("");
  };
  svg.onmouseleave = () => { tt.style.display = "none";
    document.getElementById(svgId + "-x").setAttribute("opacity", "0");
    svg.querySelectorAll(".hv").forEach(c => c.setAttribute("opacity", "0")); };

  let t = `<table><tr><th>${xlabel}</th>`
    + series.map(s => `<th>${s.name}</th>`).join("") + "</tr>";
  xs.forEach((x,i) => { t += `<tr><td>${x}</td>`
    + series.map(s => `<td>${fmt(s.ys[i])}</td>`).join("") + "</tr>"; });
  document.getElementById(tbId).innerHTML = t + "</table>";
}

function render(){
  ["sb","tau","c","p"].forEach(k => {
    document.getElementById("o-" + k).textContent = fmt(P[k]);
  });
  const ks = []; for (let k = 0; k <= KMAX; k++) ks.push(k);
  draw("ch1", "tb1", ks, [
    { name: "measure only", color: SOLO, ys: ks.map(k => uSolo(k, P.sb)) },
    { name: "measure + buy", color: TRADE, ys: ks.map(k => uTrade(k, P.sb)) },
  ], "readings k");
  const sbs = []; for (let s = 0; s <= 500; s += 20) sbs.push(s);
  draw("ch2", "tb2", sbs, [
    { name: "reading's value", color: TRADE, ys: sbs.map(wtpMax) },
  ], "offset σb", [
    { name: "asking price", y: P.p, color: REF },
    { name: "measure cost", y: P.c, color: REF2 },
  ]);
  let bs = -1e9, bt = -1e9;
  for (let k = 0; k <= KMAX; k++){ bs = Math.max(bs, uSolo(k, P.sb)); bt = Math.max(bt, uTrade(k, P.sb)); }
  const w = wtpMax(P.sb), dv = bt - bs;
  document.getElementById("t-wtp").textContent = fmt(w) + " cr";
  document.getElementById("t-solo").textContent = fmt(bs) + " cr";
  document.getElementById("t-trade").textContent = fmt(bt) + " cr";
  const vd = document.getElementById("t-verdict");
  vd.textContent = (dv >= 0 ? "+" : "") + fmt(dv) + " cr";
  vd.style.color = dv > 0.005 ? TRADE : (dv < -0.005 ? SOLO : "var(--mut)");
  document.getElementById("t-verdict-d").textContent = dv > 0.005
    ? "buying beats going solo at this price"
    : (dv < -0.005 ? "at this price, measuring alone is better" : "a wash at this price");
  document.getElementById("rule-note").textContent =
    "the cooperative preset's reward rule (reward = max(0, " + D.reward_max
    + " − ⌊error/" + fmt(D.bucket) + "⌋) credits), survival cost " + fmt(D.survival)
    + ", and a symmetric partner who has taken the same number of readings and sells one.";
}
[["s-sb","sb"],["s-tau","tau"],["s-c","c"],["s-p","p"]].forEach(([id,k]) => {
  const el = document.getElementById(id);
  el.value = P[k];
  el.addEventListener("input", () => { P[k] = parseFloat(el.value); render(); });
});
render();
</script></div></body></html>"""


# Populate the gallery at import so a fresh (ephemeral) deploy is never empty,
# and fail any runs a previous process left behind.
seed_samples()
_mark_interrupted()


def main() -> None:
    """Launch the dev server (seeding the sample gallery first)."""
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    print(f"Agora web UI on http://{host}:{port}")
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
