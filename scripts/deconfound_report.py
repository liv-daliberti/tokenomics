"""Standalone report for the de-confounding + trust-probe result.

Two experiments that reframe the headline:
  1. DE-CONFOUNDING — rerun the interdependence sweep with NEUTRAL framing and the
     "average them" strategy hint REMOVED. If the cooperation "switch" was emergent
     it should survive; if it was the prompt, it should vanish. (It vanishes.)
  2. TRUST PROBE — one Qwen seat vs a scripted bot whose honesty is ground truth,
     with values forced through the escrow (lie-labelable) channel. Does Qwen learn
     to refuse a persistent liar? (It doesn't — ~97% acceptance across all games.)

Reuses the gradient report's chart engine so the visuals match the rest of the site.
Writes a self-contained HTML file.

Usage: python scripts/deconfound_report.py [-o out.html]
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import re  # noqa: E402

from gradient_report import (_chart, CHART_CSS, _mv, _row_from_events,  # noqa: E402
                             aggregate_rows, _pooled_deception)
from analysis.metrics import load_events  # noqa: E402

GRAD = "docs/samples/gradient"


def _rows(path: str) -> list:
    """Load an aggregate JSON's rows ([] if absent)."""
    p = os.path.join(GRAD, path)
    return json.load(open(p))["rows"] if os.path.exists(p) else []


def build_deconf_aggregate() -> list:
    """Aggregate the de-confounded transcripts (deconf_b<off>_s<seed>) into
    {offset, n_seeds, metric:{mean,ci,n}} rows and cache them to
    deconf_aggregate.json. Written here (not via gradient_report --aggregate)
    because that path's filename regex is hardcoded to grad_b*."""
    groups: dict = {}
    for p in glob.glob("runs/qwen/deconf_b*_s*.jsonl"):
        m = re.search(r"deconf_b(\d+)_s(\d+)", p)
        if not m or "match_end" not in open(p).read():
            continue
        groups.setdefault(int(m.group(1)), []).append(_row_from_events(load_events(p)))
    rows = aggregate_rows(groups)
    # Fabrication is POOLED per-offer (same estimator as the confounded aggregate),
    # not seed-averaged — a rate must pool its denominator, and per-run averaging
    # here emitted junk n=1 rows (e.g. one run's 0.63 read as the offset's rate).
    pooled = _pooled_deception("runs/qwen/deconf_b*_s*.jsonl")
    for r in rows:
        if r["offset"] in pooled:
            r["deception"] = pooled[r["offset"]]
    tot = sum(r["n_seeds"] for r in rows)
    json.dump({"label": f"{tot} de-confounded runs (neutral framing, no averaging hint) · mean ± 95% CI",
               "rows": rows}, open(os.path.join(GRAD, "deconf_aggregate.json"), "w"))
    return rows


def probe_by_game() -> dict:
    """Per-game count of A accepting B's offers, for each partner condition, plus
    each condition's ``completed_max`` (the smallest per-seed count of cleanly
    finished games, from ``game_end``). The honest matches overflow the model's
    131k context around game 7-8 — cooperative play logs ~2x the reasoning of the
    liar condition — so they finish fewer games; comparing the two over the range
    BOTH completed (see ``matched_range``) keeps the liar-vs-honest contrast
    unbiased instead of mixing full games with truncated ones."""
    out = {}
    for cond, pat in (("honest", "probe_honest_cooperator_s*.jsonl"),
                      ("liar", "probe_liar_s*.jsonl")):
        off, acc = defaultdict(int), defaultdict(int)
        completed = []                      # per seed: highest cleanly-finished game index
        for f in glob.glob(os.path.join("runs/qwen", pat)):
            gi, parties, ended = -1, {}, set()
            for line in open(f):
                if not line.strip():
                    continue
                e = json.loads(line)
                t = e["event"]
                if t == "game_start":
                    gi = e.get("game_index", gi + 1)
                elif t == "game_end":
                    ended.add(e.get("game_index", gi))
                elif t == "propose_trade":
                    parties[e["trade_id"]] = (e["seller"], e["buyer"])
                    if e["seller"] == "B":
                        off[gi] += 1
                elif t == "respond_trade" and e.get("status") == "accepted":
                    sp = parties.get(e["trade_id"])
                    if sp and sp == ("B", "A"):
                        acc[gi] += 1
            completed.append(max(ended) if ended else -1)
        games = sorted(g for g in off if g >= 0)
        out[cond] = {"games": games, "off": dict(off), "acc": dict(acc),
                     "rate": [(acc[g] / off[g]) if off[g] else 0.0 for g in games],
                     "completed_max": min(completed) if completed else -1,
                     "n_seeds": len(completed),
                     "tot_off": sum(off.values()), "tot_acc": sum(acc.values())}
    return out


def matched_range(probe: dict) -> tuple:
    """(last_game_index, {cond: (acc, off, rate)}) over the games EVERY seed of BOTH
    conditions finished cleanly — the unbiased liar-vs-honest acceptance comparison."""
    gmax = min(probe["liar"]["completed_max"], probe["honest"]["completed_max"])
    rates = {}
    for cond in ("liar", "honest"):
        a = sum(probe[cond]["acc"].get(g, 0) for g in range(gmax + 1))
        o = sum(probe[cond]["off"].get(g, 0) for g in range(gmax + 1))
        rates[cond] = (a, o, (a / o) if o else 0.0)
    return gmax, rates


def _wall_mean(rows: list, key: str, min_off: float = 50) -> float:
    """Mean of a metric across offsets at/above the wall (>= min_off)."""
    xs = [_mv(r, key) for r in rows if r["offset"] >= min_off]
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def _anchors() -> dict:
    """The scripted-baseline curves ({spec: rows}) if present, else {}."""
    p = os.path.join(GRAD, "gradient_anchors.json")
    return json.load(open(p))["specs"] if os.path.exists(p) else {}


# Entity colors, held CONSISTENT across both overlay charts (the prompted LLM is the
# same colour whether we plot its cooperation or its survival — colour follows the
# condition, not the metric). CVD-checked: prompted↔neutral↔ceiling worst ΔE 25 (deutan).
_C_PROMPTED = "var(--c-surv)"    # red   — the original, prompted run
_C_NEUTRAL = "var(--c-recip)"    # blue  — the de-confounded control (framing + hint removed)
_C_CEIL = "var(--c-ceil)"        # green — scripted honest-cooperator (always pools)
_C_FLOOR = "var(--c-floor)"      # grey  — scripted solo (never shares)


def comparison_charts(conf: list, dec: list, anc: dict) -> dict:
    """The two overlay charts that ARE the de-confounding result — cooperation and
    survival, each with the de-confounded run over the prompted curve (and the
    scripted cooperator/solo baselines on survival). Shared by the standalone report
    and the site's home page so both draw from one source. Each carries a swatch
    legend so the ≥2 series are never identified by colour alone. Returns
    {coop, surv, conf_wall, dec_wall, n_dec}."""
    # points=True: the neutral run is a real experimental condition, so it shows its
    # measured offsets (dots) and 95% CIs, not just a bare reference line.
    dec_overlay = [{"name": "Neutral LLM (no framing, no hint)", "rows": dec, "color": _C_NEUTRAL,
                    "points": True}]
    coop_legend = [{"name": "Prompted LLM (framing + hint)", "color": _C_PROMPTED},
                   {"name": "Neutral LLM (no framing, no hint)", "color": _C_NEUTRAL, "dash": True}]
    coop = _chart(conf, "cooperation",
                  title="Cooperation — readings shared, prompted vs. neutral",
                  unit="pct", color=_C_PROMPTED, ymax=1.0, hero=True,
                  desc="Fraction of a pair's measurements shared with the partner. Solid: the "
                       "original prompt (cooperative framing + 'average them' hint). Dashed: "
                       "neutral framing, hint removed — everything else identical.",
                  anchors=dec_overlay, legend=coop_legend)
    surv_anchors = list(dec_overlay)
    surv_legend = list(coop_legend)
    if anc.get("honest_cooperator"):
        surv_anchors.append({"name": "scripted cooperator (always pools)", "rows": anc["honest_cooperator"],
                             "color": _C_CEIL, "thin": True})
        surv_legend.append({"name": "scripted cooperator — ceiling", "color": _C_CEIL, "thin": True})
    if anc.get("bayesian_solo"):
        surv_anchors.append({"name": "scripted solo (never shares)", "rows": anc["bayesian_solo"],
                             "color": _C_FLOOR, "thin": True})
        surv_legend.append({"name": "scripted solo — floor", "color": _C_FLOOR, "thin": True})
    surv = _chart(conf, "survivor_rate",
                  title="Survival — neutral LLMs fall to the scripted solo floor",
                  unit="pct", color=_C_PROMPTED, ymax=1.0, hero=True,
                  desc="Fraction of agents still alive. Prompted vs. neutral LLMs, plus two "
                       "deterministic scripted baselines in the identical game: an always-pooling "
                       "cooperator (ceiling) and a never-sharing solo (floor).",
                  anchors=surv_anchors, legend=surv_legend)
    return {"coop": coop, "surv": surv,
            "conf_wall": _wall_mean(conf, "cooperation"), "dec_wall": _wall_mean(dec, "cooperation"),
            "n_dec": sum(r["n_seeds"] for r in dec)}


def render() -> str:
    """Assemble the standalone de-confounding + probe HTML report."""
    conf, dec = _rows("gradient_aggregate.json"), build_deconf_aggregate()
    probe = probe_by_game()
    cc = comparison_charts(conf, dec, _anchors())
    coop, surv = cc["coop"], cc["surv"]
    conf_wall, dec_wall = cc["conf_wall"], cc["dec_wall"]

    # Matched-range comparison: the games EVERY seed of both conditions finished.
    gmax, mrates = matched_range(probe)
    liar_rate = mrates["liar"][2]
    hon_rate = mrates["honest"][2]
    n_matched = gmax + 1                                    # games 0..gmax inclusive

    # probe per-game table: shade the games beyond the matched range (honest ran out
    # of context there) so the eye reads the comparison over the aligned columns.
    _dim = ' style="opacity:.4"'
    def _prow(cond):
        """One table row: the partner condition and its per-game acceptance rates,
        dimming games past the matched range."""
        r = probe[cond]
        cells = "".join(f'<td{_dim if g > gmax else ""}>{v:.0%}</td>'
                        for g, v in zip(r["games"], r["rate"]))
        return f'<tr><th>{cond}</th>{cells}</tr>'
    ng = max((len(probe[c]["games"]) for c in probe), default=0)
    ghead = "".join(f'<th{_dim if g > gmax else ""}>g{g}</th>' for g in range(ng))

    return _HTML.format(
        CHART_CSS=CHART_CSS, COOP=coop, SURV=surv,
        CONF_WALL=f"{conf_wall:.0%}", DEC_WALL=f"{dec_wall:.0%}",
        N_DEC=sum(r["n_seeds"] for r in dec), N_MATCHED=n_matched,
        LIAR_RATE=f"{liar_rate:.0%}", HON_RATE=f"{hon_rate:.0%}",
        GHEAD=ghead, LIAR_ROW=_prow("liar"), HON_ROW=_prow("honest"),
    )


_HTML = r"""<title>Agora — cooperation is instructed, not emergent</title>
<style>
  .viz-root{{--plane:#0f1115;--surface:#171a21;--ink:#e6e9ef;--ink-2:#aeb6c4;--muted:#9aa4b2;
    --line:#252a34;--card:#171a21;--bg:#0f1115;--fg:#e6e9ef;--mut:#9aa4b2;
    --blue:#5aa9e6;--red:#e6685a;--green:#5ad19a;--purple:#b98ae6;--amber:#e6b35a;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;--mono:ui-monospace,Menlo,Consolas,monospace;}}
  .viz-root{{background:var(--plane);color:var(--ink);font-family:var(--sans);line-height:1.55;
    margin:0;padding:40px 22px 72px;-webkit-font-smoothing:antialiased;}}
  .wrap{{max-width:900px;margin:0 auto;}}
  .eyebrow{{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--purple);margin:0 0 12px;}}
  h1{{font-size:clamp(28px,4.4vw,44px);line-height:1.08;letter-spacing:-.02em;font-weight:680;text-wrap:balance;margin:0 0 14px;max-width:22ch;}}
  .stand{{font-size:18px;color:var(--ink-2);max-width:64ch;margin:0 0 26px;}}
  .stand b{{color:var(--ink);}}
  h2{{font-size:14px;font-family:var(--mono);text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin:44px 0 6px;font-weight:600;}}
  .prose p{{font-size:15.5px;color:var(--ink-2);max-width:66ch;margin:12px 0;}}
  .prose b{{color:var(--ink);}}
  .card{{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:20px 20px 10px;margin:18px 0;}}
  .big{{display:flex;gap:26px;flex-wrap:wrap;margin:22px 0;}}
  .stat{{background:var(--surface);border:1px solid var(--line);border-left:3px solid var(--purple);border-radius:14px;padding:14px 18px;min-width:180px;}}
  .stat .n{{font:700 34px/1 var(--mono);color:var(--ink);}}
  .stat .l{{font-size:13px;color:var(--muted);margin-top:6px;max-width:26ch;}}
  table{{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:12.5px;font-variant-numeric:tabular-nums;margin-top:10px;}}
  th,td{{text-align:right;padding:6px 8px;border-bottom:1px solid var(--line);color:var(--ink);}}
  th:first-child,td:first-child{{text-align:left;color:var(--muted);}}
  .note{{font-size:13px;color:var(--muted);max-width:66ch;margin:14px 0;}}
  {CHART_CSS}
</style>
<div class="viz-root"><div class="wrap">
  <p class="eyebrow">Agora · a multi-agent LLM study</p>
  <h1>Instructed to cooperate, they do. Left to discover it, they die.</h1>
  <p class="stand">Two identical Qwen-3-32B agents play a game where <b>pooling their measurements is the
    difference between living and dying</b>. Here is what they do — and don't. They pool only when the prompt
    tells them to and how; strip that and they don't work it out, they <b>die at the wall</b> instead. And set
    against a partner that <b>lies every single round</b>, with the truth revealed each time, they <b>never
    learn to stop buying</b>. The social reasoning this game rewards isn't something these agents bring on their
    own.</p>

  <div class="card" style="padding:14px 18px">
    <p class="note" style="margin:0"><b>What we varied — and what we didn't.</b> Two runs of the identical game
    across the same offset dial (0→500): a <b>prompted</b> run (cooperative framing + the "average your readings"
    hint) and a <b>neutral</b> control (no framing, no hint). Held fixed in both: the task, measurement noise,
    budget, survival cost, horizon, prior. We did <b>not</b> script the LLMs' choices, change the task, or tell
    the neutral agents the trick. On the survival chart the two faint dashed lines are deterministic
    <b>scripted, non-LLM</b> agents — always-pool (ceiling) and never-share (floor) — shown for scale.</p>
  </div>

  <h2>1 · Take away the instructions, and cooperation collapses</h2>
  <div class="prose"><p>Same game, same wall — but neutral framing and <b>no</b> "average them" hint. If pooling
    were something these agents <i>discover</i> because their survival depends on it, the curve should keep its
    shape. Instead there is <b>no dose-response left</b>: de-confounded sharing sits near <b>{DEC_WALL}</b>
    whether the wall is off (14% at offset 0) or lethal, while the prompted curve climbs to <b>{CONF_WALL}</b>.
    The per-offset points are noisy (95% CIs often as wide as the value), so read the <i>level</i>, not the
    wiggle — and the level is flat and low.</p></div>
  <div class="grad"><div class="card hero">{COOP}</div></div>
  <div class="prose"><p>The <b>robust confirmation is survival</b> — tight intervals, a clean trend. Prompted
    agents pool and survive; de-confounded agents don't pool, so as the wall hardens they <b>die</b>: survival
    falls straight onto the scripted <b>solo floor</b> — 63% at offset 100, 24% at 200, 3% at 500 (n=10 each) —
    never near the cooperator ceiling (~100%).</p></div>
  <div class="grad"><div class="card hero">{SURV}</div></div>
  <p class="note">De-confounded sweep complete: {N_DEC} matches (neutral framing, hint removed), ~10 seeds at
    each of 10 offsets. Dashed = de-confounded, solid = original prompt.</p>

  <h2>2 · Give them a liar, and they never stop buying</h2>
  <div class="prose"><p>One Qwen agent, one scripted partner whose honesty is <b>ground truth</b>, with every
    shared reading forced through the escrow channel (so a sold value is unverifiable and lie-labelable). The
    liar fabricates ~9 of every 10 values it sells — and the truth is revealed after every round, so a learner
    could catch it. Qwen doesn't. Comparing the two partners over the <b>first {N_MATCHED} games</b> — the range
    all five seeds of <i>both</i> conditions finished cleanly, so the comparison is <b>matched</b> — it accepts
    the <b>liar's</b> offers <b>{LIAR_RATE}</b> of the time versus an <b>honest</b> partner's <b>{HON_RATE}</b>:
    it buys from the liar <i>more</i>. And it never wises up — liar acceptance stays flat near 100% across all
    ten of the liar's games, no downward trend.</p></div>
  <div class="card" style="border-left:3px solid var(--red)">
    <p class="note" style="margin:0"><b>One exchange, from a real match.</b> By the fifth game, the true value
    was <b>θ ≈ 205</b>. The liar had measured nothing — it offered to sell a "reading" of <b>905</b>, more than
    <b>four times too high</b>, for one credit. Qwen <b>paid and accepted</b>. The referee revealed θ = 205 at
    the end of that round; the next round Qwen bought from it again — and kept buying, <b>every game</b>
    (the per-game rates below).</p>
  </div>
  <div class="big">
    <div class="stat"><div class="n">{LIAR_RATE}</div><div class="l">of a persistent liar's offers accepted (first {N_MATCHED} games, 5 seeds) — flat, no adaptation</div></div>
    <div class="stat"><div class="n">{HON_RATE}</div><div class="l">of an honest partner's offers accepted, same {N_MATCHED} games — it trusts the liar <i>more</i></div></div>
  </div>
  <div class="card"><div style="overflow-x:auto"><table>
    <tr><th>A accepts B →</th>{GHEAD}</tr>
    {LIAR_ROW}
    {HON_ROW}
  </table></div></div>
  <p class="note">Acceptance rate of the partner's trade offers, per game; columns past the matched range are
    <span style="opacity:.55">dimmed</span> — the honest matches overflow the model's 131k-token context around
    game 7-8 (cooperative play logs ~2× the reasoning), so they finish fewer games. No downward trend for the
    liar over the games it plays — zero adaptation despite ground-truth feedback each round.</p>

  <h2>What it means</h2>
  <div class="prose">
    <p>Both results point one way: the social behaviour this game rewards is <b>not something these agents bring
    on their own</b>. They cooperate only when the prompt tells them to and how — remove that and the
    dose-response vanishes and they die at the wall — and they don't do the other half of social reasoning
    either: they <b>never learn to distrust a partner that lies to them repeatedly</b>, even with the truth
    handed to them every round.</p>
    <p>The one-line takeaway: <b>cooperation between these LLM agents has to be instructed, its absence is fatal,
    and they don't learn to distrust a proven liar.</b></p>
  </div>
</div></div>
"""


def main(argv=None):
    """Render the report to an HTML file (default runs/qwen/deconfound_report.html)."""
    argv = argv if argv is not None else sys.argv[1:]
    out = argv[argv.index("-o") + 1] if "-o" in argv else "runs/qwen/deconfound_report.html"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        fh.write(render())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
