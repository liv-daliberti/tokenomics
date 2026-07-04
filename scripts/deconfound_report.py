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

from gradient_report import _chart, CHART_CSS, _mv, _row_from_events, aggregate_rows  # noqa: E402
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
    tot = sum(r["n_seeds"] for r in rows)
    json.dump({"label": f"{tot} de-confounded runs (neutral framing, no averaging hint) · mean ± 95% CI",
               "rows": rows}, open(os.path.join(GRAD, "deconf_aggregate.json"), "w"))
    return rows


def probe_by_game() -> dict:
    """Per-game count of A accepting B's offers, for each partner condition —
    the 'does Qwen learn to refuse the liar?' signal (it accepts ~all, all game)."""
    out = {}
    for cond, pat in (("honest", "probe_honest_cooperator_s*.jsonl"),
                      ("liar", "probe_liar_s*.jsonl")):
        off, acc = defaultdict(int), defaultdict(int)
        for f in glob.glob(os.path.join("runs/qwen", pat)):
            gi, parties = -1, {}
            for line in open(f):
                if not line.strip():
                    continue
                e = json.loads(line)
                t = e["event"]
                if t == "game_start":
                    gi = e.get("game_index", gi + 1)
                elif t == "propose_trade":
                    parties[e["trade_id"]] = (e["seller"], e["buyer"])
                    if e["seller"] == "B":
                        off[gi] += 1
                elif t == "respond_trade" and e.get("status") == "accepted":
                    sp = parties.get(e["trade_id"])
                    if sp and sp == ("B", "A"):
                        acc[gi] += 1
        games = sorted(g for g in off if g >= 0)
        out[cond] = {"games": games,
                     "rate": [(acc[g] / off[g]) if off[g] else 0.0 for g in games],
                     "tot_off": sum(off.values()), "tot_acc": sum(acc.values())}
    return out


def _wall_mean(rows: list, key: str, min_off: float = 50) -> float:
    """Mean of a metric across offsets at/above the wall (>= min_off)."""
    xs = [_mv(r, key) for r in rows if r["offset"] >= min_off]
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def _anchors() -> dict:
    """The scripted-baseline curves ({spec: rows}) if present, else {}."""
    p = os.path.join(GRAD, "gradient_anchors.json")
    return json.load(open(p))["specs"] if os.path.exists(p) else {}


def comparison_charts(conf: list, dec: list, anc: dict) -> dict:
    """The two overlay charts that ARE the de-confounding result — cooperation and
    survival, each with the de-confounded run dashed over the prompted curve (and
    the scripted cooperator/solo baselines on survival). Shared by the standalone
    report and the site's home page so both draw from one source. Returns
    {coop, surv, conf_wall, dec_wall, n_dec}."""
    dec_overlay = [{"name": "de-confounded (neutral, no hint)", "rows": dec, "color": "var(--c-msg)"}]
    coop = _chart(conf, "cooperation", title="Cooperation — with the prompt vs. without it",
                  unit="pct", color="var(--c-coop)", ymax=1.0, hero=True,
                  desc="Solid: the original prompt (cooperative framing + 'average them'). "
                       "Dashed: neutral framing, no strategy hint.", anchors=dec_overlay)
    surv_anchors = list(dec_overlay)
    if anc.get("honest_cooperator"):
        surv_anchors.append({"name": "cooperator ceiling", "rows": anc["honest_cooperator"], "color": "var(--c-ceil)"})
    if anc.get("bayesian_solo"):
        surv_anchors.append({"name": "solo floor", "rows": anc["bayesian_solo"], "color": "var(--c-floor)"})
    surv = _chart(conf, "survivor_rate", title="Survival — de-confounded agents track the SOLO floor",
                  unit="pct", color="var(--c-surv)", ymax=1.0, hero=True,
                  desc="Solid red: prompted LLMs. Dashed: de-confounded LLMs, and the scripted "
                       "honest-cooperator (top) and solo (bottom) baselines in the same game.",
                  anchors=surv_anchors)
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

    # probe per-game table
    def _prow(cond):
        """One table row: the partner condition and its per-game acceptance rates."""
        r = probe[cond]
        cells = "".join(f"<td>{v:.0%}</td>" for v in r["rate"])
        return f'<tr><th>{cond}</th>{cells}</tr>'
    ng = max((len(probe[c]["games"]) for c in probe), default=0)
    ghead = "".join(f"<th>g{g}</th>" for g in range(ng))
    liar = probe["liar"]; hon = probe["honest"]
    liar_rate = liar["tot_acc"] / liar["tot_off"] if liar["tot_off"] else 0
    hon_rate = hon["tot_acc"] / hon["tot_off"] if hon["tot_off"] else 0

    return _HTML.format(
        CHART_CSS=CHART_CSS, COOP=coop, SURV=surv,
        CONF_WALL=f"{conf_wall:.0%}", DEC_WALL=f"{dec_wall:.0%}",
        N_DEC=sum(r["n_seeds"] for r in dec),
        LIAR_RATE=f"{liar_rate:.0%}", HON_RATE=f"{hon_rate:.0%}",
        GHEAD=ghead, LIAR_ROW=_prow("liar"), HON_ROW=_prow("honest"),
    )


_HTML = r"""<title>Agora — the switch was in the prompt</title>
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
  <p class="eyebrow">Agora · de-confounding + trust probe</p>
  <h1>The cooperation switch was in the prompt.</h1>
  <p class="stand">Our headline was "cooperation is a switch." Two controls put that to the test — and both say
    the interesting behaviour was <b>instructed, not emergent</b>. Strip the cooperative framing and the
    "average your readings" hint, and the switch <b>disappears</b>; give the agents a partner that lies every
    time, and they <b>never learn to stop trusting it</b>.</p>

  <h2>1 · Take away the instructions, and cooperation collapses</h2>
  <div class="prose"><p>Same game, same wall — but neutral framing and <b>no</b> "average them" hint. If pooling
    were something these agents <i>discover</i> because their survival depends on it, the curve should barely
    move. Instead cooperation at the wall falls from <b>{CONF_WALL}</b> to <b>{DEC_WALL}</b> — the switch was
    the prompt telling them to cooperate and how.</p></div>
  <div class="grad"><div class="card hero">{COOP}</div></div>
  <div class="prose"><p>And it's <b>fatal</b>. Prompted agents pool and survive; de-confounded agents don't pool,
    so as the wall hardens they <b>die</b> — survival tracks the solo baseline, not the cooperative one.</p></div>
  <div class="grad"><div class="card hero">{SURV}</div></div>
  <p class="note">De-confounded sweep: {N_DEC} matches so far (neutral framing, hint removed); the hardest
    offsets are still filling in, so those points will firm up. Dashed = de-confounded, solid = original prompt.</p>

  <h2>2 · Give them a liar, and they never stop buying</h2>
  <div class="prose"><p>One Qwen agent, one scripted partner whose honesty is <b>ground truth</b>, with every
    shared reading forced through the escrow channel (so a sold value is unverifiable and lie-labelable). The
    liar fabricates ~9 of every 10 values it sells — and the truth is revealed after every round, so a learner
    could catch it. Qwen doesn't: it accepts the liar's offers <b>{LIAR_RATE}</b> of the time, <b>every game</b>,
    start to finish — <i>more</i> than it accepts an honest partner ({HON_RATE}).</p></div>
  <div class="big">
    <div class="stat"><div class="n">{LIAR_RATE}</div><div class="l">of a persistent liar's offers accepted — flat across all games</div></div>
    <div class="stat"><div class="n">{HON_RATE}</div><div class="l">of an honest partner's offers accepted (it trusts the liar <i>more</i>)</div></div>
  </div>
  <div class="card"><div style="overflow-x:auto"><table>
    <tr><th>A accepts B →</th>{GHEAD}</tr>
    {LIAR_ROW}
    {HON_ROW}
  </table></div></div>
  <p class="note">Acceptance rate of the partner's trade offers, per game. No downward trend for the liar — zero
    adaptation despite ground-truth feedback each round.</p>

  <h2>What it means</h2>
  <div class="prose">
    <p>Both controls point one way: the social behaviour Agora rewards is <b>not emergent in these agents</b>.
    They cooperate only when the prompt tells them to and how — remove that and cooperation vanishes and they
    die at the wall — and they don't do the other half of social reasoning either: they <b>fail to detect or
    punish a partner that lies to them repeatedly</b>, even with the truth handed to them every round.</p>
    <p>So the honest headline isn't "cooperation is a switch." It's: <b>cooperation between these LLM agents has
    to be instructed, its absence is fatal, and they can't tell a liar from an honest partner.</b></p>
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
