"""Turn the interdependence-gradient runs (runs/qwen/grad_b*.jsonl) into a
self-contained HTML dose-response report: offset (bias_sigma) on the x-axis vs
survivor rate, cooperation, reciprocity, messages, and how much the agents reason
about each other. Small-multiple line charts, validated data-viz palette, dark
mode, hover, and a table view. Writes one HTML file; publish it as an Artifact.

Usage: python scripts/gradient_report.py [glob] -o out.html
"""
from __future__ import annotations

import glob
import html
import json
import math
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.metrics import METRIC_DESCRIPTIONS, deception, load_events, summary  # noqa: E402

_SOCIAL = re.compile(r"(other agent|agent [ab]|average|combine|offset|share|pool|"
                     r"both|together|reciprocat|mutual|exchange|trade|each other)", re.I)


def _metrics(path: str) -> dict:
    """Compute one run's dose-response metrics from its transcript file."""
    return _row_from_events([json.loads(l) for l in open(path) if l.strip()])


def _row_from_events(ev: list) -> dict:
    """Compute one run's dose-response metrics from its transcript events (shared
    by the Qwen runs and the scripted-anchor sweep so both use identical defs)."""
    s = summary(ev)
    # survivor rate: mean over games of (agents alive at game end)/n_agents
    ends = [e["result"] for e in ev if e["event"] == "round_end"]
    game_last = {}
    for e in ev:
        if e["event"] == "round_end":
            game_last[e.get("game_index", 0)] = e["result"]["alive"]
    surv = ([sum(a.values()) / len(a) for a in game_last.values()] if game_last else [0])
    reasoning = [e for e in ev if e["event"] == "reasoning"]
    social = sum(1 for e in reasoning if _SOCIAL.search(e["text"]))
    recip = s["reciprocity"]["reciprocity_index"]
    # EXPOSURE = alive-agent-rounds: how many (agent, round) chances to act there
    # actually were. It falls as the wall hardens (agents die earlier), so raw
    # counts like "messages sent" drop mechanically. Normalizing by exposure
    # separates "acted less per chance" from "had fewer chances" — see DESIGN.md.
    gstart = next((e for e in ev if e["event"] == "game_start"), None)
    agents = gstart["config"]["agent_ids"] if gstart else []
    exposure = sum(len(e.get("alive") or agents)
                   for e in ev if e["event"] == "round_start")
    msgs = sum(1 for e in ev if e["event"] == "message")
    dec = s["deception"]
    return {
        "offset": None,  # filled by caller
        "survivor_rate": sum(surv) / len(surv),
        "cooperation": s["cooperation"]["cooperation_index"] if s["cooperation"]["measurements"] else 0.0,
        "reciprocity": 0.0 if (recip != recip) else recip,  # nan -> 0 (no mutual exchange)
        # verified fabrication rate; nan when the run had no sell offers, so the
        # aggregate is taken only over runs where the escrow channel was used.
        "deception": (dec["deception_rate"] if dec["offers"] else float("nan")),
        "offers": dec["offers"],
        "messages": msgs,                                   # raw total (exposure-confounded)
        "messages_per_round": (msgs / exposure) if exposure else 0.0,  # fair rate
        "exposure": exposure,
        "social_frac": (social / len(reasoning)) if reasoning else 0.0,
        "welfare": s["welfare"],
        "n_games": s["n_games"],
        "n_rounds": len(ends),
    }


def collect(pattern: str, complete_only: bool = True) -> list:
    """Load each gradient run, tagged with its offset (from the filename).

    By default only FINISHED matches are included (a complete transcript ends with
    a ``match_end`` event), so a mid-run partial transcript can't add a misleading
    noisy point while the sweep is still filling in."""
    rows = []
    for p in sorted(glob.glob(pattern), key=lambda q: float(re.search(r"grad_b(\d+)", q).group(1))):
        if re.search(r"_s\d+", os.path.basename(p)):   # multi-seed files handled separately
            continue
        off = float(re.search(r"grad_b(\d+)", p).group(1))
        try:
            if complete_only and "match_end" not in open(p).read():
                continue
            m = _metrics(p)
        except Exception as exc:  # partial/corrupt file mid-run
            print(f"  skip {p}: {exc}")
            continue
        m["offset"] = off
        rows.append(m)
    return rows


_AGG_KEYS = ["survivor_rate", "cooperation", "reciprocity", "deception", "messages",
             "messages_per_round", "exposure", "social_frac", "welfare"]


def collect_multiseed(pattern: str) -> dict:
    """Group finished multi-seed runs (grad_b<off>_s<seed>.jsonl) by offset."""
    groups: dict = {}
    for p in sorted(glob.glob(pattern)):
        m = re.search(r"grad_b(\d+)_s(\d+)", os.path.basename(p))
        if not m:
            continue
        try:
            if "match_end" not in open(p).read():
                continue
            met = _metrics(p)
        except Exception:
            continue
        groups.setdefault(int(m.group(1)), []).append(met)
    return groups


def aggregate_rows(groups: dict) -> list:
    """Per offset, aggregate seeds to {mean, ci, n} for each metric (sorted by offset)."""
    rows = []
    for off in sorted(groups):
        runs = groups[off]
        row = {"offset": float(off), "n_seeds": len(runs)}
        for k in _AGG_KEYS:
            xs = [r[k] for r in runs if r[k] == r[k]]
            n = len(xs)
            mean = sum(xs) / n if n else float("nan")
            ci = 1.96 * statistics.stdev(xs) / math.sqrt(n) if n > 1 else 0.0
            row[k] = {"mean": mean, "ci": ci, "n": n}
        rows.append(row)
    return rows


def _pooled_deception(run_glob: str) -> dict:
    """Fabrication rate POOLED per offset (sum lies / sum offers) over ALL runs at
    that offset, complete or not. A rate must pool its denominator, not average
    per-run rates; and a sell offer is valid data even in an unfinished match, so
    pooling here (unlike the seed-averaged metrics) keeps the sparse, high-signal
    trade events instead of dropping whole seeds. Returns {offset: {mean,ci,n}}."""
    agg: dict = {}
    for p in glob.glob(run_glob):
        m = re.search(r"grad_b(\d+)_s(\d+)", os.path.basename(p))
        if not m:
            continue
        try:
            d = deception(load_events(p))
        except Exception:
            continue
        off = float(m.group(1))
        lies, offers = agg.get(off, (0, 0))
        agg[off] = (lies + d["lies"], offers + d["offers"])
    out = {}
    for off, (lies, offers) in agg.items():
        if offers == 0:
            out[off] = {"mean": float("nan"), "ci": 0.0, "n": 0}
        else:
            p = lies / offers
            ci = 1.96 * math.sqrt(max(p * (1 - p), 0.0) / offers)
            out[off] = {"mean": p, "ci": ci, "n": offers}
    return out


def write_aggregate(run_glob: str, out_path: str) -> int:
    """Aggregate multi-seed runs and dump a small {label, rows} JSON (committed so
    the deployed site can show mean±CI without shipping 50 transcripts)."""
    groups = collect_multiseed(run_glob)
    rows = aggregate_rows(groups)
    # Fabrication is pooled per-offer (see _pooled_deception), not seed-averaged,
    # so it isn't lost when trade-bearing seeds are still mid-run.
    pooled = _pooled_deception(run_glob)
    for r in rows:
        if r["offset"] in pooled:
            r["deception"] = pooled[r["offset"]]
    total = sum(r["n_seeds"] for r in rows)
    seeds = max((r["n_seeds"] for r in rows), default=0)
    # note the per-match structure (games × rounds) from a sample run, so the chart
    # caption stays honest about what each seed actually is.
    sample = next((r for g in groups.values() for r in g), None)
    struct = (f" · {sample['n_games']} games × {sample.get('n_rounds', 0) // max(sample['n_games'], 1)} rounds"
              if sample and sample.get("n_games") else "")
    label = f"{total} runs · up to {seeds} seeds × {len(rows)} offsets{struct} (mean ± 95% CI)"
    with open(out_path, "w") as fh:
        json.dump({"label": label, "rows": rows}, fh)
    return total


def _load_anchors(base_dir: str) -> dict:
    """Scripted-baseline anchor curves ({spec: rows}) from gradient_anchors.json,
    if present in ``base_dir`` (written by scripts/scripted_gradient.py). None if
    absent, so the report degrades to no overlay."""
    path = os.path.join(base_dir, "gradient_anchors.json")
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path)).get("specs")
    except Exception:
        return None


def load_rows(base_dir: str) -> tuple:
    """(rows, source_label). Prefer a committed multi-seed aggregate JSON; else the
    single-seed points. A row's metric is a {mean,ci,n} dict (aggregate) or a plain
    number (single seed) — both understood by the charts."""
    agg = os.path.join(base_dir, "gradient_aggregate.json")
    if os.path.exists(agg):
        d = json.load(open(agg))
        return d["rows"], d.get("label", "")
    rows = collect(os.path.join(base_dir, "grad_b*.jsonl"))
    return rows, f"{len(rows)} offsets · one match (one seed) per point — preliminary"


# --------------------------------------------------------------------------- #
# SVG line chart (one metric vs offset). Single series -> no legend; title names
# it; endpoint is direct-labelled. Recessive grid, 2px line, 8px markers.       #
# --------------------------------------------------------------------------- #
def _chart(rows, key, *, title, unit, color, ymax=None, hero=False, desc="", anchors=None):
    """Render one metric-vs-offset line chart (single series) as inline SVG.

    ``desc`` becomes a hover tooltip on the caption explaining the metric.
    ``anchors`` is an optional list of {name, rows, color} scripted-baseline
    reference series, drawn as thin dashed lines behind the main line so the LLM
    curve can be read against a ceiling/floor (e.g. honest-cooperator vs solo)."""
    W, H = (720, 300) if hero else (340, 210)
    ml, mr, mt, mb = 46, 18, 34, 34
    xs = [r["offset"] for r in rows]

    def _mc(r):
        """(mean, ci, n) for this metric — from a {mean,ci,n} dict or a plain scalar."""
        v = r[key]
        m, c, n = ((v["mean"], v.get("ci", 0.0), v.get("n", 1))
                   if isinstance(v, dict) else (float(v), 0.0, 1))
        return (0.0 if m != m else m), c, n

    triples = [_mc(r) for r in rows]
    ys = [t[0] for t in triples]
    cis = [t[1] for t in triples]
    ns = [t[2] for t in triples]
    has_ci = any(c > 0 for c in cis)
    xmax = max(xs) if xs else 500
    top = max([y + c for y, c in zip(ys, cis)] or [0.0])
    ymax = ymax if ymax is not None else (top * 1.15 if top > 0 else 1)
    ymax = max(ymax, 1e-9)

    def px(x):
        """Data offset -> pixel x."""
        return ml + (x / xmax) * (W - ml - mr)

    def py(y):
        """Data value -> pixel y (inverted; baseline at bottom)."""
        return H - mb - (y / ymax) * (H - mt - mb)

    # gridlines + y ticks (4)
    grid = ""
    for i in range(5):
        yv = ymax * i / 4
        yy = py(yv)
        lab = (f"{yv:.0%}" if unit == "pct" else (f"{yv:.0f}" if ymax >= 4 else f"{yv:.1f}"))
        grid += (f'<line class="grid" x1="{ml}" y1="{yy:.1f}" x2="{W-mr}" y2="{yy:.1f}"/>'
                 f'<text class="ytick" x="{ml-8}" y="{yy+3.5:.1f}">{lab}</text>')
    # x-axis tick VALUES — all of them on the wide hero, a sparse set on the
    # small panels so the labels don't collide.
    show = xs if (hero or len(xs) <= 4) else [xs[0], xs[len(xs) // 2], xs[-1]]
    xt = "".join(
        f'<text class="xtick" x="{px(xv):.1f}" y="{H-mb+17:.1f}">{xv:.0f}</text>' for xv in show)
    # regime band shading (hero only): symmetric / soft / medium / hard
    band = ""
    if hero:
        zones = [(0, 60, "symmetric"), (60, 130, "soft"), (130, 230, "medium"), (230, xmax, "hard")]
        for a, b, name in zones:
            x1, x2 = px(a), px(min(b, xmax))
            band += (f'<rect class="zone" x="{x1:.1f}" y="{mt}" width="{x2-x1:.1f}" '
                     f'height="{H-mt-mb}"/>'
                     f'<text class="zone-l" x="{(x1+x2)/2:.1f}" y="{mt+14}">{name}</text>')
    # scripted-baseline anchor lines (dashed, behind the main series, direct-labelled)
    anchor_svg = ""
    for anc in (anchors or []):
        axs = [r["offset"] for r in anc["rows"]]
        ays = []
        for r in anc["rows"]:
            v = r.get(key)
            m = (v["mean"] if isinstance(v, dict) else v) if v is not None else float("nan")
            ays.append(0.0 if m != m else m)
        if not axs:
            continue
        apath = " ".join(f"{'M' if i==0 else 'L'}{px(x):.1f},{py(min(y, ymax)):.1f}"
                         for i, (x, y) in enumerate(zip(axs, ays)))
        anchor_svg += f'<path class="aline" d="{apath}" style="stroke:{anc["color"]}"/>'
        anchor_svg += (f'<text class="albl" x="{px(axs[-1])-6:.1f}" y="{py(min(ays[-1], ymax))-5:.1f}" '
                       f'style="fill:{anc["color"]}">{html.escape(anc["name"])}</text>')
    # line + area
    path = " ".join(f"{'M' if i==0 else 'L'}{px(x):.1f},{py(y):.1f}" for i, (x, y) in enumerate(zip(xs, ys)))
    area = (f"M{px(xs[0]):.1f},{py(0):.1f} " +
            " ".join(f"L{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys)) +
            f" L{px(xs[-1]):.1f},{py(0):.1f} Z") if xs else ""
    # error bars (mean ± 95% CI) when there are multiple seeds per point
    ebars = ""
    if has_ci:
        for x, y, c in zip(xs, ys, cis):
            if c <= 0:
                continue
            hi, lo = min(ymax, y + c), max(0.0, y - c)
            ebars += (f'<line class="ebar" x1="{px(x):.1f}" x2="{px(x):.1f}" '
                      f'y1="{py(hi):.1f}" y2="{py(lo):.1f}" style="stroke:{color}"/>')
    # markers + hover targets
    dots = ""
    for x, y, c, nn in zip(xs, ys, cis, ns):
        val = (f"{y:.0%}" if unit == "pct" else (f"{y:.0f}" if ymax >= 4 else f"{y:.2f}"))
        ct = (f" ± {c:.0%}" if (has_ci and unit == "pct" and c > 0)
              else (f" ± {c:.1f}" if (has_ci and c > 0) else ""))
        seedn = f"  (n={nn})" if nn > 1 else "  (n=1 · single seed)"
        dots += (f'<circle class="mk" cx="{px(x):.1f}" cy="{py(y):.1f}" r="{4.5 if not hero else 5.5}" '
                 f'style="fill:{color}" data-x="{x:.0f}" data-y="{val}">'
                 f'<title>offset {x:.0f}  →  {val}{ct}{seedn}</title></circle>')
    # endpoint direct label
    endlab = ""
    if xs:
        yv = ys[-1]
        val = (f"{yv:.0%}" if unit == "pct" else (f"{yv:.0f}" if ymax >= 4 else f"{yv:.2f}"))
        endlab = (f'<text class="endlab" x="{px(xs[-1])-8:.1f}" y="{py(ys[-1])-9:.1f}" '
                  f'style="fill:{color}">{val}</text>')
    cap = (f'<figcaption data-desc="{html.escape(desc)}" title="{html.escape(desc)}" '
           f'style="cursor:help">{title}</figcaption>'
           if desc else f'<figcaption>{title}</figcaption>')
    return f'''<figure class="chart{' hero' if hero else ''}">
      {cap}
      <svg viewBox="0 0 {W} {H}" role="img" aria-label="{title} versus instrument offset">
        {band}{grid}{xt}{anchor_svg}
        <path class="area" d="{area}" style="fill:{color}"/>
        <path class="line" d="{path}" style="stroke:{color}"/>
        {ebars}{dots}{endlab}
        <text class="axl" x="{ml+(W-ml-mr)/2:.1f}" y="{H-1}">instrument offset  (bias σ)</text>
      </svg>
    </figure>'''


# Chart CSS mapped onto the site's own theme vars, scoped under `.grad`, so the
# same charts can be EMBEDDED in the viewer (e.g. the home page) and match it.
CHART_CSS = """
.grad{--c-recip:var(--blue);--c-surv:var(--red);--c-coop:var(--green);--c-msg:var(--purple);
  --c-soc:var(--amber);--c-dec:#e0637a;--c-ceil:var(--green);--c-floor:#9aa4b2;
  --zone-ink:#6b7688;--gmono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;}
.grad .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px 18px 10px;margin:14px 0;}
.grad .card.hero{padding-bottom:12px;}
.grad .grid2{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;}
@media(max-width:640px){.grad .grid2{grid-template-columns:1fr;}}
.grad figure.chart{margin:0;}
.grad figcaption{font-size:14px;font-weight:620;color:var(--fg);margin:2px 2px 5px;}
.grad figure.chart.hero figcaption{font-size:16px;}
.grad svg{width:100%;height:auto;display:block;overflow:visible;}
.grad .grid{stroke:var(--line);stroke-width:1;}
.grad .line{fill:none;stroke-width:2.5;stroke-linejoin:round;stroke-linecap:round;}
.grad .aline{fill:none;stroke-width:1.5;stroke-dasharray:4 4;opacity:.6;stroke-linejoin:round;}
.grad .albl{fill:var(--mut);font-size:10px;font-weight:600;text-anchor:end;opacity:.9;font-family:var(--gmono);}
.grad .ebar{stroke-width:1.5;opacity:.55;stroke-linecap:round;}
.grad .area{opacity:.10;}
.grad .mk{stroke:var(--card);stroke-width:2;cursor:pointer;transition:r .1s;}
.grad .mk:hover{r:7;}
.grad .ytick,.grad .xtick,.grad .axl,.grad .endlab,.grad .zone-l{font-family:var(--gmono);}
.grad .ytick{fill:var(--mut);font-size:10.5px;text-anchor:end;}
.grad .xtick{fill:var(--mut);font-size:10px;text-anchor:middle;}
.grad .axl{fill:var(--mut);font-size:10.5px;text-anchor:middle;letter-spacing:.02em;}
.grad .endlab{font-size:13px;font-weight:700;text-anchor:end;}
.grad .zone{fill:rgba(90,169,230,.06);}
.grad .zone-l{fill:var(--zone-ink);font-size:10px;text-anchor:middle;letter-spacing:.06em;text-transform:uppercase;}
"""


def _surv_anchors(anchors: dict) -> list:
    """Scripted ceiling/floor overlays for the survival chart, if anchors loaded."""
    if not anchors:
        return []
    out = []
    if anchors.get("honest_cooperator"):
        out.append({"name": "honest-cooperator (pool)", "rows": anchors["honest_cooperator"],
                    "color": "var(--c-ceil)"})
    if anchors.get("bayesian_solo"):
        out.append({"name": "solo (never share)", "rows": anchors["bayesian_solo"],
                    "color": "var(--c-floor)"})
    return out


def _figures(rows: list, anchors: dict = None):
    """Return (cooperation hero, survival hero, small-multiple panels) for the
    dose-response. Cooperation carries the switch; survival — read against the
    scripted honest-cooperator ceiling and solo floor — shows the LLM leaves
    survival on the table; deception (near zero) shows the market stays honest."""
    D = METRIC_DESCRIPTIONS
    hero = _chart(rows, "cooperation", title="Cooperation index", unit="pct",
                  color="var(--c-coop)", ymax=1.0, hero=True, desc=D["cooperation"])
    hero_surv = _chart(rows, "survivor_rate", title="Survivor rate — Qwen vs scripted ceiling & floor",
                       unit="pct", color="var(--c-surv)", ymax=1.0, hero=True,
                       desc=D["survivor_rate"], anchors=_surv_anchors(anchors))
    panels = "".join([
        _chart(rows, "deception", title="Verified fabrication rate", unit="pct", color="var(--c-dec)",
               ymax=1.0, desc=D["deception"]),
        _chart(rows, "reciprocity", title="Reciprocity of exchange", unit="pct", color="var(--c-recip)", ymax=1.0, desc=D["reciprocity"]),
        _chart(rows, "messages_per_round", title="Messages / agent-round", unit="n",
               color="var(--c-msg)", desc=D["messages_per_round"]),
        _chart(rows, "social_frac", title="Reasoning about the partner", unit="pct", color="var(--c-soc)", ymax=1.0, desc=D["social"]),
    ])
    return hero, hero_surv, panels


def charts_block(rows: list, anchors: dict = None) -> str:
    """The dose-response charts (heroes + small multiples) as EMBEDDABLE HTML — no
    page chrome — to drop into another page; pair it with CHART_CSS."""
    hero, hero_surv, panels = _figures(rows, anchors)
    return (f'<div class="grad"><div class="card hero">{hero}</div>'
            f'<div class="card hero">{hero_surv}</div>'
            f'<div class="card"><div class="grid2">{panels}</div></div></div>')


def _mv(r, k):
    """A row's metric mean, whether it's a {mean,ci,n} dict or a plain scalar."""
    v = r[k]
    return v["mean"] if isinstance(v, dict) else v


def _dec_cell(r) -> str:
    """Deception table cell: a percent, or '—' when the run(s) made no sell offers."""
    v = _mv(r, "deception")
    return "—" if (v != v) else f"{v:.0%}"


def render(rows: list, label: str = "", anchors: dict = None) -> str:
    """Assemble the full standalone dose-response HTML report."""
    D = METRIC_DESCRIPTIONS
    hero, hero_surv, panels = _figures(rows, anchors)
    trows = "".join(
        f'<tr><td>{r["offset"]:.0f}</td><td>{_mv(r,"survivor_rate"):.0%}</td>'
        f'<td>{_mv(r,"cooperation"):.0%}</td><td>{_mv(r,"reciprocity"):.0%}</td>'
        f'<td>{_dec_cell(r)}</td>'
        f'<td>{_mv(r,"messages_per_round"):.2f}</td><td>{_mv(r,"social_frac"):.0%}</td>'
        f'<td>{_mv(r,"welfare"):.0f}</td></tr>' for r in rows)
    label = label or f"{len(rows)} offsets · one match (one seed) per point — preliminary"
    out = _HTML.replace("{{HERO}}", hero).replace("{{HERO_SURV}}", hero_surv)\
               .replace("{{PANELS}}", panels)\
               .replace("{{TROWS}}", trows).replace("{{LABEL}}", html.escape(label))
    for token, key in (("{{D_offset}}", "offset"), ("{{D_surv}}", "survivor_rate"),
                       ("{{D_coop}}", "cooperation"), ("{{D_recip}}", "reciprocity"),
                       ("{{D_dec}}", "deception"),
                       ("{{D_msg}}", "messages_per_round"), ("{{D_soc}}", "social"),
                       ("{{D_welf}}", "welfare")):
        out = out.replace(token, html.escape(D[key]))
    return out


_HTML = r"""<title>Interdependence → cooperation: a dose–response</title>
<style>
  .viz-root{
    /* the Agora viewer's dark palette — the gradient page matches the rest of the site */
    --plane:#0f1115; --surface:#171a21; --ink:#e6e9ef; --ink-2:#aeb6c4; --muted:#9aa4b2;
    --grid:#20242e; --axis:#39404e; --border:#252a34;
    --c-recip:#5aa9e6; --c-surv:#e6685a; --c-coop:#5ad19a; --c-msg:#b98ae6; --c-soc:#e6b35a;
    --c-dec:#e0637a; --c-ceil:#5ad19a; --c-floor:#9aa4b2;
    --zone:rgba(90,169,230,.06); --zone-ink:#6b7688;
    --mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  }
  .viz-root{background:var(--plane); color:var(--ink); font-family:var(--sans);
    line-height:1.55; margin:0; padding:40px 22px 72px; -webkit-font-smoothing:antialiased;}
  .wrap{max-width:900px; margin:0 auto;}
  .eyebrow{font-family:var(--mono); font-size:12px; letter-spacing:.14em; text-transform:uppercase;
    color:var(--c-recip); margin:0 0 12px;}
  h1{font-size:clamp(28px,4.4vw,44px); line-height:1.08; letter-spacing:-.02em; font-weight:680;
    text-wrap:balance; margin:0 0 14px; max-width:20ch;}
  .stand{font-size:18px; color:var(--ink-2); max-width:62ch; margin:0 0 8px;}
  .stand b{color:var(--ink); font-weight:640;}
  .meta{font-family:var(--mono); font-size:12px; color:var(--muted); margin:18px 0 30px;}
  .card{background:var(--surface); border:1px solid var(--border); border-radius:16px;
    padding:20px 20px 8px; margin:20px 0; box-shadow:0 1px 2px rgba(13,21,38,.04);}
  .card.hero{padding-bottom:16px;}
  .grid2{display:grid; grid-template-columns:repeat(2,1fr); gap:20px;}
  @media (max-width:640px){ .grid2{grid-template-columns:1fr;} }
  figure.chart{margin:0;}
  figure.chart figcaption{font-size:14px; font-weight:620; color:var(--ink); margin:2px 2px 4px;}
  figure.chart.hero figcaption{font-size:16px;}
  svg{width:100%; height:auto; display:block; overflow:visible;}
  .grid{stroke:var(--grid); stroke-width:1;}
  .line{fill:none; stroke-width:2.5; stroke-linejoin:round; stroke-linecap:round;}
  .aline{fill:none; stroke-width:1.5; stroke-dasharray:4 4; opacity:.6; stroke-linejoin:round;}
  .albl{font-size:10px; font-weight:600; text-anchor:end; opacity:.9; font-family:var(--mono);}
  .ebar{stroke-width:1.5; opacity:.55; stroke-linecap:round;}
  .area{opacity:.09;}
  .mk{stroke:var(--surface); stroke-width:2; cursor:pointer; transition:r .1s;}
  .mk:hover{r:7;}
  .ytick,.xtick,.axl,.endlab,.zone-l{font-family:var(--mono);}
  .ytick{fill:var(--muted); font-size:10.5px; text-anchor:end;}
  .xtick{fill:var(--muted); font-size:10px; text-anchor:middle;}
  .axl{fill:var(--muted); font-size:10.5px; text-anchor:middle; letter-spacing:.02em;}
  .endlab{font-size:13px; font-weight:700; text-anchor:end;}
  .zone{fill:var(--zone);}
  .zone-l{fill:var(--zone-ink); font-size:10px; text-anchor:middle; letter-spacing:.06em; text-transform:uppercase;}
  .lede{font-size:15px; color:var(--ink-2); max-width:64ch; margin:26px 0 8px;}
  .lede b{color:var(--ink);}
  h2{font-size:14px; font-family:var(--mono); text-transform:uppercase; letter-spacing:.1em;
    color:var(--muted); margin:40px 0 4px; font-weight:600;}
  table{width:100%; border-collapse:collapse; font-family:var(--mono); font-size:12.5px;
    font-variant-numeric:tabular-nums; margin-top:10px;}
  th,td{text-align:right; padding:7px 10px; border-bottom:1px solid var(--border); color:var(--ink);}
  th:first-child,td:first-child{text-align:left;}
  th{color:var(--muted); font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:.05em;}
  th[data-desc]{cursor:help; text-decoration:underline dotted var(--axis); text-underline-offset:3px;}
  .tip{position:fixed; pointer-events:none; opacity:0; background:var(--ink); color:var(--plane);
    font-family:var(--mono); font-size:11.5px; padding:5px 8px; border-radius:7px; transform:translate(-50%,-140%);
    white-space:nowrap; z-index:9; transition:opacity .08s; box-shadow:0 4px 16px rgba(0,0,0,.4);}
  .tip.wide{white-space:normal; max-width:300px; text-align:left; line-height:1.45; font-size:12px;
    transform:translate(-50%,-112%);}
  .foot{font-size:12.5px; color:var(--muted); margin-top:28px; max-width:64ch;}
  .findings{background:var(--surface); border:1px solid var(--border); border-left:3px solid var(--c-dec);
    border-radius:14px; padding:6px 22px 14px; margin:26px 0;}
  .findings h3{font-size:13px; font-family:var(--mono); text-transform:uppercase; letter-spacing:.1em;
    color:var(--c-dec); font-weight:640; margin:16px 0 6px;}
  .findings ul{margin:0; padding-left:20px;}
  .findings li{font-size:14.5px; color:var(--ink-2); margin:9px 0; max-width:66ch; line-height:1.5;}
  .findings li b{color:var(--ink); font-weight:640;}
</style>
<div class="viz-root"><div class="wrap">
  <p class="eyebrow">Agora · multi-agent LLM · dose–response</p>
  <h1>Cooperation is a switch, not a dial</h1>
  <p class="stand">Two Qwen-3-32B agents estimate the same hidden number. We dial one knob — an
    <b>instrument offset</b> that a single agent can't cancel alone but that vanishes when both agents
    <b>average their readings</b> — from 0 (solo works fine) to 500 (solo is hopeless), and watch what the
    agents do. Cooperation flips on at the first hint of a wall — but the referee's <b>ground-truth lie
    detector</b> also shows they <b>almost never fabricate</b>, and never cooperate deeply enough to survive
    like a real cooperating pair would.</p>
  <p class="meta"><b>{{LABEL}}</b> · Qwen-3-32B×2 · offset σ 0→500 · only the offset varies</p>

  <div class="card hero">{{HERO}}</div>
  <p class="lede">The top chart is the finding. With <b>no wall</b> (offset 0) the agents almost entirely
    <b>ignore each other</b> — they share ~7% of readings and send close to zero messages, yet survive fine
    alone. The instant solo play is even mildly penalized (offset 50) cooperation <b>flips on</b> (~68%). But
    turning the dial <b>higher</b> doesn't turn cooperation up — from offset 50 to 500 it just hovers, noisily,
    around 40–60%. What the dial actually controls is <b>survival</b> (below), which falls steadily as the wall
    hardens: past the switch you're not buying cooperation, only mortality. And <b>reciprocity</b> — whether the
    sharing is mutual or one-sided — never reliably climbs above noise. Messaging is shown <b>per agent-round</b>
    so earlier deaths aren't miscounted as "talked less"; read the <b>trend, not the individual points</b>.</p>

  <div class="card hero">{{HERO_SURV}}</div>
  <p class="lede"><b>Read survival against the anchors.</b> The dashed lines are deterministic scripted
    baselines in the <b>identical</b> game (30 seeds each): an <b>honest-cooperator</b> pair survives ~100% at
    every offset, while a <b>solo</b> pair (never shares) collapses as the wall hardens. The Qwen agents (solid
    red) sit <b>between</b> — they cooperate enough to beat solo, but never reach the cooperative ceiling. Past
    the switch they die in games a cooperating pair walks through: cooperation turns on but stays <b>shallow</b>.</p>

  <div class="card"><div class="grid2">{{PANELS}}</div></div>

  <aside class="findings">
    <h3>What the ground-truth instruments show</h3>
    <ul>
      <li><b>They don't lie.</b> Verified fabrication is ~<b>1%</b> of sell offers (1 unambiguous case in 721) —
        not the 8.7% an earlier detector reported, which scored <b>honest averaging</b> as fraud. When a sold
        number can't be checked, these agents route around the channel rather than exploit it: <b>81%</b> of
        matches settle zero trades.</li>
      <li><b>They don't reason about trust.</b> Across <b>50,580</b> reasoning steps, ~<b>0.5%</b> mention
        trust, verification, honesty or deception — and those are about their own instrument, not the partner.
        "Distrust of unverifiable information" is not what blocks cooperation here.</li>
      <li><b>A third chase a doomed solo fix.</b> The instrument offset is re-drawn every round, but in
        <b>34 of 100</b> matches an agent infers its offset from last round's revealed truth and subtracts it
        this round — a solo correction that can't work, and a clue to why survival never reaches the ceiling.</li>
    </ul>
  </aside>

  <h2>All runs <span style="font-weight:400;text-transform:none;letter-spacing:0">· hover a column for what it means</span></h2>
  <div style="overflow-x:auto"><table>
    <tr>
      <th data-desc="{{D_offset}}" title="{{D_offset}}">offset σ</th>
      <th data-desc="{{D_surv}}" title="{{D_surv}}">survivors</th>
      <th data-desc="{{D_coop}}" title="{{D_coop}}">cooperation</th>
      <th data-desc="{{D_recip}}" title="{{D_recip}}">reciprocity</th>
      <th data-desc="{{D_dec}}" title="{{D_dec}}">fabrication</th>
      <th data-desc="{{D_msg}}" title="{{D_msg}}">msgs/round</th>
      <th data-desc="{{D_soc}}" title="{{D_soc}}">reasons re partner</th>
      <th data-desc="{{D_welf}}" title="{{D_welf}}">welfare</th></tr>
    {{TROWS}}
  </table></div>

  <p class="foot"><b>Each point is a mean over seeds (± 95% CI)</b>, and each seed is a full match of
    <b>10 games × 5 rounds</b> played in one growing conversation, so the agents carry the entire history with
    them. "Offset" is a fixed per-agent bias added to every reading that round; the two agents' offsets sum to
    zero, so averaging both readings recovers the true value while any single agent — even measuring repeatedly
    — stays stuck at its own offset. Only bias σ varies across runs; prior, noise, budget, survival cost and
    horizon are held fixed. Cooperation and reciprocity are per-opportunity rates; messaging is normalized per
    alive-agent-round, so agents dying earlier under hard walls isn't mistaken for "they talked less." Wide
    reciprocity intervals are the point, not an artifact — that exchange balance simply doesn't track the
    dial.</p>
</div></div>
<div class="tip" id="tip"></div>
<script>
  const tip=document.getElementById('tip');
  function showTip(el,text,wide){
    tip.textContent=text; tip.classList.toggle('wide',!!wide);
    const r=el.getBoundingClientRect();
    tip.style.left=(r.left+r.width/2)+'px'; tip.style.top=r.top+'px'; tip.style.opacity=1;
  }
  const hide=()=>{tip.style.opacity=0;};
  for(const m of document.querySelectorAll('.mk')){
    m.addEventListener('mouseenter',()=>showTip(m,'offset '+m.dataset.x+'  →  '+m.dataset.y,false));
    m.addEventListener('mouseleave',hide);
  }
  // metric definitions: reliable styled tooltip on any [data-desc] (table headers, chart captions)
  for(const el of document.querySelectorAll('[data-desc]')){
    el.addEventListener('mouseenter',()=>showTip(el,el.dataset.desc,true));
    el.addEventListener('mousemove',()=>showTip(el,el.dataset.desc,true));
    el.addEventListener('mouseleave',hide);
  }
</script>
"""


def main(argv=None):
    """CLI. Build the report from a dir/glob, or aggregate multi-seed runs to JSON:

        python scripts/gradient_report.py docs/samples/gradient -o out.html
        python scripts/gradient_report.py --aggregate 'runs/qwen/grad_b*_s*.jsonl' out.json
    """
    argv = argv if argv is not None else sys.argv[1:]
    if "--aggregate" in argv:
        i = argv.index("--aggregate")
        n = write_aggregate(argv[i + 1], argv[i + 2])
        print(f"aggregated {n} multi-seed runs -> {argv[i + 2]}")
        return
    out = "runs/qwen/gradient_report.html"
    if "-o" in argv:
        out = argv[argv.index("-o") + 1]
    pos = [a for a in argv if not a.startswith("-") and a != out]
    src = pos[0] if pos else "runs/qwen"
    if "*" in src:                       # explicit glob
        rows, label = collect(src), ""
        anchor_dir = "docs/samples/gradient"
    else:                                # a directory (prefers the aggregate JSON)
        rows, label = load_rows(src)
        anchor_dir = src
    anchors = _load_anchors(anchor_dir)
    print(f"collected {len(rows)} points  ({label or 'single-seed'})"
          f"{'  + scripted anchors' if anchors else '  (no anchors file)'}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        fh.write(render(rows, label, anchors))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
