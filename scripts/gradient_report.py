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
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.metrics import METRIC_DESCRIPTIONS, summary  # noqa: E402

_SOCIAL = re.compile(r"(other agent|agent [ab]|average|combine|offset|share|pool|"
                     r"both|together|reciprocat|mutual|exchange|trade|each other)", re.I)


def _metrics(path: str) -> dict:
    """Compute one run's dose-response metrics from its transcript."""
    ev = [json.loads(l) for l in open(path) if l.strip()]
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
    return {
        "offset": None,  # filled by caller
        "survivor_rate": sum(surv) / len(surv),
        "cooperation": s["cooperation"]["cooperation_index"] if s["cooperation"]["measurements"] else 0.0,
        "reciprocity": 0.0 if (recip != recip) else recip,  # nan -> 0 (no mutual exchange)
        "messages": sum(1 for e in ev if e["event"] == "message"),
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


# --------------------------------------------------------------------------- #
# SVG line chart (one metric vs offset). Single series -> no legend; title names
# it; endpoint is direct-labelled. Recessive grid, 2px line, 8px markers.       #
# --------------------------------------------------------------------------- #
def _chart(rows, key, *, title, unit, color, ymax=None, hero=False, desc=""):
    """Render one metric-vs-offset line chart (single series) as inline SVG.

    ``desc`` becomes a hover tooltip on the caption explaining the metric."""
    W, H = (720, 300) if hero else (340, 210)
    ml, mr, mt, mb = 46, 18, 34, 34
    xs = [r["offset"] for r in rows]
    ys = [r[key] for r in rows]
    xmax = max(xs) if xs else 500
    ymax = ymax if ymax is not None else (max(ys) * 1.15 if ys and max(ys) > 0 else 1)
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
    # x ticks
    xt = ""
    for xv in xs:
        xx = px(xv)
        xt += f'<text class="xtick" x="{xx:.1f}" y="{H-mb+18:.1f}">{xv:.0f}</text>'
    # regime band shading (hero only): symmetric / soft / medium / hard
    band = ""
    if hero:
        zones = [(0, 60, "symmetric"), (60, 130, "soft"), (130, 230, "medium"), (230, xmax, "hard")]
        for a, b, name in zones:
            x1, x2 = px(a), px(min(b, xmax))
            band += (f'<rect class="zone" x="{x1:.1f}" y="{mt}" width="{x2-x1:.1f}" '
                     f'height="{H-mt-mb}"/>'
                     f'<text class="zone-l" x="{(x1+x2)/2:.1f}" y="{mt+14}">{name}</text>')
    # line + area
    path = " ".join(f"{'M' if i==0 else 'L'}{px(x):.1f},{py(y):.1f}" for i, (x, y) in enumerate(zip(xs, ys)))
    area = (f"M{px(xs[0]):.1f},{py(0):.1f} " +
            " ".join(f"L{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys)) +
            f" L{px(xs[-1]):.1f},{py(0):.1f} Z") if xs else ""
    # markers + hover targets
    dots = ""
    for x, y in zip(xs, ys):
        val = (f"{y:.0%}" if unit == "pct" else (f"{y:.0f}" if ymax >= 4 else f"{y:.2f}"))
        dots += (f'<circle class="mk" cx="{px(x):.1f}" cy="{py(y):.1f}" r="{4.5 if not hero else 5.5}" '
                 f'style="fill:{color}" data-x="{x:.0f}" data-y="{val}"/>')
    # endpoint direct label
    endlab = ""
    if xs:
        yv = ys[-1]
        val = (f"{yv:.0%}" if unit == "pct" else (f"{yv:.0f}" if ymax >= 4 else f"{yv:.2f}"))
        endlab = (f'<text class="endlab" x="{px(xs[-1])-8:.1f}" y="{py(ys[-1])-9:.1f}" '
                  f'style="fill:{color}">{val}</text>')
    cap = (f'<figcaption title="{html.escape(desc)}" style="cursor:help">{title}</figcaption>'
           if desc else f'<figcaption>{title}</figcaption>')
    return f'''<figure class="chart{' hero' if hero else ''}">
      {cap}
      <svg viewBox="0 0 {W} {H}" role="img" aria-label="{title} versus instrument offset">
        {band}{grid}
        <path class="area" d="{area}" style="fill:{color}"/>
        <path class="line" d="{path}" style="stroke:{color}"/>
        {dots}{endlab}
        <text class="axl" x="{ml+(W-ml-mr)/2:.1f}" y="{H-2}">instrument offset  (bias σ)</text>
      </svg>
    </figure>'''


def render(rows: list) -> str:
    """Assemble the full dose-response HTML report."""
    D = METRIC_DESCRIPTIONS
    hero = _chart(rows, "reciprocity", title="Reciprocity of exchange", unit="pct",
                  color="var(--c-recip)", ymax=1.0, hero=True, desc=D["reciprocity"])
    panels = "".join([
        _chart(rows, "survivor_rate", title="Survivor rate", unit="pct", color="var(--c-surv)", ymax=1.0, desc=D["survivor_rate"]),
        _chart(rows, "cooperation", title="Cooperation index", unit="pct", color="var(--c-coop)", ymax=1.0, desc=D["cooperation"]),
        _chart(rows, "messages", title="Messages sent", unit="n", color="var(--c-msg)", desc=D["messages"]),
        _chart(rows, "social_frac", title="Reasoning about the partner", unit="pct", color="var(--c-soc)", ymax=1.0, desc=D["social"]),
    ])
    trows = "".join(
        f'<tr><td>{r["offset"]:.0f}</td><td>{r["survivor_rate"]:.0%}</td>'
        f'<td>{r["cooperation"]:.0%}</td><td>{r["reciprocity"]:.0%}</td>'
        f'<td>{r["messages"]:.0f}</td><td>{r["social_frac"]:.0%}</td>'
        f'<td>{r["welfare"]:.0f}</td></tr>' for r in rows)
    n = len(rows)
    out = _HTML.replace("{{HERO}}", hero).replace("{{PANELS}}", panels)\
               .replace("{{TROWS}}", trows).replace("{{N}}", str(n))
    for token, key in (("{{D_offset}}", "offset"), ("{{D_surv}}", "survivor_rate"),
                       ("{{D_coop}}", "cooperation"), ("{{D_recip}}", "reciprocity"),
                       ("{{D_msg}}", "messages"), ("{{D_soc}}", "social"),
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
  th,td{text-align:right; padding:7px 10px; border-bottom:1px solid var(--border);}
  th:first-child,td:first-child{text-align:left;}
  th{color:var(--muted); font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:.05em;}
  .tip{position:fixed; pointer-events:none; opacity:0; background:var(--ink); color:var(--plane);
    font-family:var(--mono); font-size:11.5px; padding:5px 8px; border-radius:7px; transform:translate(-50%,-140%);
    white-space:nowrap; z-index:9; transition:opacity .08s;}
  .foot{font-size:12.5px; color:var(--muted); margin-top:28px; max-width:64ch;}
</style>
<div class="viz-root"><div class="wrap">
  <p class="eyebrow">Agora · multi-agent LLM · dose–response</p>
  <h1>Cooperation switches on when you make it necessary</h1>
  <p class="stand">Two Qwen-3-32B agents estimate the same hidden number. We dial one knob — an
    <b>instrument offset</b> that a single agent can't cancel alone but that vanishes when both agents
    <b>average their readings</b> — from 0 (solo works fine) to 500 (solo is hopeless), and watch what the
    agents do.</p>
  <p class="meta">{{N}} offsets · Qwen-3-32B×2 · <b>one match (one seed) per point — preliminary</b> · offset σ 0→500 · only the offset varies</p>

  <div class="card hero">{{HERO}}</div>
  <p class="lede">Read the <b>trend, not the individual points</b> — each point here is a <b>single match</b>,
    so the middle of the range is noisy. The robust signal is at the extremes: when solo play is viable (low
    offset) engagement is weak and any sharing is one-sided; pushing toward "solo is impossible" moves the
    exchange toward <b>mutual</b>. Note the far right: at the hardest walls <b>survival collapses</b> — agents
    die before they can establish cooperation, so reciprocity falls back too. A multi-seed sweep (mean ± CI)
    would sharpen this into a clean dose–response; the cleanest three-point contrast is on the
    <b>Compare</b> page (5-game runs: reciprocity 0.28 → 0.45 → 0.97).</p>

  <div class="card"><div class="grid2">{{PANELS}}</div></div>

  <h2>All runs <span style="font-weight:400;text-transform:none;letter-spacing:0">· hover a column for what it means</span></h2>
  <div style="overflow-x:auto"><table>
    <tr>
      <th title="{{D_offset}}" style="cursor:help">offset σ</th>
      <th title="{{D_surv}}" style="cursor:help">survivors</th>
      <th title="{{D_coop}}" style="cursor:help">cooperation</th>
      <th title="{{D_recip}}" style="cursor:help">reciprocity</th>
      <th title="{{D_msg}}" style="cursor:help">messages</th>
      <th title="{{D_soc}}" style="cursor:help">reasons re partner</th>
      <th title="{{D_welf}}" style="cursor:help">welfare</th></tr>
    {{TROWS}}
  </table></div>

  <p class="foot"><b>Caveat: one match (one seed) per point</b>, so per-point values carry real sampling
    noise — the curve should firm up with several seeds averaged per offset. "Offset" is a fixed per-agent
    bias added to every reading that round; the two agents' offsets sum to zero, so averaging both readings
    recovers the true value while any single agent — even measuring repeatedly — stays stuck at its own
    offset. Only bias σ varies across runs; prior, noise, budget, survival cost and horizon are held fixed.</p>
</div></div>
<div class="tip" id="tip"></div>
<script>
  const tip=document.getElementById('tip');
  for(const m of document.querySelectorAll('.mk')){
    m.addEventListener('mouseenter',e=>{tip.textContent='offset '+m.dataset.x+'  →  '+m.dataset.y;
      const r=m.getBoundingClientRect(); tip.style.left=(r.left+r.width/2)+'px'; tip.style.top=r.top+'px'; tip.style.opacity=1;});
    m.addEventListener('mouseleave',()=>tip.style.opacity=0);
  }
</script>
"""


def main(argv=None):
    """CLI: build the dose-response report from the gradient runs."""
    argv = argv if argv is not None else sys.argv[1:]
    pattern = "runs/qwen/grad_b*.jsonl"
    out = "runs/qwen/gradient_report.html"
    if "-o" in argv:
        out = argv[argv.index("-o") + 1]
    pos = [a for a in argv if not a.startswith("-") and a != out]
    if pos:
        pattern = pos[0]
    rows = collect(pattern)
    print(f"collected {len(rows)} runs: offsets {[int(r['offset']) for r in rows]}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        fh.write(render(rows))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
