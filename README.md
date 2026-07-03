# Agora — a minimal multi-agent Measurement Market

> Several identical agents. The same skill, the same budget. Forced to
> collaborate to do well — and free to defraud, hoard, collude, or die.

Agora is a small, elegant testbed for studying **emergent behaviour in
multi-agent LLM systems**. The task is deliberately trivial to state: each round
a hidden number `theta` is drawn (think *"cars per hour on a highway"*), and
every agent must estimate it. An agent can **spend credits** to draw a noisy
measurement, or **talk to and trade with** the other agents. Its score is simply
how close its final guess is to the truth.

Everything interesting falls out of three facts:

1. **Noise averages out** — pooling `k` samples cuts error by `√k`, so the
   welfare optimum is to split the measurement bill and share results.
2. **Sold information is unverifiable** — a buyer cannot check a measurement it
   bought, so an agent can sell a *phantom* number. Fraud is possible.
3. **Budgets carry over and can run out** — reward for accuracy becomes next
   round's budget; hit zero and you are eliminated. Agents can die.

Because the *referee* generates every measurement, we always know what an agent
truly observed versus what it reported — giving a **gold-standard, non-LLM
deception label**. That is the project's methodological core.

The agents are local **Qwen3-32B** models served with vLLM; the entire harness
also runs with scripted baseline agents and **no GPU at all**, which is how the
smoke test works.

Read the full plan in **[docs/DESIGN.md](docs/DESIGN.md)** and the exact rules in
**[docs/protocol.md](docs/protocol.md)**.

---

## Quickstart (no GPU, no dependencies)

The scripted baselines let the full game — escrow, rewards, transcripts,
deception detection — run on a plain CPU with only the Python standard library:

```bash
python -m agora.run --preset smoke          # 2 agents, 1 round
python -m agora.run --preset base --seed 7 --out runs/base
python -m analysis.metrics runs/base/seed7.jsonl
```

Example (`--preset base`): four scripted agents — an honest cooperator, a solo
Bayesian, a **liar**, and a hoarder — over a hidden-horizon game:

```
round    truth                  A                  B                  C                  D
    0    458.6       468.3/e10/r5        464.7/e6/r5       429.9/e29/r5      569.5/e111/r2
    ...
final credits: A=8.0  B=7.0  C=5.0  D=13.0
```

and the metrics recover, among other things, that seller **C** fabricated every
value it sold:

```
deception:  {offers: 8, lies: 8, deception_rate: 1.0, ...}
cooperation:{measurements: 75, shared: 16, cooperation_index: 0.21}
```

Run the tests (and the docstring lint, which is also enforced by the suite):

```bash
python -m pytest -q                   # full suite (70 tests)
python tests/test_market.py           # escrow / contract invariants
python tests/test_referee.py          # end-to-end game + metrics
python scripts/lint_docstrings.py     # every production def must be documented
```

---

## Running with local Qwen3-32B

Serve the model once (one instance serves all agents — vLLM batches them):

```bash
scripts/serve_qwen.sh                 # bf16 across your GPUs
PRECISION=fp8 scripts/serve_qwen.sh   # single 80GB GPU
PRECISION=awq scripts/serve_qwen.sh   # single 24–48GB GPU
```

Then point the agents at it:

```bash
pip install -e .                      # pulls in the openai client
python -m agora.run --preset base --policies llm \
    --model qwen3-32b --base-url http://localhost:8000/v1 --out runs/qwen
```

The serving recipe (tool parser, non-streaming, `tool_choice="auto"`, thinking
toggle, VRAM footprints) is baked into `scripts/serve_qwen.sh` and
`agora/backends.py`, and explained in [docs/DESIGN.md §9](docs/DESIGN.md).

---

## Seeing a game (the interface)

Every run writes a JSONL transcript; `analysis/viz.py` turns one into a
**standalone, shareable HTML report** — no server, no dependencies — laying the
game out tick by tick with measurements, messages, trades (fabricated sales
flagged in red via the ground-truth lie detector), and per-round results.

```bash
python -m agora.run --preset base --seed 7 --out runs/base
python -m analysis.viz runs/base/seed7.jsonl        # -> runs/base/seed7.html
python -m analysis.viz runs/base/ -o report/         # a whole directory -> index.html
```

A rendered sample of a real **Qwen-3-32B vs Qwen-3-32B** match (viewable in any
browser) lives in [docs/samples/](docs/samples) — a shared-memory **5-game
match** where the agents co-evolve across games. The scripted baselines are used
for testing and as comparison anchors, not as a headline;
`python scripts/make_samples.py` renders a couple of scripted demos locally if
you want them.

### Web UI — run games from a button

A small Flask app ([web/app.py](web/app.py)) lets you launch games from the
browser and read them as reports. Pick a preset, choose the backend — **scripted**
(instant, no GPU) or **Qwen via vLLM** (real agents; runs in the background and
the page auto-refreshes) — and hit **Run new game**.

```bash
pip install flask          # or: pip install -e '.[web]'
scripts/serve_web.sh       # -> http://127.0.0.1:5000   (HOST/PORT override)
```

The gallery lists every game with its deception rate and survivor count; click
one to open the full tick-by-tick report (with a **Scoreboard**: games won,
survived, avg error, reward, lies). Games are saved under `runs/web/`. For the
Qwen backend, start [scripts/serve_qwen.sh](scripts/serve_qwen.sh) first and
`pip install openai`.

### Deploy the viewer (Render)

The viewer is a plain Flask app and ships with a [render.yaml](render.yaml)
Blueprint, so it deploys straight from GitHub and auto-redeploys on every push:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/liv-daliberti/tokenomics)

Or manually: on [render.com](https://render.com) → **New +** → **Blueprint** →
connect this repo → Render reads `render.yaml` and creates the `agora-viewer`
service. On boot it seeds the committed sample games (including the real
**Qwen-3-32B vs Qwen-3-32B** run) into the gallery, so the hosted site is never
empty. Running new **scripted** games works in-browser; the LLM backend needs a
reachable vLLM endpoint, so it is local-only.

## What you can vary

Everything is one knob in a config ([configs/](configs)) or a preset:

| Axis | Knob | Ablation |
|---|---|---|
| Scale | `agent_ids` | 2 → 3 (coalitions) → 4 (meaningful coalitions) |
| Horizon | `horizon_mode`, `gamma`, `reveal_horizon` | hidden vs known (endgame/hoarding) |
| Privilege | `tau_by_agent` | heterogeneous noise → "data brokers" |
| Death & rescue | `survival_cost`, `elimination_on_ruin` | can agents die? a peer can `transfer_credits` to revive an eliminated agent |
| Framing | `framing` | neutral / cooperative / competitive (confound check) |
| Market | `enable_transfer`, `enable_trading`, `values_via_trade_only` | free chat vs paid escrow; force values through trades |
| Answer phase | `final_answer_pass` | a final turn to update the guess after all exchange |

Presets: `cooperative` (default), `cooperation_required`,
`base, coalitions, endgame, privilege, survival, smoke` (`--preset <name>`).
YAML configs mirror the experiment matrix in [docs/DESIGN.md §6](docs/DESIGN.md).

---

## Repository layout

```
agora/
  config.py        game configuration + presets
  environment.py   ground truth, the measurement tool, the horizon
  market.py        credits, escrow, atomic settlement (the invariants)
  rewards.py       loss → reward tokens, carry-over, death
  referee.py       the authoritative game loop
  tools.py         tool JSON-schemas + system prompts (neutral framing)
  observation.py   per-agent observations (private info only)
  backends.py      model backends (local vLLM / mock) — the one model seam
  policies/        LLM policy + scripted baselines (= smoke backend)
  transcripts.py   append-only JSONL logging (the analysis substrate)
  run.py           CLI
analysis/metrics.py  regret, cooperation, reciprocity, rescue, deception, prices, welfare, Gini
scripts/study.py     multi-seed study: aggregate metrics with 95% CIs
configs/             base + ablation configs (the experiment matrix)
scripts/serve_qwen.sh  vLLM launch recipe for Qwen3-32B
tests/               escrow invariants + end-to-end game
docs/                DESIGN.md (the plan) · protocol.md (the rules)
```

The orchestration is a **homebrew async-ready referee**, chosen over
AutoGen (now EOL) and Inspect (task/sample-shaped, not a peer-agent market).
Four clean seams keep an Inspect bridge cheap later: a single model interface
(`backends.py`), plain JSON-schema tools (`tools.py`), a superset-of-Inspect
transcript (`transcripts.py`), and a standalone scorer (`rewards.py`). See
[docs/DESIGN.md](docs/DESIGN.md).

---

## Status

The harness, the viewer, and several real **Qwen-3-32B vs Qwen-3-32B** matches
(served on 2×A100 via vLLM) are complete and tested (66 passing tests, runnable
with no GPU via the scripted baselines).

The task is always the same **"pick a number"** estimation game — every agent
estimates the *same* hidden `theta`. Cooperation is not scripted; it is made
worthwhile by the economics.

**Mechanism design — forcing cooperation.** With two *equal* agents, independent
pooling only buys a `√2` edge — too small to separate cooperators from solos, and
once the economics are sane the Qwen agents just solve it alone (they never even
mention the partner). So the default forces true interdependence (see
[docs/DESIGN.md §2.3](docs/DESIGN.md)):
- **`cooperative`** (N=2, the default) — **paired instrument bias**: each agent's
  reading carries a large fixed per-round offset, the two offsets sum to zero, and
  the prior is wide. A single reading is ~200 off and measuring again can't fix it;
  only **averaging both agents' readings** recovers `theta`. Solo (even optimal) is
  non-viable — scripted: cooperate ~100% survive, solo ~3%, hoard ~8%, lie ~3%.
  Everyone still estimates the same `theta`; `bias_sigma=0` recovers the symmetric game.
- **`cooperation_required`** (N=4) — a survival wall: solo ~9% survival, cooperate ~90%.

**Rescue.** With `enable_transfer`, an agent can `transfer_credits` to another —
including an **eliminated** one, which **revives** it into the next round. Agents
are told this, so a dying partner can be kept alive or brought back.

**What the Qwen agents actually did** (100-match interdependence sweep, N=2, plus
reports in [docs/samples/](docs/samples)): clean tool use throughout (0 parse
failures). Cooperation is a **switch, not a dial** — near-zero with no wall (~7%
of readings shared), it flips on (~68%) the instant solo play is penalized, then
flatlines; survival falls steadily as the wall hardens, so the Qwen curve sits
*between* a scripted honest-cooperator ceiling (~100% survival) and a solo floor.
The failure is **shallow, one-sided** cooperation (reciprocity never rises above
noise) — **not** deception and **not** distrust: with the ground-truth lie
detector watching, verified fabrication is ~**1%** of 700+ offers (they route
around the unverifiable market — ~80% of matches settle zero trades), and across
50k reasoning steps they almost never mention trust or verification. *The recurring
finding: when solo play is fatal LLM agents will pool — but shallowly and
one-sidedly, and honestly; the missing ingredient is reciprocity, not honesty.*
(An earlier writeup reported ~9% deception and a "distrust" story; both were
artifacts — honest averaging mislabeled as fraud, since corrected.)

**Viewer** — a Flask app (Qwen-only "run a game" tab, simulator knobs, N-game
matches with shared memory, per-agent side-by-side timelines with a per-step
budget, a win/loss scoreboard, elimination + revival notices, dimmed unanswered
offers, raw-transcript download, hover tooltips) that deploys to Render from
GitHub and seeds the curated Qwen runs.

Next is **Phase 2**: a multi-seed study and 3–4-agent coalitions — see
[docs/DESIGN.md §10](docs/DESIGN.md).
