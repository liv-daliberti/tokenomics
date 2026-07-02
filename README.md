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

Run the tests:

```bash
python tests/test_market.py     # escrow / contract invariants
python tests/test_referee.py    # end-to-end game + metrics
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

Two rendered samples (real engine output, viewable in any browser) live in
[docs/samples/](docs/samples): **cooperation_and_fraud** (a liar sells a phantom
value every round) and **broker_and_death** (heterogeneous privilege + a
survival cost drives three of four agents to ruin). Regenerate with
`python scripts/make_samples.py`.

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
| Death | `survival_cost`, `elimination_on_ruin` | can agents die? |
| Framing | `framing` | neutral / cooperative / competitive (confound check) |
| Market | `enable_transfer`, `enable_trading` | free chat vs paid escrow |

Presets: `smoke, base, coalitions, endgame, privilege, survival`
(`--preset <name>`). YAML configs mirror the experiment matrix in
[docs/DESIGN.md §6](docs/DESIGN.md).

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
analysis/metrics.py  regret, cooperation, verifiable deception, welfare, Gini
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

**Phase 0 (harness)** and **Phase 1 (Qwen smoke)** are complete and tested.

- **Harness** — the game loop, escrow, rewards, transcripts, scripted baselines,
  and metrics run with no GPU; the suite is green.
- **Real Qwen run** — served `Qwen/Qwen3-32B` (bf16, 2×A100) on vLLM and ran a
  2-agent cooperative *match* end to end: **clean tool use (0 parse failures),
  agents reasoned and one shared its readings to pool, and the tight budget
  produced real eliminations.** Estimation was weak vs. the pooling oracle and
  no trades were used. See the report:
  [docs/samples/qwen3-32b_cooperative.html](docs/samples/qwen3-32b_cooperative.html).
- **Viewer** — a Flask app (run-a-game button, simulator knobs, N-game matches
  with shared agent memory, per-agent reasoning, a win/loss scoreboard, raw
  transcript download) that deploys to Render from GitHub.

Next is **Phase 2**: a multi-seed baseline study and scaling to 3–4 agents — see
the roadmap in [docs/DESIGN.md §10](docs/DESIGN.md).
