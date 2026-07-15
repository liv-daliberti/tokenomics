"""Drive the full GPT-5.4 replication program against a hosted endpoint.

Replicates the complete Qwen study — the prompted interdependence gradient,
the de-confounded control, the trust probe — plus the NEW cross-game memory
ablation (full context vs markdown notes), writing everything under a
separate runs dir (default runs/gpt54) so the original Qwen data is never
touched. Cloud calls need no GPU: this runs anywhere with network access.

    # 0. free end-to-end validation against the local stub endpoint
    python scripts/gpt54_program.py --stage pilot --stub

    # 1. the money pilot: 3 real matches, prints measured tokens and a
    #    projected cost for every remaining stage — decide BEFORE committing
    AZURE_OPENAI_API_KEY=... python scripts/gpt54_program.py --stage pilot \\
        --price-in 1.25 --price-out 10

    # 2. the full program (resumable; re-running only backfills)
    AZURE_OPENAI_API_KEY=... python scripts/gpt54_program.py --stage all --jobs 3

Stages: pilot (3 matches), grad (140), deconf (100), probe (10), mem (30).
Every match is one scripts/qwen_match.py subprocess; a match whose output
already contains match_end is skipped, so the program is resumable and a
crashed run costs only its own partial transcript.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AZURE_URL = "https://liv.services.ai.azure.com/openai/v1"
STUB_URL = "http://127.0.0.1:8111/v1"

# Every protocol variable qwen_match.py honors. These are SCRUBBED from the
# child environment and re-set per run, so a stray `export FRAMING=neutral`
# in the launching shell can never silently change a paid stage's condition.
# (Credential vars — AZURE_OPENAI_API_KEY etc. — are deliberately NOT listed.)
PROTOCOL_VARS = ["MODEL", "BASE_URL", "PROVIDER", "PRESET", "GAMES", "ROUNDS",
                 "MAXTICKS", "SEED", "OUT", "BIAS_SIGMA", "PRIOR_SIGMA", "TAU",
                 "SURVIVAL_COST", "FRAMING", "STRATEGY_HINT", "MEMORY",
                 "POLICIES", "VALUES_VIA_TRADE_ONLY", "REQUIRE_PAID_TRADES",
                 "MIN_TRADE_PRICE", "STARTING_CREDITS"]

# The exact grids of the Qwen study (from runs/qwen + the SLURM scripts).
GRAD_OFFSETS = [0, 10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 500]
DECONF_OFFSETS = [0, 50, 100, 150, 200, 250, 300, 350, 400, 500]
MEM_OFFSETS = [0, 100, 300]          # no wall / soft wall / hard wall
GAME_SHAPE = {"PRESET": "cooperative", "GAMES": "10", "ROUNDS": "5", "MAXTICKS": "4"}
TRUST_SEEDS = 2                      # pilot size for the knowing-doing-gap probe
TRUST_GAMES = 5                      # games per match (does trust build across games?)
# The price-sweep probe: hard wall (offset 200, where an HONEST reading is
# worth ~6 credits by the cost-utility model), a comfortable budget so refusal
# is a choice not an affordability limit, and a min-trade-price dial swept
# across / past that value. Does the buy-rate on the liar's fake (worth ~0)
# fall as it gets pricier, and does the model keep buying the honest reading up
# to its value and no further?
PRICE_OFFSET = 200
PRICE_LEVELS = [0.5, 2, 8, 32]       # credits; measure_cost=1, honest value ~6
PRICE_START_CREDITS = 60             # budget so even the top price is affordable
# The difficulty x price x partner grid for the cost-error scatter: two
# difficulties, three information prices, three partner honesty regimes
# (all-lies / no-lies / mixed). Error is read from the LLM's own estimates.
GRID_OFFSETS = [0, 200]              # easy / hard
GRID_PRICES = [1, 8, 32]            # cheap / medium / expensive
GRID_PARTNERS = ["liar", "honest_cooperator", "mixed_liar"]
GRID_SEEDS = 2

USAGE_RE = re.compile(r"\[qwen_match\] USAGE (\{.*\})")


def build_matrix(stage: str, seeds: int, mem_seeds: int,
                 market: str = "paid") -> list:
    """The (name, env-overrides) list for a stage. Names double as filenames.

    ``market="paid"`` (the program default) runs EVERY stage under the strict
    market regime: numbers (digits and number words) censored from chat, and
    trade prices strictly greater than 0 — values can only move through paid
    trades. This is a deliberate protocol change from the Qwen study, where
    free chat could carry values; ``market="open"`` recovers the Qwen-identical
    rules and suffixes every filename with _open so the two regimes can never
    mix in one aggregate."""
    runs = []
    if stage in ("grad", "all"):
        for off in GRAD_OFFSETS:
            for s in range(seeds):
                runs.append((f"grad_b{off}_s{s}",
                             {"BIAS_SIGMA": str(off), "SEED": str(s), **GAME_SHAPE}))
    if stage in ("deconf", "all"):
        for off in DECONF_OFFSETS:
            for s in range(seeds):
                runs.append((f"deconf_b{off}_s{s}",
                             {"BIAS_SIGMA": str(off), "SEED": str(s),
                              "FRAMING": "neutral", "STRATEGY_HINT": "0", **GAME_SHAPE}))
    if stage in ("probe", "all"):
        for bot in ("liar", "honest_cooperator"):
            for s in range(min(5, seeds)):
                runs.append((f"probe_{bot}_s{s}",
                             {"PRESET": "probe_trust", "GAMES": "10",
                              "POLICIES": f"llm,{bot}", "SEED": str(s)}))
    if stage == "trust":
        # The knowing-doing-gap probe: one LLM seat vs a scripted bot whose
        # honesty is ground truth, at two calibrated difficulties (offset 0 =
        # solo survives, buying is optional; offset 200 = solo dies, the LLM
        # MUST buy the partner's readings to live). liar = ground-truth
        # positives, honest_cooperator = negatives (both priced identically
        # under the paid market, so accept/reject turns on trust not price).
        for bot in ("liar", "honest_cooperator"):
            for off, lvl in ((0, "easy"), (200, "hard")):
                for s in range(TRUST_SEEDS):
                    runs.append((f"trust_{bot}_{lvl}_b{off}_s{s}",
                                 {"PRESET": "probe_trust", "GAMES": str(TRUST_GAMES),
                                  "POLICIES": f"llm,{bot}", "BIAS_SIGMA": str(off),
                                  "SEED": str(s)}))
    if stage == "price":
        # As the lie gets more expensive: sweep the min-trade-price floor at the
        # hard wall vs the liar (does the buy-rate on a worthless fake fall as it
        # costs more?) and the honest partner (does it keep buying real info up
        # to its ~6-credit value and stop above?).
        for bot in ("liar", "honest_cooperator"):
            for price in PRICE_LEVELS:
                for s in range(TRUST_SEEDS):
                    tag = ("p" + str(price).replace(".", "_"))
                    runs.append((f"price_{bot}_{tag}_s{s}",
                                 {"PRESET": "probe_trust", "GAMES": str(TRUST_GAMES),
                                  "POLICIES": f"llm,{bot}", "BIAS_SIGMA": str(PRICE_OFFSET),
                                  "MIN_TRADE_PRICE": str(price),
                                  "STARTING_CREDITS": str(PRICE_START_CREDITS),
                                  "SEED": str(s)}))
    if stage == "grid":
        # difficulty x price x partner-honesty, for the cost-error scatter.
        for partner in GRID_PARTNERS:
            for off in GRID_OFFSETS:
                for price in GRID_PRICES:
                    for s in range(GRID_SEEDS):
                        tag = "p" + str(price).replace(".", "_")
                        runs.append((f"grid_{partner}_b{off}_{tag}_s{s}",
                                     {"PRESET": "probe_trust", "GAMES": str(TRUST_GAMES),
                                      "POLICIES": f"llm,{partner}", "BIAS_SIGMA": str(off),
                                      "MIN_TRADE_PRICE": str(price),
                                      "STARTING_CREDITS": str(PRICE_START_CREDITS),
                                      "SEED": str(s)}))
    if stage in ("mem", "all"):
        # The context-vs-markdown ablation, on the PROMPTED condition (where
        # there is cross-game behaviour worth remembering). The context arm is
        # protocol-identical to the gradient at these offsets/seeds, so it
        # REUSES the grad_b* names: if the grad stage already ran them, resume
        # skips them — 15 matches never paid for twice. memory_report.py reads
        # grad_b* files as the context arm for the same reason.
        for off in MEM_OFFSETS:
            for s in range(mem_seeds):
                runs.append((f"grad_b{off}_s{s}",
                             {"BIAS_SIGMA": str(off), "SEED": str(s), **GAME_SHAPE}))
                runs.append((f"mem_markdown_b{off}_s{s}",
                             {"BIAS_SIGMA": str(off), "SEED": str(s),
                              "MEMORY": "markdown", **GAME_SHAPE}))
    if stage == "pilot":
        # Three real matches sized like the real thing: the cheapest and the
        # costliest full-context shapes, plus one markdown-memory match, so
        # every stage's cost can be projected before money is committed.
        runs = [
            ("pilot_grad_b0_s900", {"BIAS_SIGMA": "0", "SEED": "900", **GAME_SHAPE}),
            ("pilot_grad_b300_s900", {"BIAS_SIGMA": "300", "SEED": "900", **GAME_SHAPE}),
            ("pilot_mem_markdown_b300_s900",
             {"BIAS_SIGMA": "300", "SEED": "900", "MEMORY": "markdown", **GAME_SHAPE}),
        ]
    # "all" emits the mem context arm AND the gradient; de-duplicate by name
    # (first occurrence wins) so no match is ever launched twice in one call.
    seen, unique = set(), []
    suffix = "" if market == "paid" else f"_{market}"
    extra = ({"VALUES_VIA_TRADE_ONLY": "1", "REQUIRE_PAID_TRADES": "1"}
             if market == "paid" else {})
    for name, ov in runs:
        if name not in seen:
            seen.add(name)
            unique.append((name + suffix, {**ov, **extra}))
    return unique


def is_complete(out_base: str) -> bool:
    """A match counts as done only if its transcript reached match_end."""
    path = out_base + ".jsonl"
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as fh:
            fh.seek(max(0, os.path.getsize(path) - 4096))
            return b'"match_end"' in fh.read()
    except OSError:
        return False


def _save_record(args, manifest: dict, lock, name: str, rec: dict) -> None:
    """Merge one record into the manifest and write it atomically.

    Re-reads the on-disk manifest first so two driver processes sharing a runs
    dir (e.g. grad and mem in separate terminals) never clobber each other's
    records, and writes temp+rename so a kill mid-dump can't truncate it."""
    path = os.path.join(args.runs_dir, "manifest.json")
    with lock:
        try:
            on_disk = json.load(open(path))
        except (OSError, ValueError):
            on_disk = {}
        on_disk.update(manifest)
        # never let a bare "skipped" note erase a real record's usage/seconds
        old = on_disk.get(name)
        if not (rec.get("usage") is None and old and old.get("usage") is not None):
            on_disk[name] = rec
        manifest.clear()
        manifest.update(on_disk)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(on_disk, fh, indent=1, sort_keys=True)
        os.replace(tmp, path)


def run_one(name: str, overrides: dict, args, manifest: dict, lock) -> None:
    """Run one match as a qwen_match.py subprocess; record status + usage."""
    out_base = os.path.join(args.runs_dir, name)
    if is_complete(out_base):
        _save_record(args, manifest, lock, name, {"status": "skipped (complete)"})
        print(f"[program] skip {name} (already complete)", flush=True)
        return
    # Scrub every protocol var, then set exactly this run's condition — a
    # stray exported FRAMING/MEMORY/etc. must never leak into a paid match.
    env = {k: v for k, v in os.environ.items() if k not in PROTOCOL_VARS}
    env.update(MODEL=args.model, BASE_URL=args.base_url, PROVIDER="openai",
               OUT=out_base, **overrides)
    if args.stub:
        env["API_KEY"] = "test-key-123"
    t0 = time.time()
    print(f"[program] start {name}", flush=True)
    with open(out_base + ".log", "w") as logf:
        try:
            proc = subprocess.run(
                [sys.executable, os.path.join(REPO, "scripts", "qwen_match.py")],
                cwd=REPO, env=env, stdout=logf, stderr=subprocess.STDOUT,
                timeout=args.match_timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    usage = None
    try:
        m = USAGE_RE.findall(open(out_base + ".log").read())
        usage = json.loads(m[-1]) if m else None
    except (OSError, ValueError):
        pass
    rec = {"status": "done" if rc == 0 and is_complete(out_base) else f"error rc={rc}",
           "seconds": round(time.time() - t0, 1), "usage": usage}
    _save_record(args, manifest, lock, name, rec)
    print(f"[program] end   {name}: {rec['status']} ({rec['seconds']}s) "
          f"usage={usage}", flush=True)


def project_costs(manifest: dict, args) -> None:
    """From pilot usage, project tokens and (if prices given) dollars per stage."""
    suffix = "" if args.market == "paid" else f"_{args.market}"
    full = [manifest[n]["usage"] for n in
            (f"pilot_grad_b0_s900{suffix}", f"pilot_grad_b300_s900{suffix}")
            if manifest.get(n, {}).get("usage")]
    md = manifest.get(f"pilot_mem_markdown_b300_s900{suffix}", {}).get("usage")
    if not full:
        # fall back to pilots from the other market regime, clearly labelled —
        # token shape is similar, but rerun the pilot under THIS market to pin it
        full = [r["usage"] for n, r in sorted(manifest.items())
                if n.startswith("pilot_grad") and r.get("usage")]
        md = md or next((r["usage"] for n, r in sorted(manifest.items())
                         if n.startswith("pilot_mem_markdown") and r.get("usage")), None)
        if full:
            print(f"[program] NOTE: no pilot for market={args.market}; "
                  "projecting from another market's pilot (approximate)")
    if not full:
        print("[program] no pilot usage recorded — cannot project costs")
        return
    avg = {k: sum(u[k] for u in full) / len(full) for k in full[0]}
    # probe_trust has ONE LLM seat (vs two in the pilot matches), so weight it
    # at half; counts come from build_matrix so --seeds/--mem-seeds are honored.
    half = {k: v / 2 for k, v in avg.items()}
    n_grad = len(build_matrix("grad", args.seeds, args.mem_seeds, args.market))
    n_dec = len(build_matrix("deconf", args.seeds, args.mem_seeds, args.market))
    n_probe = len(build_matrix("probe", args.seeds, args.mem_seeds, args.market))
    n_md = sum(1 for n, _ in build_matrix("mem", args.seeds, args.mem_seeds, args.market)
               if n.startswith("mem_markdown"))
    stages = [("grad (incl. mem-context)", n_grad, avg), ("deconf", n_dec, avg),
              ("probe (1 LLM seat)", n_probe, half),
              ("mem: markdown arm", n_md, md or avg)]
    print("\n===== COST PROJECTION (from pilot) =====")
    print(f"pilot full-context match avg: {avg['prompt_tokens']:,.0f} in / "
          f"{avg['completion_tokens']:,.0f} out tokens over {avg['calls']:.0f} calls")
    if md:
        print(f"pilot markdown match:         {md['prompt_tokens']:,.0f} in / "
              f"{md['completion_tokens']:,.0f} out tokens")
    total_usd = 0.0
    for label, n, u in stages:
        tin, tout = n * u["prompt_tokens"], n * u["completion_tokens"]
        line = f"  {label:<26} {n:>4} matches  ~{tin/1e6:8.1f}M in / {tout/1e6:7.1f}M out"
        if args.price_in is not None and args.price_out is not None:
            usd = tin / 1e6 * args.price_in + tout / 1e6 * args.price_out
            total_usd += usd
            line += f"  ~${usd:,.0f}"
        print(line)
    if total_usd:
        print(f"  {'TOTAL (all stages)':<26} {'':>4}          "
              f"{'':>26}  ~${total_usd:,.0f}")
    print("Projection assumes every match looks like the pilot; low-offset "
          "prompted matches ran LONGEST on Qwen, so treat this as ±2x.\n")


def main(argv=None) -> None:
    """Parse args, guard the key, and run the requested stage of the program."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True,
                    choices=["pilot", "grad", "deconf", "probe", "mem", "trust",
                             "price", "grid", "all"])
    ap.add_argument("--runs-dir", default=os.path.join(REPO, "runs", "gpt54"))
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--base-url", default=AZURE_URL)
    ap.add_argument("--stub", action="store_true",
                    help="run against the local stub endpoint (free pipeline check)")
    ap.add_argument("--market", choices=["paid", "open"], default="paid",
                    help="paid (default): numbers censored from chat AND trade "
                         "prices must be > 0 — values move only through paid "
                         "trades. open: Qwen-identical rules (files suffixed _open)")
    ap.add_argument("--jobs", type=int, default=2,
                    help="matches run concurrently (mind your rate limits)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--mem-seeds", type=int, default=5)
    ap.add_argument("--match-timeout", type=int, default=4 * 3600)
    ap.add_argument("--price-in", type=float, default=None,
                    help="$ per 1M input tokens, for the pilot's cost projection")
    ap.add_argument("--price-out", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the run matrix and exit")
    args = ap.parse_args(argv)
    default_dir = os.path.join(REPO, "runs", "gpt54")
    if args.stub:
        args.base_url = STUB_URL
        # Stub transcripts look complete, so they must never share a dir with
        # paid runs — resume would silently skip the real matches.
        if os.path.abspath(args.runs_dir) == default_dir:
            args.runs_dir = os.path.join(REPO, "runs", "gpt54-stub")
    # The Qwen study uses the same filenames; pointing the driver there would
    # overwrite partial originals and skip the rest. Refuse outright.
    if os.path.basename(os.path.normpath(args.runs_dir)) == "qwen":
        raise SystemExit("refusing --runs-dir that targets the original Qwen "
                         "data; use a fresh directory (default runs/gpt54)")

    runs = build_matrix(args.stage, args.seeds, args.mem_seeds, args.market)
    print(f"[program] stage={args.stage} market={args.market} matches={len(runs)} "
          f"jobs={args.jobs} model={args.model} url={args.base_url} "
          f"-> {args.runs_dir}", flush=True)
    if args.dry_run:
        for name, ov in runs:
            print(f"  {name}: {ov}")
        return

    if not args.stub and not (os.environ.get("API_KEY")
                              or os.environ.get("AGORA_API_KEY")
                              or os.environ.get("AZURE_OPENAI_API_KEY")
                              or os.environ.get("OPENAI_API_KEY")):
        raise SystemExit(
            "No API key found. Export AZURE_OPENAI_API_KEY (or API_KEY) first — "
            "this program makes PAID model calls. Use --stub for a free "
            "pipeline check, and run --stage pilot before anything larger.")

    os.makedirs(args.runs_dir, exist_ok=True)
    manifest_path = os.path.join(args.runs_dir, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            manifest = json.load(open(manifest_path))
        except ValueError:
            manifest = {}
    lock = threading.Lock()
    ex = ThreadPoolExecutor(max_workers=args.jobs)
    futs = [ex.submit(run_one, name, ov, args, manifest, lock)
            for name, ov in runs]
    try:
        for f in futs:
            f.result()
    except KeyboardInterrupt:
        # Ctrl-C must stop SPENDING: drop everything still queued; matches
        # already in flight finish their current subprocess and are recorded.
        print("[program] interrupted — cancelling queued matches", flush=True)
        ex.shutdown(wait=True, cancel_futures=True)
        raise
    ex.shutdown(wait=True)

    done = sum(1 for name, _ in runs
               if str(manifest.get(name, {}).get("status", "")).startswith(("done", "skipped")))
    print(f"[program] finished: {done}/{len(runs)} complete "
          f"(manifest: {manifest_path})", flush=True)
    if args.stage == "pilot":
        project_costs(manifest, args)


if __name__ == "__main__":
    main()
