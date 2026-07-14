"""Command-line entry point.

Examples
--------
Zero-dependency smoke test (scripted agents, no model server, runs anywhere):
    python -m agora.run --preset smoke

A mixed baseline game with transcripts written to disk:
    python -m agora.run --preset base --policies honest_cooperator,liar,hoarder,bayesian_solo \\
        --out runs/base

Real Qwen agents against a local vLLM endpoint (see scripts/serve_qwen.sh):
    python -m agora.run --preset base --policies llm \\
        --model qwen3-32b --base-url http://localhost:8000/v1 --out runs/qwen

Replicate over many seeds (the unit of analysis is the game seed):
    python -m agora.run --preset base --seeds 30 --out runs/base
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

from .config import PRESETS, GameConfig, load_config
from .policies import REGISTRY, LLMPolicy
from .referee import GameResult, Referee
from .transcripts import Transcript


def build_policies(cfg: GameConfig, spec: str, model: str, base_url: str,
                   n_games: int = 1, api_key: str = None,
                   provider: str = None) -> Dict[str, object]:
    """Build the per-agent policy map from a spec, cycled over the agents. Tokens
    are 'llm' (an LLMPolicy on an OpenAI-compatible endpoint — local vLLM or a
    hosted Azure/OpenAI model, see ``OpenAIBackend``) or a scripted baseline name.
    A MIXED spec puts an LLM seat next to a ground-truth bot — e.g. 'llm,liar'
    pits one LLM agent against a scripted liar (the D1/D2 probe: does the LLM
    discount a proven liar, and does a judge recover the referee's lie label?).
    ``n_games`` is passed to LLM seats so the system prompt announces the real
    match length; ``api_key``/``provider`` are handed to the backend."""
    ids = cfg.agent_ids
    names = [n.strip() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n != "llm" and n not in REGISTRY]
    if unknown:
        raise SystemExit(f"unknown scripted policies: {unknown}; choose from {sorted(REGISTRY)} (or 'llm')")
    backend = None
    if "llm" in names:
        from .backends import OpenAIBackend
        backend = OpenAIBackend(model=model, base_url=base_url,
                                api_key=api_key, provider=provider)

    def _make(name: str, aid: str):
        """One agent's policy: an LLM seat or a named scripted baseline."""
        if name == "llm":
            return LLMPolicy(backend, cfg, aid, [p for p in ids if p != aid], n_games=n_games)
        return REGISTRY[name](cfg, aid, ids)

    return {aid: _make(names[i % len(names)], aid) for i, aid in enumerate(ids)}


def summarize(result: GameResult, policy_spec: str) -> None:
    """Print a compact per-round table (truth, each agent's estimate/error/reward), final credits, and a data-quality + deception/cooperation line."""
    cfg = result.config
    print(f"\n=== game seed={cfg.seed}  agents={cfg.agent_ids}  "
          f"rounds={len(result.rounds)}  policies=[{policy_spec}] ===")
    hdr = f"{'round':>5} {'truth':>8} " + " ".join(f"{a:>18}" for a in cfg.agent_ids)
    print(hdr)
    for rr in result.rounds:
        cells = []
        for a in cfg.agent_ids:
            est = rr.estimates[a]
            est_s = f"{est:.1f}" if est is not None else "-"
            cells.append(f"{est_s}/e{rr.errors[a]:.0f}/r{rr.rewards[a]:.0f}".rjust(18))
        print(f"{rr.round_index:>5} {rr.truth:>8.1f} " + " ".join(cells))
    print("final credits: " +
          "  ".join(f"{a}={result.states[a].credits:.1f}"
                    f"{'' if result.states[a].alive else ' (dead)'}"
                    for a in cfg.agent_ids))

    # Data-quality diagnostics: a noisy tool-driver invalidates behavioural
    # claims, so surface parse-fail / mis-address rates on every run.
    from analysis.metrics import cooperation, deception, diagnostics
    ev = result.transcript.events
    dg = diagnostics(ev)
    dec = deception(ev)
    co = cooperation(ev)
    print(f"diagnostics: parse_fail={dg['parse_fail_rate']:.1%} "
          f"misaddress={dg['misaddress_rate']:.1%} "
          f"no_estimate={dg['rounds_without_estimate']}  |  "
          f"deception={dec['deception_rate']:.2f} ({dec['lies']}/{dec['offers']})  "
          f"cooperation={co['cooperation_index']:.2f}")


def main(argv: List[str] = None) -> None:
    """CLI entry point: parse args, build the config and policies, run one or more seeds, and write transcripts."""
    ap = argparse.ArgumentParser(description="Run the Agora Measurement Market.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--preset", choices=sorted(PRESETS), help="a built-in config preset")
    src.add_argument("--config", help="path to a YAML/JSON config file")
    ap.add_argument("--policies", default="honest_cooperator,bayesian_solo,liar,hoarder",
                    help="'llm', or comma-separated scripted policy names (cycled over agents)")
    ap.add_argument("--seed", type=int, help="override the config seed")
    ap.add_argument("--seeds", type=int, default=1, help="run this many consecutive seeds")
    ap.add_argument("--model", default="qwen3-32b",
                    help="served-model-name (vLLM) or deployment name (Azure/OpenAI)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default=None,
                    help="key for a hosted endpoint (or set AZURE_OPENAI_API_KEY / OPENAI_API_KEY)")
    ap.add_argument("--provider", choices=["vllm", "openai"], default=None,
                    help="endpoint flavour; inferred from --base-url when omitted")
    ap.add_argument("--memory", choices=["context", "markdown"], default=None,
                    help="LLM cross-game memory: full context (default) or per-round "
                         "markdown notes with a context reset each game")
    ap.add_argument("--trade-only", action="store_true",
                    help="censor numbers (digits and number words) from chat; values "
                         "can only be handed over via propose_trade")
    ap.add_argument("--min-trade-price", type=float, default=None,
                    help="reject trade offers priced below this (>0 forbids free gifts)")
    ap.add_argument("--out", default=None, help="directory for JSONL transcripts")
    args = ap.parse_args(argv)

    if args.config:
        base = load_config(args.config)
    else:
        base = PRESETS[args.preset or "smoke"]
    if args.seed is not None:
        base = base.with_(seed=args.seed)
    if args.memory:
        base = base.with_(memory=args.memory)
    if args.trade_only:
        base = base.with_(values_via_trade_only=True)
    if args.min_trade_price is not None:
        base = base.with_(min_trade_price=args.min_trade_price)

    for s in range(args.seeds):
        cfg = base.with_(seed=base.seed + s)
        tx = None
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            tx = Transcript(os.path.join(args.out, f"seed{cfg.seed}.jsonl"))
        policies = build_policies(cfg, args.policies, args.model, args.base_url,
                                  api_key=args.api_key, provider=args.provider)
        result = Referee(cfg, policies, tx).run()
        summarize(result, args.policies)
        if tx:
            tx.close()
    if args.out:
        print(f"\ntranscripts written to {args.out}/")


if __name__ == "__main__":
    main()
