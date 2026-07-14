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
                   provider: str = None, api_keys: dict = None) -> Dict[str, object]:
    """Build the per-agent policy map from a spec, cycled over the agents.

    Tokens, mixable in one spec:
      * 'llm'                          — an LLM seat on the DEFAULT endpoint
                                         (the ``model``/``base_url`` arguments);
      * '<model>@<base_url>[#provider]' — an LLM seat on its OWN endpoint, so
                                         different models play each other in the
                                         same game (e.g. 'gpt-5.4@https://liv.
                                         services.ai.azure.com/openai/v1' vs
                                         'qwen3-32b@http://node302:8765/v1');
      * a scripted baseline name        — a ground-truth bot (e.g. 'liar').

    Seats sharing an endpoint share one client. Every seat plays the identical
    game — same prompt, tools, and rules; only the model behind it differs.
    ``api_key`` is the default key; ``api_keys`` optionally maps a base_url (or
    a substring of it, e.g. the host) to a key for that endpoint. ``n_games``
    is passed to LLM seats so the system prompt announces the real match length."""
    ids = cfg.agent_ids
    names = [n.strip() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n != "llm" and "@" not in n and n not in REGISTRY]
    if unknown:
        raise SystemExit(f"unknown scripted policies: {unknown}; choose from "
                         f"{sorted(REGISTRY)}, 'llm', or 'model@base_url[#provider]'")

    backends: Dict[tuple, object] = {}   # (model, url, provider) -> shared client

    def _backend(m: str, u: str, p: str):
        """One client per distinct endpoint, with that endpoint's key."""
        sig = (m, u, p)
        if sig not in backends:
            from .backends import OpenAIBackend
            key = None
            if api_keys:
                key = api_keys.get(u) or next(
                    (v for h, v in api_keys.items() if h and h in u), None)
            backends[sig] = OpenAIBackend(model=m, base_url=u,
                                          api_key=key or api_key, provider=p)
        return backends[sig]

    def _make(name: str, aid: str):
        """One agent's policy: an LLM seat (default or per-seat endpoint) or a
        named scripted baseline."""
        peers = [p for p in ids if p != aid]
        if name == "llm":
            return LLMPolicy(_backend(model, base_url, provider), cfg, aid,
                             peers, n_games=n_games)
        if "@" in name:
            m, _, rest = name.partition("@")
            u, _, prov = rest.partition("#")
            return LLMPolicy(_backend(m.strip(), u.strip(), prov.strip() or None),
                             cfg, aid, peers, n_games=n_games)
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
    ap.add_argument("--paid-trades", action="store_true",
                    help="offers must cost more than 0 (any positive price; never free)")
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
    if args.paid_trades:
        base = base.with_(require_paid_trades=True)

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
