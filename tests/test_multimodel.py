"""Mixed-model matches: per-seat endpoints via 'model@base_url[#provider]'.

Different models must be able to play each other in the SAME game — a hosted
gpt vs a local qwen vs a scripted bot — with one shared client per endpoint,
per-host key resolution, and the seating recorded in the transcript so a
mixed match is self-describing.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import openai  # noqa: F401
    _HAVE_OPENAI = True
except ImportError:
    _HAVE_OPENAI = False

from agora.backends import LLMResponse, MockBackend, RawToolCall
from agora.config import GameConfig
from agora.policies import LLMPolicy, REGISTRY
from agora.referee import run_match
from agora.transcripts import Transcript

AZURE = "https://liv.services.ai.azure.com/openai/v1"
LOCAL = "http://localhost:8000/v1"


def test_seat_spec_mixes_models_and_scripted_bots():
    if not _HAVE_OPENAI:
        print("skip: openai not installed"); return
    from agora.run import build_policies
    cfg = GameConfig(agent_ids=["A", "B", "C"], seed=0)
    pol = build_policies(cfg, f"gpt-5.4@{AZURE},liar,qwen3-32b@{LOCAL}",
                         "unused-default", "http://unused/v1")
    assert isinstance(pol["A"], LLMPolicy) and pol["A"].backend.model == "gpt-5.4"
    assert pol["A"].backend.provider == "openai"           # inferred from azure URL
    assert type(pol["B"]).__name__ == "Liar"
    assert isinstance(pol["C"], LLMPolicy) and pol["C"].backend.model == "qwen3-32b"
    assert pol["C"].backend.provider == "vllm"             # localhost = vLLM recipe


def test_seats_sharing_an_endpoint_share_one_client():
    if not _HAVE_OPENAI:
        print("skip: openai not installed"); return
    from agora.run import build_policies
    cfg = GameConfig(agent_ids=["A", "B", "C"], seed=0)
    pol = build_policies(cfg, f"m@{LOCAL},m@{LOCAL},llm", "deflt", LOCAL)
    assert pol["A"].backend is pol["B"].backend            # same endpoint -> same client
    assert pol["A"].backend is not pol["C"].backend        # different model name


def test_per_host_keys_and_provider_override():
    if not _HAVE_OPENAI:
        print("skip: openai not installed"); return
    from agora.run import build_policies
    cfg = GameConfig(agent_ids=["A", "B"], seed=0)
    pol = build_policies(cfg, f"gpt-5.4@{AZURE},claude-x@http://127.0.0.1:8111/v1#openai",
                         "d", LOCAL,
                         api_keys={"liv.services.ai.azure.com": "azure-key",
                                   "127.0.0.1": "other-key"})
    assert pol["A"].backend.client.api_key == "azure-key"
    b = pol["B"].backend
    assert b.provider == "openai"                          # explicit #provider wins
    assert b.client.api_key == "other-key"


def test_anthropic_url_autodetects_hosted_flavour():
    if not _HAVE_OPENAI:
        print("skip: openai not installed"); return
    from agora.backends import OpenAIBackend
    be = OpenAIBackend(model="claude-sonnet-5",
                       base_url="https://api.anthropic.com/v1/", api_key="k")
    assert be.provider == "openai"


def test_match_start_records_the_seating():
    def script(messages, tools, cfg):
        if messages[-1]["role"] == "user":
            return LLMResponse(content=None,
                               tool_calls=[RawToolCall("c1", "measure", {})])
        return LLMResponse(content=None, tool_calls=[
            RawToolCall("c2", "submit_estimate", {"value": 500.0}),
            RawToolCall("c3", "end_turn", {}),
        ])
    cfg = GameConfig(agent_ids=["A", "B"], horizon_mode="fixed", n_rounds=1,
                     max_ticks=1, seed=0)
    policies = {"A": LLMPolicy(MockBackend(script), cfg, "A", ["B"]),
                "B": REGISTRY["liar"](cfg, "B", ["A", "B"])}
    tx = Transcript()
    run_match(cfg, policies, 1, tx)
    seats = next(e for e in tx.events if e["event"] == "match_start")["seats"]
    assert seats["A"] == "mock" and seats["B"] == "Liar"

    from analysis.viz import render_simple
    assert "seats:" in render_simple(tx.events, "t")


def test_web_key_field_parses_bare_and_pairs():
    from web.app import _parse_keys
    assert _parse_keys("") == {}
    assert _parse_keys("sk-abc123") == {"*": "sk-abc123"}
    # an Azure-style key ending in '=' stays a bare key, never a host=key pair
    assert _parse_keys("AbCd12==") == {"*": "AbCd12=="}
    pairs = _parse_keys("liv.services.ai.azure.com=K1  api.anthropic.com=K2")
    assert pairs == {"liv.services.ai.azure.com": "K1", "api.anthropic.com": "K2"}
    # one malformed token -> the whole field is treated as a single key
    assert "*" in _parse_keys("liv.azure.com=K1 notahost")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
