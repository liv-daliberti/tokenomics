"""OpenAIBackend request shaping (no model server needed).

The one thing the two provider flavours must not share is the sampling block:
vLLM needs the Qwen recipe (temperature>0, top_p, top_k, chat_template_kwargs),
while hosted reasoning models (gpt-5.x on Azure/OpenAI) reject exactly those
arguments. A stub client captures the kwargs each flavour actually sends.
"""
from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import openai  # noqa: F401  (OpenAIBackend imports it lazily)
    _HAVE_OPENAI = True
except ImportError:
    _HAVE_OPENAI = False

from agora.config import GameConfig


def _stub_client():
    """A fake OpenAI client whose chat.completions.create records its kwargs."""
    box = {}

    def create(**kwargs):
        box["kwargs"] = kwargs
        msg = types.SimpleNamespace(content="ok", tool_calls=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    return client, box


_A_TOOL = [{"type": "function",
            "function": {"name": "noop",
                         "parameters": {"type": "object", "properties": {}}}}]


def _generate(backend, tools=_A_TOOL):
    """Run one generate() through a stub client and return the captured kwargs."""
    backend.client, box = _stub_client()
    cfg = GameConfig(agent_ids=["A", "B"], seed=0)
    resp = backend.generate([{"role": "user", "content": "hi"}], tools, cfg)
    assert resp.content == "ok" and resp.tool_calls == []
    return box["kwargs"], cfg


def test_vllm_flavour_sends_qwen_sampling_knobs():
    if not _HAVE_OPENAI:
        print("skip: openai not installed"); return
    from agora.backends import OpenAIBackend
    be = OpenAIBackend()  # defaults: localhost vLLM
    assert be.provider == "vllm"
    kwargs, cfg = _generate(be)
    assert kwargs["temperature"] == cfg.temperature
    assert kwargs["top_p"] == 0.8
    assert kwargs["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": cfg.enable_thinking}


def test_hosted_flavour_sends_only_portable_arguments():
    if not _HAVE_OPENAI:
        print("skip: openai not installed"); return
    from agora.backends import OpenAIBackend
    be = OpenAIBackend(model="gpt-5.4",
                       base_url="https://liv.services.ai.azure.com/openai/v1",
                       api_key="test-key")
    assert be.provider == "openai"  # inferred from the Azure URL
    kwargs, _ = _generate(be)
    for banned in ("temperature", "top_p", "extra_body"):
        assert banned not in kwargs
    assert kwargs["tool_choice"] == "auto" and kwargs["stream"] is False
    assert kwargs["model"] == "gpt-5.4"


def test_empty_tools_are_omitted_from_the_request():
    # A text-only call (the markdown note-writing step) must not send an empty
    # tools array — hosted APIs reject `tools: []`.
    if not _HAVE_OPENAI:
        print("skip: openai not installed"); return
    from agora.backends import OpenAIBackend
    kwargs, _ = _generate(OpenAIBackend(), tools=[])
    assert "tools" not in kwargs and "tool_choice" not in kwargs


def test_invalid_prompt_filter_skips_the_turn_not_the_match():
    # Hosted endpoints sometimes 400 a whole conversation as 'invalid_prompt'
    # (usage-policy heuristic; observed on notebook injections). One filtered
    # turn must degrade to a visible no-op, never kill a paid match — and any
    # OTHER error must still raise.
    if not _HAVE_OPENAI:
        print("skip: openai not installed"); return
    from agora.backends import OpenAIBackend
    be = OpenAIBackend()

    class _Filtered(Exception):
        code = "invalid_prompt"

    def raise_filtered(**kwargs):
        raise _Filtered("flagged as potentially violating our usage policy")

    be.client = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=raise_filtered)))
    cfg = GameConfig(agent_ids=["A", "B"], seed=0)
    resp = be.generate([{"role": "user", "content": "hi"}], _A_TOOL, cfg)
    assert resp.tool_calls == [] and "provider filtered" in resp.content
    assert be.usage["filtered"] == 1 and be.usage["calls"] == 1

    class _Other(Exception):
        code = "context_length_exceeded"

    def raise_other(**kwargs):
        raise _Other("too long")

    be.client.chat.completions.create = raise_other
    try:
        be.generate([{"role": "user", "content": "hi"}], _A_TOOL, cfg)
        raise AssertionError("expected the non-filter error to propagate")
    except _Other:
        pass


def test_provider_explicit_override_and_validation():
    if not _HAVE_OPENAI:
        print("skip: openai not installed"); return
    from agora.backends import OpenAIBackend
    # A remote vLLM node must NOT be misread as hosted: explicit provider wins.
    be = OpenAIBackend(base_url="http://node302:8765/v1", provider="vllm")
    assert be.provider == "vllm"
    try:
        OpenAIBackend(provider="bogus")
        raise AssertionError("expected ValueError for unknown provider")
    except ValueError:
        pass


if __name__ == "__main__":
    test_vllm_flavour_sends_qwen_sampling_knobs()
    test_hosted_flavour_sends_only_portable_arguments()
    test_provider_explicit_override_and_validation()
    print("ok")
