"""Model backends — the single seam through which every model call flows.

Keeping all generation behind ``Backend.generate`` means swapping the raw
local-vLLM client for an Inspect ``get_model()`` bridge later is a one-file
change. Two backends ship here:

  * ``OpenAIBackend`` — drives any OpenAI-compatible chat endpoint, in two
    flavours. ``provider="vllm"`` is the local Qwen3-32B server and uses the
    settings the 2026 serving recipe requires: non-stream, tool_choice="auto",
    temperature>0, thinking toggled via chat_template_kwargs.
    ``provider="openai"`` is a hosted endpoint — Azure AI Foundry's
    OpenAI-compatible ``/openai/v1`` (e.g. a gpt-5.4 deployment) or
    api.openai.com — whose reasoning models reject the vLLM-only sampling
    knobs, so only the portable arguments are sent.
  * ``MockBackend`` — a scripted, dependency-free stand-in for testing the LLM
    policy plumbing with no server.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .config import GameConfig


@dataclass
class RawToolCall:
    """A parsed tool call from the model: its id, function name, and decoded arguments."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    """A normalised model reply: spoken `content`, parsed `tool_calls`, and (thinking mode) `reasoning`."""
    content: Optional[str]
    tool_calls: List[RawToolCall] = field(default_factory=list)
    reasoning: Optional[str] = None


class OpenAIBackend:
    """Any OpenAI-compatible chat endpoint: local vLLM Qwen3-32B or a hosted cloud model.

    Local (provider="vllm"), serve with (see scripts/serve_qwen.sh):
        vllm serve Qwen/Qwen3-32B --served-model-name qwen3-32b \\
          --enable-auto-tool-choice --tool-call-parser hermes \\
          --reasoning-parser qwen3 --max-model-len 32768

    Hosted (provider="openai"), e.g. an Azure AI Foundry deployment:
        OpenAIBackend(model="gpt-5.4",
                      base_url="https://<resource>.services.ai.azure.com/openai/v1",
                      api_key="<key>")
    """

    def __init__(self, model: str = "qwen3-32b",
                 base_url: str = "http://localhost:8000/v1",
                 api_key: Optional[str] = None,
                 provider: Optional[str] = None):
        """Create the client. ``provider`` is "vllm" or "openai"; left unset it is
        inferred from the URL (Azure/OpenAI hosts -> "openai", else "vllm"). The
        key falls back to AGORA_API_KEY / AZURE_OPENAI_API_KEY / OPENAI_API_KEY."""
        from openai import OpenAI  # lazy: only needed for real runs
        self.model = model
        if provider not in (None, "", "vllm", "openai"):
            raise ValueError(f"unknown provider {provider!r}; use 'vllm' or 'openai'")
        hosted = any(h in base_url for h in ("azure", "openai.com", "anthropic.com"))
        self.provider = provider or ("openai" if hosted else "vllm")
        # AGORA_API_KEY is an explicit opt-in and applies to any endpoint; the
        # provider-brand vars apply ONLY to the hosted flavour — matched to the
        # host, so a mixed-model match resolves each seat's key from the right
        # variable — and a vLLM run never picks up (and transmits) a cloud key
        # the user exported for something else. No key at all falls back to
        # vLLM's accept-anything placeholder.
        key = api_key or os.environ.get("AGORA_API_KEY")
        if not key and self.provider == "openai":
            if "anthropic.com" in base_url:
                key = os.environ.get("ANTHROPIC_API_KEY")
            elif "azure" in base_url:
                key = os.environ.get("AZURE_OPENAI_API_KEY")
            key = key or os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(base_url=base_url, api_key=key or "EMPTY")
        # Running token meter — hosted models bill per token and this game's
        # growing-conversation design makes cost quadratic in match length, so
        # every driver/pilot needs real numbers, not guesses.
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def generate(self, messages: List[Dict[str, Any]],
                 tools: List[Dict[str, Any]], cfg: GameConfig) -> LLMResponse:
        """Call the chat endpoint (non-streaming, tool_choice=auto) and return a normalised LLMResponse."""
        sampling: Dict[str, Any] = {}
        if self.provider == "vllm":
            sampling = dict(
                temperature=cfg.temperature,    # never 0 for Qwen3
                top_p=0.8,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": cfg.enable_thinking},
                },
            )
        # hosted reasoning models (gpt-5.x) reject temperature/top_p/top_k
        # overrides and unknown body params, so "openai" sends none of them.
        # A text-only call (e.g. the markdown note-writing step) passes no
        # tools; the API rejects an EMPTY tools array, so omit it entirely.
        if tools:
            sampling["tools"] = tools
            sampling["tool_choice"] = "auto"    # "required" is buggy on Qwen3+vLLM
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,                       # streaming breaks hermes tool parsing
            **sampling,
        )
        u = getattr(resp, "usage", None)
        self.usage["calls"] += 1
        if u is not None:
            self.usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
            self.usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
            # cached prompt tokens are billed at a deep discount on hosted
            # endpoints; track them so cost projections reflect the real bill
            det = getattr(u, "prompt_tokens_details", None)
            cached = getattr(det, "cached_tokens", 0) or 0 if det else 0
            self.usage["cached_tokens"] = self.usage.get("cached_tokens", 0) + cached
        msg = resp.choices[0].message
        calls: List[RawToolCall] = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"__raw__": tc.function.arguments}
            calls.append(RawToolCall(id=tc.id, name=tc.function.name, arguments=args))
        reasoning = getattr(msg, "reasoning_content", None)
        return LLMResponse(content=msg.content, tool_calls=calls, reasoning=reasoning)


class MockBackend:
    """Returns pre-scripted responses; for unit-testing the LLM policy path.

    ``script`` is a callable ``(messages, tools, cfg) -> LLMResponse``.
    """

    def __init__(self, script: Callable[..., LLMResponse]):
        """Wrap a scripted (messages, tools, cfg) -> LLMResponse callable."""
        self.script = script
        self.model = "mock"          # seat label in match_start, like a real backend

    def generate(self, messages, tools, cfg) -> LLMResponse:
        """Return the scripted response (for tests, no server)."""
        return self.script(messages, tools, cfg)
