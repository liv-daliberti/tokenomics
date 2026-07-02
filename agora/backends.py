"""Model backends — the single seam through which every model call flows.

Keeping all generation behind ``Backend.generate`` means swapping the raw
local-vLLM client for an Inspect ``get_model()`` bridge later is a one-file
change. Two backends ship here:

  * ``OpenAIBackend`` — drives a local vLLM OpenAI-compatible endpoint serving
    Qwen3-32B. Uses the settings the 2026 serving recipe requires: non-stream,
    tool_choice="auto", temperature>0, thinking toggled via chat_template_kwargs.
  * ``MockBackend`` — a scripted, dependency-free stand-in for testing the LLM
    policy plumbing with no server.
"""
from __future__ import annotations

import json
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
    """Local Qwen3-32B behind vLLM's OpenAI-compatible server.

    Serve with (see scripts/serve_qwen.sh):
        vllm serve Qwen/Qwen3-32B --served-model-name qwen3-32b \\
          --enable-auto-tool-choice --tool-call-parser hermes \\
          --reasoning-parser qwen3 --max-model-len 32768
    """

    def __init__(self, model: str = "qwen3-32b",
                 base_url: str = "http://localhost:8000/v1",
                 api_key: str = "EMPTY"):
        """Create an OpenAI-compatible client pointed at the local vLLM server."""
        from openai import OpenAI  # lazy: only needed for real runs
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(self, messages: List[Dict[str, Any]],
                 tools: List[Dict[str, Any]], cfg: GameConfig) -> LLMResponse:
        """Call the chat endpoint (non-streaming, tool_choice=auto, temperature>0) and return a normalised LLMResponse."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",                 # "required" is buggy on Qwen3+vLLM
            temperature=cfg.temperature,        # never 0 for Qwen3
            top_p=0.8,
            stream=False,                       # streaming breaks hermes tool parsing
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": cfg.enable_thinking},
            },
        )
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

    def generate(self, messages, tools, cfg) -> LLMResponse:
        """Return the scripted response (for tests, no server)."""
        return self.script(messages, tools, cfg)
