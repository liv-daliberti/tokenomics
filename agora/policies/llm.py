"""LLM-backed policy: a single chat history that co-evolves across the game.

The whole game is one growing conversation per agent (system prompt once, an
observation appended each turn, assistant/tool messages within a turn). This is
the "co-evolving within a context window" setting from the brief. The policy
only talks to the model through a ``Backend``; the referee executes the tool
calls and feeds results back.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from ..config import GameConfig
from ..tools import parse_action, system_prompt, tool_schemas
from .base import Policy, ToolInvocation


class LLMPolicy(Policy):
    def __init__(self, backend, cfg: GameConfig, agent_id: str, peers: List[str]):
        self.backend = backend
        self.cfg = cfg
        self.agent_id = agent_id
        self.tools = tool_schemas(cfg)
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt(cfg, agent_id, peers)}
        ]
        self._pending_ids: List[str] = []
        self.parse_failures = 0

    def start_turn(self, observation_text: str, observation: Dict[str, Any]) -> None:
        self.messages.append({"role": "user", "content": observation_text})

    def next_actions(self) -> List[ToolInvocation]:
        resp = self.backend.generate(self.messages, self.tools, self.cfg)

        # Reconstruct the assistant turn so the running history stays valid.
        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": resp.content or ""}
        if resp.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in resp.tool_calls
            ]
        self.messages.append(assistant_msg)

        invocations: List[ToolInvocation] = []
        for tc in resp.tool_calls:
            try:
                action = parse_action(tc.name, tc.arguments)
                invocations.append(ToolInvocation(tc.id, tc.name, action))
            except ValueError as exc:
                self.parse_failures += 1
                invocations.append(ToolInvocation(tc.id, tc.name, None, error=str(exc)))
        self._pending_ids = [inv.call_id for inv in invocations]
        return invocations

    def observe_results(self, results: List[Tuple[str, str]]) -> None:
        for call_id, result in results:
            self.messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": result}
            )
        self._pending_ids = []
