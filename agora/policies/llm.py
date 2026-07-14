"""LLM-backed policy: a single chat history that co-evolves across the game.

The whole game is one growing conversation per agent (system prompt once, an
observation appended each turn, assistant/tool messages within a turn). This is
the "co-evolving within a context window" setting from the brief. The policy
only talks to the model through a ``Backend``; the referee executes the tool
calls and feeds results back.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from ..config import GameConfig
from ..tools import parse_action, system_prompt, tool_schemas
from .base import Policy, ToolInvocation


class LLMPolicy(Policy):
    """Drives an agent with a chat model, keeping ONE growing conversation across the whole match (system prompt once, an observation per turn, assistant/tool messages within a turn) — the 'co-evolving within a context window' setting."""
    def __init__(self, backend, cfg: GameConfig, agent_id: str, peers: List[str],
                 n_games: int = 1):
        """Build the tool schemas and seed the history with the system prompt (which announces the match length)."""
        self.backend = backend
        self.cfg = cfg
        self.agent_id = agent_id
        self.n_games = n_games
        self.tools = tool_schemas(cfg)
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt(cfg, agent_id, peers, n_games)}
        ]
        self._pending_ids: List[str] = []
        self._game_prefix = ""
        self._thought: Optional[str] = None
        self.parse_failures = 0
        # markdown-memory mode: the per-round notes this agent has written,
        # as (game_index, round_index, markdown) — the source of its notebook.
        self.notes: List[tuple] = []

    def reset_game(self, game_index: int, n_games: int = 1) -> None:
        """Mark a new-game boundary. In "context" mode the whole conversation is
        kept; in "markdown" mode it is RESET to the system prompt and the agent
        gets its own accumulated notes instead."""
        if game_index == 0:
            return
        total = max(n_games, self.n_games)
        if self.cfg.memory == "markdown":
            self.messages = self.messages[:1]   # keep only the system prompt
            doc = self.notes_doc()
            notes_block = (f"Your notes from the previous games:\n\n{doc}"
                           if doc else "You have no notes from previous games.")
            self._game_prefix = (
                f"=== A NEW GAME (game {game_index + 1} of {total}) begins now. The hidden "
                f"value is drawn afresh and every agent's credits are reset to the starting "
                f"amount. You are playing the same agents as before. Your conversation from "
                f"the previous games has been cleared — what you know about them is the "
                f"notes you wrote. ===\n\n{notes_block}"
            )
        else:
            # Keep the whole conversation; just mark the boundary (prepended to
            # the next observation to avoid two consecutive user turns).
            self._game_prefix = (
                f"=== A NEW GAME (game {game_index + 1} of {total}) begins now. The hidden "
                f"value is drawn afresh and every agent's credits are reset to the starting "
                f"amount, but you are playing the same agents and you remember everything "
                f"from the previous games. ==="
            )

    def consume_boundary_note(self) -> str:
        """Hand the pending game-boundary note (with the notebook, in markdown
        mode) to the referee, which prepends it to the next observation so the
        logged prompt matches what the model actually receives."""
        note, self._game_prefix = self._game_prefix, ""
        return note

    def start_turn(self, observation_text: str, observation: Dict[str, Any]) -> None:
        """Append this turn's observation as a user message (the referee has
        already folded in any game-boundary note)."""
        self.messages.append({"role": "user", "content": observation_text})

    def next_actions(self) -> List[ToolInvocation]:
        """Call the model once, capture its reasoning, append the assistant turn, and return the parsed tool calls (empty list = done)."""
        resp = self.backend.generate(self.messages, self.tools, self.cfg)

        # Capture the model's reasoning for this step: its <think> content (only
        # present in thinking mode) plus any spoken explanation in `content`.
        bits = []
        if resp.reasoning:
            bits.append(resp.reasoning.strip())
        if resp.content:
            bits.append(resp.content.strip())
        self._thought = "\n".join(b for b in bits if b) or None

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

    def last_reasoning(self) -> Optional[str]:
        """Return, once, the reasoning captured for the most recent model step."""
        thought, self._thought = self._thought, None
        return thought

    def observe_results(self, results: List[Tuple[str, str]]) -> None:
        """Append one tool-result message per executed call so the next model step sees the outcomes."""
        for call_id, result in results:
            self.messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": result}
            )
        self._pending_ids = []

    # ------------------------------------------------------- markdown memory --
    def write_round_notes(self, game_index: int, round_index: int,
                          outcome_text: str = "") -> Optional[str]:
        """One extra text-only model call asking the agent to journal the round.

        Only active in "markdown" memory mode. The request (and the reply) stay
        in the conversation, which is cleared at the next game boundary — the
        note itself is what survives, via ``notes_doc``."""
        if self.cfg.memory != "markdown":
            return None
        ask = (f"The round is over.{' ' + outcome_text if outcome_text else ''} "
               f"Append concise markdown notes on what happened this round — what "
               f"you measured, said, traded, estimated, how it turned out, and "
               f"anything worth remembering for later games. At each new game your "
               f"conversation is cleared and ONLY these notes are shown back to "
               f"you, so write what your future self needs. Reply with ONLY the "
               f"markdown to append (a few bullet points; no preamble).")
        self.messages.append({"role": "user", "content": ask})
        resp = self.backend.generate(self.messages, [], self.cfg)  # text-only: no tools
        text = (resp.content or "").strip()
        self.messages.append({"role": "assistant", "content": text})
        if not text:
            return None
        self.notes.append((game_index, round_index, text))
        return text

    def notes_doc(self) -> str:
        """The agent's whole notebook: every round note under game/round headings."""
        parts, last_game = [], None
        for g, r, text in self.notes:
            if g != last_game:
                parts.append(f"# Game {g + 1}")
                last_game = g
            parts.append(f"## Round {r}\n\n{text}")
        return "\n\n".join(parts)
