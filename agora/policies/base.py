"""The policy interface the referee drives.

A policy proposes actions; it never mutates world state. The referee owns
execution and hands back tool-result strings, which lets an LLM policy run its
inner tool-call loop within a single tick while a scripted policy simply doles
out a precomputed plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..types import Action


@dataclass
class ToolInvocation:
    """One tool call proposed by a policy in a single model step."""

    call_id: str
    name: str
    action: Optional[Action]      # None if the call could not be parsed
    error: Optional[str] = None   # parse-error message (counted as a data-quality event)


class Policy:
    """Base class. Subclasses implement the three-method turn protocol."""

    def reset_game(self, game_index: int, n_games: int = 1) -> None:
        """Called once before each game in a match. Memory persists across games."""

    def reset_round(self, round_index: int) -> None:
        """Called once at the start of each round."""

    def consume_boundary_note(self) -> str:
        """A pending new-game marker to prepend to this agent's next observation.

        The referee prepends it BEFORE logging the prompt event, so the
        transcript shows exactly what the agent was sent (in markdown-memory
        mode this note carries the agent's whole notebook). Returned once then
        cleared; '' when there is nothing pending."""
        return ""

    def start_turn(self, observation_text: str, observation: Dict[str, Any]) -> None:
        """Called once at the start of an agent's turn within a tick."""

    def next_actions(self) -> List[ToolInvocation]:
        """Return the tool calls for the next model step (empty list = done)."""
        raise NotImplementedError

    def last_reasoning(self) -> Optional[str]:
        """The agent's reasoning for the step just returned by next_actions().

        Returned once then cleared, so the referee can log it. None if the policy
        has no reasoning to surface (e.g. a purely scripted rule)."""
        return None

    def observe_results(self, results: List[Tuple[str, str]]) -> None:
        """Feed back (call_id, result_string) pairs for the calls just executed."""

    def write_round_notes(self, game_index: int, round_index: int,
                          outcome_text: str = "") -> Optional[str]:
        """Called by the referee after each round when markdown memory is on.

        Returns the markdown the agent wants to append to its running notes
        document (logged to the transcript), or None to write nothing —
        the default for scripted policies, which need no notebook."""
        return None

    def notes_doc(self) -> str:
        """The agent's accumulated markdown notes document ('' if none)."""
        return ""
