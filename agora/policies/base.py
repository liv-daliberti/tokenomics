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

    def reset_round(self, round_index: int) -> None:
        """Called once at the start of each round."""

    def start_turn(self, observation_text: str, observation: Dict[str, Any]) -> None:
        """Called once at the start of an agent's turn within a tick."""

    def next_actions(self) -> List[ToolInvocation]:
        """Return the tool calls for the next model step (empty list = done)."""
        raise NotImplementedError

    def observe_results(self, results: List[Tuple[str, str]]) -> None:
        """Feed back (call_id, result_string) pairs for the calls just executed."""
