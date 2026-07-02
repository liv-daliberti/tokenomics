"""Structured, append-only transcript logging.

Every event in a game is written as one JSON object per line (JSONL). The
transcript is the ground truth for all downstream analysis and is designed to
be the substrate for mechanistic-interpretability work later: because the
referee generates each measurement, we log both what an agent *observed* and
what it later *reported*, which makes deception directly checkable.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .types import to_jsonable


class Transcript:
    """Append-only JSONL event log (one JSON object per line) — the substrate for all analysis, optionally streamed to a file as the game runs."""
    def __init__(self, path: Optional[str] = None):
        """Open the log, optionally to a file path (creating parent dirs)."""
        self.path = path
        self.events: List[Dict[str, Any]] = []
        self._fh = None
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._fh = open(path, "w")

    def log(self, event_type: str, **fields: Any) -> None:
        """Record one event (type plus fields), converting dataclasses/enums to plain JSON."""
        event = {"event": event_type}
        event.update({k: to_jsonable(v) for k, v in fields.items()})
        self.events.append(event)
        if self._fh:
            self._fh.write(json.dumps(event) + "\n")
            self._fh.flush()

    def close(self) -> None:
        """Close the underlying file, if any."""
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "Transcript":
        """Context-manager entry; returns self."""
        return self

    def __exit__(self, *exc) -> None:
        """Context-manager exit; closes the file."""
        self.close()
