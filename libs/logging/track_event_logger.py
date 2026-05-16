"""
track_event_logger.py — Persist TrackLifecycleEvent objects to console + JSONL.

Writes one JSON object per line to ``data/logs/tracks.jsonl`` (configurable)
and mirrors every event to the Python stdlib logger at INFO level.

The JSONL schema follows the contract from Issue #14::

    {"event": "BIRTH", "track_id": 5, "timestamp": "...", "zone": "restricted_door"}
    {"event": "LOST",  "track_id": 5, "timestamp": "...", "dwell_time_sec": 18}
    {"event": "DEAD",  "track_id": 5, "timestamp": "..."}
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from libs.schemas.tracking import TrackLifecycleEvent

logger = logging.getLogger(__name__)


class TrackEventLogger:
    """
    Dual-sink logger for track lifecycle events.

    * **Console**: Python ``logging`` at INFO level.
    * **File**: Append-mode JSONL at *log_path* (parent dirs created automatically).

    Parameters
    ----------
    log_path : Path
        Destination JSONL file.  Defaults to ``data/logs/tracks.jsonl``.
    """

    DEFAULT_LOG_PATH = Path("data/logs/tracks.jsonl")

    def __init__(self, log_path: Path | None = None) -> None:
        self.log_path = log_path or self.DEFAULT_LOG_PATH
        os.makedirs(self.log_path.parent, exist_ok=True)

    # ── Public API ──────────────────────────────────────────────────────────

    def log_event(self, event: TrackLifecycleEvent) -> None:
        """
        Serialize *event* to the issue-spec JSON schema, write to file + console.
        """
        record = event.to_jsonl_dict()

        # Console
        logger.info("Track %s: #%d %s", record["event"], record["track_id"],
                     json.dumps(record, default=str))

        # JSONL file (append)
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def log_batch(self, events: list[TrackLifecycleEvent]) -> None:
        """
        Convenience wrapper: log every event in *events*.
        """
        for event in events:
            self.log_event(event)
