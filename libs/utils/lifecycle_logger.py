"""
lifecycle_logger.py — Reusable helper for structured JSONL lifecycle logging.

Thin facade around :class:`TrackEventLogger` providing the module-level
``log(event)`` convenience function requested in issue #31.

Usage::

    from libs.utils.lifecycle_logger import lifecycle_logger
    lifecycle_logger.log(event)

The output path defaults to the ``LIFECYCLE_LOG_PATH`` env variable
(``data/logs/tracks.jsonl`` when unset).
"""
from __future__ import annotations

from libs.logging.track_event_logger import TrackEventLogger
from libs.schemas.tracking import TrackLifecycleEvent


class LifecycleLogger:
    """Lazy-initialised singleton wrapper around :class:`TrackEventLogger`.

    The underlying file handle is created on first ``log()`` call, not at
    import time, so importing this module is always safe (even in tests or
    CI where the filesystem location may not exist yet).
    """

    def __init__(self) -> None:
        self._logger: TrackEventLogger | None = None

    def _ensure_logger(self) -> TrackEventLogger:
        if self._logger is None:
            self._logger = TrackEventLogger()  # reads LIFECYCLE_LOG_PATH from settings
        return self._logger

    def log(self, event: TrackLifecycleEvent) -> None:
        """Append a single lifecycle event to the JSONL file + console."""
        self._ensure_logger().log_event(event)

    def log_batch(self, events: list[TrackLifecycleEvent]) -> None:
        """Append multiple lifecycle events."""
        self._ensure_logger().log_batch(events)


# Module-level singleton — ``lifecycle_logger.log(event)``
lifecycle_logger = LifecycleLogger()
