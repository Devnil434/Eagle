"""
Unit tests for the structured JSONL lifecycle logger (issue #31).

Validates:
  - JSONL file created automatically on first log() call
  - Events appended correctly (append mode, not overwrite)
  - Path configurable via LIFECYCLE_LOG_PATH env variable
  - Mocked file writes (no real disk I/O) where appropriate
  - The module-level lifecycle_logger singleton works
  - Existing TrackEventLogger behaviour is preserved
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import mock_open, patch, MagicMock

import pytest

from libs.schemas.tracking import TrackLifecycleEvent, TrackState
from libs.logging.track_event_logger import TrackEventLogger
from libs.utils.lifecycle_logger import LifecycleLogger


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_event(
    state: TrackState = TrackState.BORN,
    track_id: int = 3,
    frame_id: int = 0,
    zones: list[str] | None = None,
    dwell: float = 0.0,
    ts_ms: float = 1718000000000.0,
) -> TrackLifecycleEvent:
    return TrackLifecycleEvent(
        event=state,
        track_id=track_id,
        frame_id=frame_id,
        zones_present=zones or ["restricted_door"],
        dwell_time_seconds=dwell,
        timestamp_ms=ts_ms,
    )


# ── JSONL file auto-creation ─────────────────────────────────────────────────

class TestFileCreation:

    def test_jsonl_file_created_on_first_log(self, tmp_path: Path):
        log_file = tmp_path / "tracks.jsonl"
        tl = TrackEventLogger(log_path=log_file)
        tl.log_event(_make_event())
        assert log_file.exists()

    def test_parent_dirs_created_automatically(self, tmp_path: Path):
        log_file = tmp_path / "deep" / "nested" / "tracks.jsonl"
        tl = TrackEventLogger(log_path=log_file)
        tl.log_event(_make_event())
        assert log_file.exists()


# ── Append mode (no overwrite) ───────────────────────────────────────────────

class TestAppendMode:

    def test_multiple_events_appended(self, tmp_path: Path):
        log_file = tmp_path / "tracks.jsonl"
        tl = TrackEventLogger(log_path=log_file)

        tl.log_event(_make_event(TrackState.BORN, track_id=1))
        tl.log_event(_make_event(TrackState.LOST, track_id=1, dwell=12.5))
        tl.log_event(_make_event(TrackState.DEAD, track_id=1, dwell=30.1))

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

    def test_events_are_valid_json(self, tmp_path: Path):
        log_file = tmp_path / "tracks.jsonl"
        tl = TrackEventLogger(log_path=log_file)
        tl.log_event(_make_event())

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        record = json.loads(lines[0])
        assert record["event"] == "BIRTH"
        assert record["track_id"] == 3

    def test_batch_logging_appends_all(self, tmp_path: Path):
        log_file = tmp_path / "tracks.jsonl"
        tl = TrackEventLogger(log_path=log_file)

        events = [
            _make_event(TrackState.BORN, track_id=1, frame_id=0),
            _make_event(TrackState.LOST, track_id=1, frame_id=10, dwell=5.0),
            _make_event(TrackState.DEAD, track_id=1, frame_id=20),
        ]
        tl.log_batch(events)

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        parsed = [json.loads(line) for line in lines]
        assert [r["event"] for r in parsed] == ["BIRTH", "LOST", "DEAD"]


# ── Configurable path via env variable ───────────────────────────────────────

class TestEnvConfig:

    def test_lifecycle_log_path_env_override(self, tmp_path: Path, monkeypatch):
        custom_path = str(tmp_path / "custom" / "events.jsonl")
        monkeypatch.setenv("LIFECYCLE_LOG_PATH", custom_path)

        # Force fresh Settings instance to pick up env var
        from libs.config.settings import Settings
        s = Settings()
        assert s.lifecycle_log_path == custom_path

    def test_default_path_without_env(self):
        from libs.config.settings import Settings
        s = Settings()
        assert s.lifecycle_log_path == "data/logs/tracks.jsonl"


# ── Mocked file writes (no real disk I/O) ────────────────────────────────────

class TestMockedWrites:

    @patch("libs.logging.track_event_logger.os.makedirs")
    def test_log_event_calls_open_in_append_mode(self, mock_makedirs):
        m = mock_open()
        with patch("builtins.open", m):
            tl = TrackEventLogger(log_path=Path("/fake/tracks.jsonl"))
            tl.log_event(_make_event())

        m.assert_called_once_with(
            Path("/fake/tracks.jsonl"), "a", encoding="utf-8"
        )

    @patch("libs.logging.track_event_logger.os.makedirs")
    def test_log_event_writes_valid_json_line(self, mock_makedirs):
        m = mock_open()
        with patch("builtins.open", m):
            tl = TrackEventLogger(log_path=Path("/fake/tracks.jsonl"))
            tl.log_event(_make_event(TrackState.DEAD, track_id=7))

        written = m().write.call_args[0][0]
        record = json.loads(written.strip())
        assert record["event"] == "DEAD"
        assert record["track_id"] == 7

    @patch("libs.logging.track_event_logger.os.makedirs")
    def test_log_event_writes_newline_terminated(self, mock_makedirs):
        m = mock_open()
        with patch("builtins.open", m):
            tl = TrackEventLogger(log_path=Path("/fake/tracks.jsonl"))
            tl.log_event(_make_event())

        written = m().write.call_args[0][0]
        assert written.endswith("\n")


# ── lifecycle_logger singleton ───────────────────────────────────────────────

class TestLifecycleLoggerSingleton:

    def test_log_delegates_to_track_event_logger(self, tmp_path: Path):
        ll = LifecycleLogger()
        ll._logger = TrackEventLogger(log_path=tmp_path / "tracks.jsonl")
        ll.log(_make_event(track_id=42))

        lines = (tmp_path / "tracks.jsonl").read_text().strip().splitlines()
        record = json.loads(lines[0])
        assert record["track_id"] == 42

    def test_singleton_importable(self):
        from libs.utils.lifecycle_logger import lifecycle_logger
        assert hasattr(lifecycle_logger, "log")
        assert hasattr(lifecycle_logger, "log_batch")
