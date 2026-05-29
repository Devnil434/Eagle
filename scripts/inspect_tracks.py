"""
Inspect track memory stored by the Eagle Redis memory layer.

Examples:
    python scripts/inspect_tracks.py
    python scripts/inspect_tracks.py --track 3
    python scripts/inspect_tracks.py --track 3 --last 10
    python scripts/inspect_tracks.py --json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit
from libs.config.settings import settings

import redis

DEFAULT_CAMERA_ID = "cam_01"
DEFAULT_REDIS_URL = settings.REDIS_URL


@dataclass
class TrackSummary:
    camera_id: str
    track_id: int
    state: str = "UNKNOWN"
    event_count: int = 0
    dwell_time_seconds: float = 0.0
    zone: str = "unknown"
    action_summary: str = "no events"
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, include_events: bool = True) -> dict[str, Any]:
        data = {
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "state": self.state,
            "event_count": self.event_count,
            "dwell_time_seconds": self.dwell_time_seconds,
            "zone": self.zone,
            "action_summary": self.action_summary,
        }
        if include_events:
            data["events"] = self.events
        return data


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _loads(raw: Any) -> Any:
    return json.loads(_decode(raw))


def _iter_keys(redis_client: Any, pattern: str) -> list[str]:
    keys = redis_client.keys(pattern)
    return sorted(_decode(key) for key in keys)


def _track_id_from_key(key: str) -> int:
    return int(key.rsplit(":", 1)[-1])


def _normalise_event(event: dict[str, Any]) -> dict[str, Any] | None:
    normalised = dict(event)
    if "event" in normalised:
        normalised["event"] = str(normalised["event"])
    try:
        if "track_id" in normalised:
            normalised["track_id"] = int(normalised["track_id"])
        if "frame_id" in normalised:
            normalised["frame_id"] = int(normalised["frame_id"])
    except (TypeError, ValueError):
        return None
    return normalised


def _event_sort_key(event: dict[str, Any]) -> tuple[int, float]:
    return (
        int(event.get("frame_id", 0)),
        float(event.get("timestamp_ms", 0.0)),
    )


def load_track_record(redis_client: Any, camera_id: str, track_id: int) -> dict[str, Any] | None:
    raw = redis_client.get(f"track:{camera_id}:{track_id}")
    return _loads(raw) if raw else None


def load_events(
    redis_client: Any,
    camera_id: str,
    track_id: int | None = None,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for key in _iter_keys(redis_client, f"event:{camera_id}:*"):
        raw = redis_client.get(key)
        if not raw:
            continue

        try:
            stored_events = _loads(raw)
        except json.JSONDecodeError:
            continue

        if not isinstance(stored_events, list):
            continue

        for event in stored_events:
            if not isinstance(event, dict):
                continue
            normalised = _normalise_event(event)
            if normalised is None:
                continue
            if track_id is not None and normalised.get("track_id") != track_id:
                continue
            events.append(normalised)

    return sorted(events, key=_event_sort_key)


def build_action_summary(events: Iterable[dict[str, Any]]) -> str:
    names = [str(event.get("event", "UNKNOWN")).lower() for event in events]
    return " -> ".join(names) if names else "no events"


def build_track_summary(
    redis_client: Any,
    camera_id: str,
    track_id: int,
    last: int | None = None,
) -> TrackSummary:
    record = load_track_record(redis_client, camera_id, track_id) or {}
    events = load_events(redis_client, camera_id, track_id=track_id)
    if last is not None:
        events = events[-last:]

    zones = record.get("zones_present") or []
    if not zones and events:
        zones = events[-1].get("zones_present") or []

    return TrackSummary(
        camera_id=camera_id,
        track_id=track_id,
        state=str(record.get("state", "UNKNOWN")),
        event_count=len(events),
        dwell_time_seconds=float(record.get("dwell_time_seconds", 0.0)),
        zone=", ".join(zones) if zones else "unknown",
        action_summary=build_action_summary(events),
        events=events,
    )


def list_track_ids(redis_client: Any, camera_id: str) -> list[int]:
    return [_track_id_from_key(key) for key in _iter_keys(redis_client, f"track:{camera_id}:*")]


def inspect_tracks(
    redis_client: Any,
    camera_id: str,
    track_id: int | None = None,
    last: int | None = None,
) -> list[TrackSummary]:
    track_ids = [track_id] if track_id is not None else list_track_ids(redis_client, camera_id)
    return [
        build_track_summary(redis_client, camera_id, current_track_id, last=last)
        for current_track_id in track_ids
    ]


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--last must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--last must be greater than 0")
    return parsed


def _safe_redis_display_url(redis_url: str) -> str:
    parsed = urlsplit(redis_url)
    if not parsed.scheme or not parsed.hostname:
        return redis_url

    host = parsed.hostname
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, "", "", ""))


def build_json_payload(
    summaries: list[TrackSummary],
    camera_id: str,
    redis_url: str,
) -> dict[str, Any]:
    return {
        "camera_id": camera_id,
        "redis_url": _safe_redis_display_url(redis_url),
        "tracks": [summary.to_dict(include_events=True) for summary in summaries],
    }


def render_text(
    summaries: list[TrackSummary],
    camera_id: str,
    redis_url: str,
    show_event_rows: bool,
) -> str:
    host = _safe_redis_display_url(redis_url).replace("redis://", "")
    lines = [f"Active tracks in {camera_id} (Redis @ {host})", ""]

    if not summaries:
        lines.append("No active tracks found.")
        return "\n".join(lines)

    for summary in summaries:
        lines.extend(
            [
                f"Track #{summary.track_id}",
                f"events: {summary.event_count}",
                f"dwell: {summary.dwell_time_seconds:.1f}s",
                f"zone: {summary.zone}",
                f"summary: {summary.action_summary}",
            ]
        )

        if show_event_rows:
            lines.append("event rows:")
            if summary.events:
                for event in summary.events:
                    lines.append(
                        "  - frame {frame}: {event} dwell={dwell:.1f}s zones={zones}".format(
                            frame=event.get("frame_id", "?"),
                            event=event.get("event", "UNKNOWN"),
                            dwell=float(event.get("dwell_time_seconds", 0.0)),
                            zones=", ".join(event.get("zones_present") or []) or "unknown",
                        )
                    )
            else:
                lines.append("  - none")
        lines.append("")

    return "\n".join(lines).rstrip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Eagle track records stored in Redis.")
    parser.add_argument(
        "--redis-url",
        default=os.getenv("REDIS_URL", DEFAULT_REDIS_URL),
        help="Redis URL. Defaults to REDIS_URL or redis://localhost:6379.",
    )
    parser.add_argument(
        "--camera",
        default=os.getenv("CAMERA_ID", DEFAULT_CAMERA_ID),
        help="Camera id to inspect. Defaults to cam_01.",
    )
    parser.add_argument("--track", type=int, help="Only inspect one track id.")
    parser.add_argument("--last", type=_positive_int, help="Limit event rows to the last N events.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    redis_client = redis.from_url(args.redis_url)
    summaries = inspect_tracks(
        redis_client,
        camera_id=args.camera,
        track_id=args.track,
        last=args.last,
    )

    if args.json:
        print(json.dumps(build_json_payload(summaries, args.camera, args.redis_url), indent=2))
        return

    print(
        render_text(
            summaries,
            camera_id=args.camera,
            redis_url=args.redis_url,
            show_event_rows=args.track is not None,
        )
    )


if __name__ == "__main__":
    main()
