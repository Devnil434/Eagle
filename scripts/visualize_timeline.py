from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Future integration point:
# from libs.schemas.memory import TrackSequence


@dataclass(frozen=True)
class TrackSequenceEvent:
    track_id: int
    start_seconds: float
    end_seconds: float
    zone: str
    action_hint: str | None = None


ZONE_COLORS = {
    "restricted": "#e00f0f",
    "safe": "#0b920f",
    "unknown": "#6c757d",
}

ACTION_MARKERS = {
    "ZONE_ENTRY": "⚡",
    "NEAR_KEYPAD": "🔑",
    "LINGERING": "⏳",
}


def _zone_color(zone: str) -> str:
    return ZONE_COLORS.get(zone.lower(), ZONE_COLORS["unknown"])


def render_timeline(events: Iterable[TrackSequenceEvent], output_file: Path) -> Path:
    events = list(events)
    if not events:
        raise ValueError("No timeline events provided")

    track_ids = sorted({event.track_id for event in events})
    y_positions = {track_id: index for index, track_id in enumerate(track_ids)}

    fig, ax = plt.subplots(figsize=(12, 5), dpi=100)

    for event in events:
        y = y_positions[event.track_id]

        ax.barh(
            y,
            event.end_seconds - event.start_seconds,
            left=event.start_seconds,
            height=0.6,
            color=_zone_color(event.zone),
            edgecolor="black",
        )

        if event.action_hint:
            marker = ACTION_MARKERS.get(event.action_hint, "•")
            ax.text(
                event.start_seconds + 0.5 * (event.end_seconds - event.start_seconds),
                y,
                marker,
                ha="center",
                va="center",
                fontsize=14,
                color="black",
                fontweight="bold",
            )

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([f"track_{tid}" for tid in track_ids])
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Track Timeline")
    ax.grid(axis="x", linestyle="--", linewidth=0.8)
    ax.invert_yaxis()

    legend_handles = [
        plt.Line2D([0], [0], color=_zone_color("restricted"), lw=10),
        plt.Line2D([0], [0], color=_zone_color("safe"), lw=10),
    ]
    ax.legend(legend_handles, ["restricted zone", "safe zone"], loc="lower right")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, format="png", bbox_inches="tight")
    plt.close(fig)
    return output_file


def create_synthetic_track_sequence(track_count: int = 5) -> List[TrackSequenceEvent]:
    events: List[TrackSequenceEvent] = []

    for track_id in range(1, track_count + 1):
        base = (track_id - 1) * 9.0

        events.extend(
            [
                TrackSequenceEvent(
                    track_id=track_id,
                    start_seconds=base,
                    end_seconds=base + 3.0,
                    zone="safe",
                    action_hint="ZONE_ENTRY",
                ),
                TrackSequenceEvent(
                    track_id=track_id,
                    start_seconds=base + 3.0,
                    end_seconds=base + 6.5,
                    zone="restricted",
                    action_hint="NEAR_KEYPAD" if track_id % 2 == 0 else None,
                ),
                TrackSequenceEvent(
                    track_id=track_id,
                    start_seconds=base + 6.5,
                    end_seconds=base + 9.0,
                    zone="safe",
                    action_hint="LINGERING",
                ),
            ]
        )

    return events


def main() -> int:
    parser = argparse.ArgumentParser(description="Render TrackSequence timeline")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Currently using synthetic data (required by acceptance criteria)
    events = create_synthetic_track_sequence(track_count=5)

    output_path = Path(args.output)
    render_timeline(events, output_path)

    print(f"Saved timeline PNG to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
