
import json
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


EVENT_TYPES = [
    "restricted_zone_intrusion",
    "loitering",
    "object_abandonment",
    "crowd_formation",
    "normal_movement",
]


def generate_event(
    event_type: Optional[str] = None,
    person_id: Optional[int] = None,
    timestamp: Optional[datetime] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)

    if event_type is None:
        event_type = random.choice(EVENT_TYPES)

    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unsupported event_type: {event_type}")

    if person_id is None:
        person_id = random.randint(1, 50)

    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    return {
        "person_id": person_id,
        "event_type": event_type,
        "timestamp": timestamp.isoformat(),
        "location": {
            "x": random.randint(0, 1920),
            "y": random.randint(0, 1080),
        },
        "confidence": round(random.uniform(0.65, 0.99), 2),
        "metadata": _build_metadata(event_type),
    }


def _build_metadata(event_type: str) -> Dict[str, Any]:
    if event_type == "restricted_zone_intrusion":
        return {
            "zone_id": random.choice(["restricted_lab", "server_room", "staff_only"]),
            "severity": random.choice(["medium", "high"]),
        }

    if event_type == "loitering":
        return {
            "duration_seconds": random.randint(60, 600),
            "severity": random.choice(["low", "medium"]),
        }

    if event_type == "object_abandonment":
        return {
            "object_type": random.choice(["bag", "box", "backpack", "package"]),
            "unattended_duration_seconds": random.randint(120, 900),
            "severity": random.choice(["medium", "high"]),
        }

    if event_type == "crowd_formation":
        return {
            "crowd_count": random.randint(5, 30),
            "area": random.choice(["entrance", "hallway", "lobby", "parking_lot"]),
            "severity": random.choice(["low", "medium", "high"]),
        }

    return {
        "movement_direction": random.choice(["north", "south", "east", "west"]),
        "speed": random.choice(["slow", "normal", "fast"]),
        "severity": "none",
    }


def generate_events(
    count: int = 10,
    start_time: Optional[datetime] = None,
    interval_seconds: int = 30,
) -> List[Dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be greater than 0")

    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than 0")

    if start_time is None:
        start_time = datetime.now(timezone.utc)

    return [
        generate_event(timestamp=start_time + timedelta(seconds=i * interval_seconds))
        for i in range(count)
    ]


def export_events_to_json(events: List[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(events, file, indent=2)
