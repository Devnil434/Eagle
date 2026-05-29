from pathlib import Path

from scripts.visualize_timeline import (
    create_synthetic_track_sequence,
    render_timeline,
)


def test_render_timeline_creates_png(tmp_path: Path):
    events = create_synthetic_track_sequence(track_count=5)
    output_file = tmp_path / "timeline.png"

    result_path = render_timeline(events, output_file)

    assert result_path.exists()
    assert result_path.suffix == ".png"
