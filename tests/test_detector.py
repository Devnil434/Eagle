import pytest
from pydantic import ValidationError
from libs.schemas.detection import BoundingBox, DetectionSchema, DetectionFrameSchema


def test_bounding_box_center():
    box = BoundingBox(x1=0, y1=0, x2=4, y2=6)
    assert box.center == (2.0, 3.0)


def test_bounding_box_area():
    box = BoundingBox(x1=0, y1=0, x2=4, y2=5)
    assert box.area == 20.0


def test_detection_schema_invalid_confidence():
    with pytest.raises(ValidationError):
        DetectionSchema(
            label="person",
            confidence=1.5,
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            center=(5.0, 5.0)
        )


def test_detection_frame_schema_stores_detections():
    det = DetectionSchema(
        label="car",
        confidence=0.9,
        bbox=BoundingBox(x1=0, y1=0, x2=50, y2=50),
        center=(25.0, 25.0)
    )
    frame = DetectionFrameSchema(frame_id=1, detections=[det])
    assert len(frame.detections) == 1
    assert frame.detections[0].label == "car"


def test_detection_frame_schema_empty_default():
    frame = DetectionFrameSchema(frame_id=1)
    assert frame.detections == []