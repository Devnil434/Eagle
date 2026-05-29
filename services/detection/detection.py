"""
detector.py — YOLOv8/v9 frame-level object detection.

Usage (CLI):
    python detector.py --source data/sample_videos/sample.mp4
    python detector.py --source 0                # webcam

Usage (API):
    from services.detection.detector import Detector
    detector = Detector()
    results = detector.detect(frame)
"""
from __future__ import annotations
import argparse
import logging

import cv2
import numpy as np
from ultralytics import YOLO

from libs.schemas.detection import DetectionFrameSchema as DetectionFrame, DetectionSchema as Detection, BoundingBox
from services.detection.zones import get_zones, get_zones_for_point
from services.reasoning.scene_graph import SceneGraphBuilder
from services.reasoning.prompts import build_reasoning_prompt
from dataclasses import dataclass
from typing import List, Tuple

from libs.config.settings import settings


@dataclass
class Detection:
    label: str
    bbox: List[float]
    confidence: float
    center: Tuple[float, float]
    zones_present: List[str]


@dataclass
class DetectionFrame:
    frame_id: int
    detections: List[Detection]
    timestamp_ms: float
    camera_id: str = "cam_01"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Detector Class ──────────────────────────────────────────────────────────

class Detector:
    """Wraps a YOLO model for frame-by-frame inference."""

    PERSON_CLASS_ID = 0     # COCO class ID for 'person'
    TARGET_LABELS = {       # labels to pass downstream (filter noise)
        "person", "backpack", "handbag", "cell phone", "laptop"
    }

    def __init__(
        self,
        model_name: str = settings.detector_model,
        confidence_threshold: float = settings.detection_confidence_threshold,
        device: str = settings.detector_device,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, got {confidence_threshold}"
            )
        logger.info(f"Loading YOLO model: {model_name} on {device}")
        self.model = YOLO(model_name)
        self.conf = confidence_threshold
        self.device = device

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> DetectionFrame:
        """
        Run YOLO inference on a single BGR frame.

        Args:
            frame:    BGR image as numpy array (H, W, 3).
            frame_id: Frame index for downstream tracking.

        Returns:
            DetectionFrame with all detected objects and zone memberships.
        """
        results = self.model(frame, device=self.device, verbose=False)
        detections: list[Detection] = []

        active_zones = get_zones()

        for box, conf, cls_id in zip(
            results[0].boxes.xyxy.cpu().numpy(),
            results[0].boxes.conf.cpu().numpy(),
            results[0].boxes.cls.cpu().numpy(),
        ):
            label = self.model.names[int(cls_id)]
            if label not in self.TARGET_LABELS:
                continue

            if float(conf) < self.conf:
                logger.debug(
                    f"Dropped detection: class={label}, conf={float(conf):.2f} "
                    f"(below threshold {self.conf})"
                )
                continue

            x1, y1, x2, y2 = box.tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            zones = [z.name for z in get_zones_for_point(cx, cy, zones=active_zones)]

            detections.append(Detection(
                label=label,
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                confidence=float(conf),
                center=(cx, cy),
                zones_present=zones,
            ))

        return DetectionFrame(
            frame_id=frame_id,
            detections=detections,
            timestamp_ms=cv2.getTickCount() / cv2.getTickFrequency() * 1000,
        )


# ─── Rendering ────────────────────────────────────────────────────────────────

LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "person":    (0, 120, 255),
    "backpack":  (255, 165, 0),
    "handbag":   (255, 165, 0),
    "cell phone":(0, 200, 200),
    "laptop":    (200, 0, 200),
}

def draw_detections(frame: np.ndarray, det_frame: DetectionFrame) -> np.ndarray:
    """Draw bounding boxes, labels, and zone overlays onto frame."""
    out = frame.copy()

    active_zones = get_zones()

    # Draw zone polygons
    for zone in active_zones:
        if not getattr(zone, 'valid', True):
            continue
        pts = zone.as_array().reshape((-1, 1, 2))
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], zone.color_bgr)
        cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)
        cv2.polylines(out, [pts], isClosed=True, color=zone.color_bgr, thickness=2)
        cv2.putText(out, zone.name, zone.polygon[0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone.color_bgr, 1)

    # Draw detections
    for det in det_frame.detections:
        x1, y1, x2, y2 = int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2)
        color = LABEL_COLORS.get(det.label, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label_text = f"{det.label} {det.confidence:.2f}"
        if det.zones_present:
            label_text += f" [{', '.join(det.zones_present)}]"

        cv2.putText(out, label_text, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Centroid dot
        cv2.circle(out, (int(det.center[0]), int(det.center[1])), 4, color, -1)

    # HUD
    cv2.putText(out, f"Frame: {det_frame.frame_id} | Detections: {len(det_frame.detections)}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return out


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Agentic Vision detection demo")
    parser.add_argument("--source", default="0", help="Video file path or camera index")
    parser.add_argument("--model", default=settings.detector_model, help="YOLO model name")
    parser.add_argument("--conf", type=float, default=settings.detection_confidence_threshold, help="Confidence threshold")
    parser.add_argument("--output", default=None, help="Optional output video path")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    detector = Detector(model_name=args.model, confidence_threshold=args.conf)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Stream: {width}x{height} @ {fps:.1f} FPS")

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        det_frame = detector.detect(frame, frame_id=frame_id)
        builder = SceneGraphBuilder(det_frame)
        
        builder.build_graph()
        graph_text = builder.serialize_graph()

        if frame_id % 30 == 0 and graph_text:
            prompt = build_reasoning_prompt(graph_text)
            print("\nLLM PROMPT:\n")
            print(prompt)
        
        

        annotated  = draw_detections(frame, det_frame)

        cv2.imshow("Agentic Vision — Detection", annotated)
        if writer:
            writer.write(annotated)

        frame_id += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    logger.info(f"Processed {frame_id} frames.")


if __name__ == "__main__":
    main()