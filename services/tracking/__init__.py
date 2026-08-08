"""services.tracking — Detection + tracking layer.

Heavy dependencies (cv2, ultralytics) are imported lazily by the individual
modules so that lightweight consumers (tests, memory service) can import
sub-modules like ``cross_camera_reid`` without pulling in the full stack.
"""
from importlib import import_module
from types import ModuleType

__all__: list[str] = []


def __getattr__(name: str) -> ModuleType:
    if name == "tracker":
        return import_module("services.tracking.tracker")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
