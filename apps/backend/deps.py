from __future__ import annotations
from services.memory.ring_buffer import MemoryStore
from services.reasoning.pipeline import ReasoningPipeline

_store: MemoryStore | None = None
_pipeline: ReasoningPipeline | None = None


def get_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


def get_pipeline() -> ReasoningPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ReasoningPipeline()
    return _pipeline
