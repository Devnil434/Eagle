"""Phase 3: Temporal Memory Service.

Sub-modules are imported directly by consumers (e.g.
``from services.memory.memory import MemoryStore``) so this __init__ stays
lightweight to avoid pulling in heavy transitive deps (prometheus_client,
confluent_kafka) during test collection.
"""
__all__: list[str] = []