"""Top-level service package.

Subpackages are intentionally not imported eagerly so focused test suites do
not need optional dependencies from unrelated services.
"""

__all__ = ["tracking", "memory", "detection", "reasoning"]
