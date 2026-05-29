from __future__ import annotations

from libs.schemas.workflow import FailureCategory


def classify_failure(error: Exception) -> FailureCategory:
    """
    Lightweight exception classification helper.
    """

    message = str(error).lower()

    if "timeout" in message:
        return FailureCategory.TIMEOUT

    if "connection" in message or "network" in message:
        return FailureCategory.NETWORK

    if "api" in message:
        return FailureCategory.API_FAILURE

    if "invalid" in message:
        return FailureCategory.INVALID_RESPONSE

    return FailureCategory.UNKNOWN