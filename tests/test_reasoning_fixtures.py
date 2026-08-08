"""
Tests for shared ReasoningResult fixtures.

Round-trip tests validate that model_dump() -> model_validate() preserves
all scalar fields correctly.

vlm_captions is asserted with membership checks (not positional index)
because model_dump() / future schema migrations do not guarantee list order.

  CORRECT:   assert "Person reaching toward keypad" in result.vlm_captions
  INCORRECT: assert result.vlm_captions[0] == "Person reaching toward keypad"
"""
from libs.schemas.reasoning import ReasoningResult

from tests.fixtures.reasoning import (
    SUSPICIOUS_RESULT,
    NORMAL_RESULT,
    LOW_CONF_RESULT,
    EMPTY_CAPTIONS_RESULT,
)


def _assert_scalar_fields_match(original: ReasoningResult, validated: ReasoningResult) -> None:
    """Compare every field except vlm_captions, which is checked separately."""
    assert validated.track_id       == original.track_id
    assert validated.camera_id      == original.camera_id
    assert validated.label          == original.label
    assert validated.confidence     == original.confidence
    assert validated.reason         == original.reason
    assert validated.key_signal     == original.key_signal
    assert validated.timestamp_ms   == original.timestamp_ms
    assert validated.severity_score == original.severity_score
    assert validated.alert_id       == original.alert_id


def _assert_captions_membership(original: ReasoningResult, validated: ReasoningResult) -> None:
    """
    Assert that the round-tripped captions contain exactly the same items
    as the original, without relying on positional order.
    """
    assert len(validated.vlm_captions) == len(original.vlm_captions)

    for caption in original.vlm_captions:
        assert caption in validated.vlm_captions, (
            f"Caption missing after round-trip: {caption!r}"
        )


def test_suspicious_result_round_trip():
    validated = ReasoningResult.model_validate(
        SUSPICIOUS_RESULT.model_dump()
    )

    _assert_scalar_fields_match(SUSPICIOUS_RESULT, validated)
    _assert_captions_membership(SUSPICIOUS_RESULT, validated)

    assert "Person reaching toward keypad"  in validated.vlm_captions
    assert "Person pressing keypad buttons" in validated.vlm_captions


def test_normal_result_round_trip():
    validated = ReasoningResult.model_validate(
        NORMAL_RESULT.model_dump()
    )

    _assert_scalar_fields_match(NORMAL_RESULT, validated)
    _assert_captions_membership(NORMAL_RESULT, validated)

    assert "Person walking through lobby"     in validated.vlm_captions
    assert "No abnormal interaction detected" in validated.vlm_captions


def test_low_conf_result_round_trip():
    validated = ReasoningResult.model_validate(
        LOW_CONF_RESULT.model_dump()
    )

    _assert_scalar_fields_match(LOW_CONF_RESULT, validated)
    _assert_captions_membership(LOW_CONF_RESULT, validated)

    assert "Person standing near access panel" in validated.vlm_captions


def test_empty_captions_result_round_trip():
    validated = ReasoningResult.model_validate(
        EMPTY_CAPTIONS_RESULT.model_dump()
    )

    _assert_scalar_fields_match(EMPTY_CAPTIONS_RESULT, validated)

    assert validated.vlm_captions == []