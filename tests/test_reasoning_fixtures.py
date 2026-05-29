from libs.schemas.reasoning import ReasoningResult

from tests.fixtures.reasoning import (
    SUSPICIOUS_RESULT,
    NORMAL_RESULT,
    LOW_CONF_RESULT,
    EMPTY_CAPTIONS_RESULT,
)


def test_suspicious_result_round_trip():
    validated = ReasoningResult.model_validate(
        SUSPICIOUS_RESULT.model_dump()
    )

    assert validated == SUSPICIOUS_RESULT


def test_normal_result_round_trip():
    validated = ReasoningResult.model_validate(
        NORMAL_RESULT.model_dump()
    )

    assert validated == NORMAL_RESULT


def test_low_conf_result_round_trip():
    validated = ReasoningResult.model_validate(
        LOW_CONF_RESULT.model_dump()
    )

    assert validated == LOW_CONF_RESULT


def test_empty_captions_result_round_trip():
    validated = ReasoningResult.model_validate(
        EMPTY_CAPTIONS_RESULT.model_dump()
    )

    assert validated == EMPTY_CAPTIONS_RESULT