from libs.schemas.reasoning import ReasoningResult


SUSPICIOUS_RESULT = ReasoningResult(
    track_id=3,
    label="Suspicious",
    confidence=0.88,
    reason="Repeated keypad interaction over 22s suggests unauthorised entry attempt.",
    key_signal="near_keypad × 4 in restricted_door",
    timestamp_ms=1718000022000.0,
    vlm_captions=[
        "Person reaching toward keypad",
        "Person pressing keypad buttons",
    ],
    severity_score=0.91,
)


NORMAL_RESULT = ReasoningResult(
    track_id=7,
    label="Normal",
    confidence=0.96,
    reason="Person passed through lobby without suspicious behaviour.",
    key_signal="walking_through_lobby",
    timestamp_ms=1718000035000.0,
    vlm_captions=[
        "Person walking through lobby",
        "No abnormal interaction detected",
    ],
    severity_score=0.08,
)


LOW_CONF_RESULT = ReasoningResult(
    track_id=11,
    label="Suspicious",
    confidence=0.51,
    reason="Brief hesitation near access panel detected.",
    key_signal="near_access_panel",
    timestamp_ms=1718000041000.0,
    vlm_captions=[
        "Person standing near access panel",
    ],
    severity_score=0.42,
)


EMPTY_CAPTIONS_RESULT = ReasoningResult(
    track_id=15,
    label="Normal",
    confidence=0.73,
    reason="No useful visual captions were generated.",
    key_signal="none",
    timestamp_ms=1718000050000.0,
    vlm_captions=[],
    severity_score=0.10,
)