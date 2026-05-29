from typing import List
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime
import base64


# Request validation: incoming feedback from POST /feedback
class FeedbackRequest(BaseModel):
    alert_id: str = Field(
        ...,
        min_length=1,
        description="Unique alert identifier"
    )

    track_id: int = Field(
        ...,
        gt=0,
        description="Track ID from detection"
    )

    caption_sequence: List[str] = Field(
        ...,
        min_items=1,
        description="Sequence of captions"
    )

    original_label: str = Field(
        ...,
        min_length=1,
        description="Original model prediction"
    )

    human_label: str = Field(
        ...,
        min_length=1,
        description="Human correction label"
    )

    human_note: str = Field(
        "",
        description="Human explanation for correction"
    )

    frame_b64: str = Field(
        ...,
        min_length=1,
        description="Base64 encoded frame image"
    )

    @field_validator("frame_b64", mode="after")
    @classmethod
    def validate_base64(cls, v):
        """Validate that frame_b64 is a valid base64 string."""
        try:
            base64.b64decode(v, validate=True)
        except Exception:
            raise ValueError("Invalid base64 string")
        return v


# Redis storage schema: what gets persisted
class FeedbackRecord(BaseModel):
    model_config = ConfigDict(ser_json_timedelta="float")
    
    alert_id: str
    track_id: int
    caption_sequence: List[str]
    original_label: str
    human_label: str
    human_note: str
    frame_b64: str
    timestamp: datetime


# LLaVA format schema: what gets exported for fine-tuning
class Conversation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    from_: str = Field(..., alias="from")
    value: str


class LLaVAConversation(BaseModel):
    image: str = Field(
        ...,
        description="Image filename"
    )

    conversations: List[Conversation] = Field(
        ...,
        min_items=1
    )

    @field_validator("conversations", mode="after")
    @classmethod
    def validate_conversations(cls, v):
        """Validate conversations alternate between human/system and gpt roles."""
        roles = [conv.from_ for conv in v]

        if not roles or roles[0] not in ["human", "system"]:
            raise ValueError(
                "First conversation must be human or system"
            )

        for i in range(1, len(roles)):
            if roles[i] == roles[i - 1]:
                raise ValueError(
                    "Conversations must alternate roles"
                )

        return v