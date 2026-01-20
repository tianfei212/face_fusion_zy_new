from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SkillErrorCode(str, Enum):
    INVALID_REQUEST = "INVALID_REQUEST"
    SKILL_NOT_FOUND = "SKILL_NOT_FOUND"
    SKILL_ERROR = "SKILL_ERROR"
    TIMEOUT = "TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class SkillError(BaseModel):
    code: SkillErrorCode = Field(..., description="Stable error code")
    message: str = Field(..., description="Human readable message")
    details: Any | None = Field(default=None, description="Optional structured debug details")


class SkillInvokeRequest(BaseModel):
    request_id: str = Field(..., description="Client generated idempotency key / correlation id")
    params: dict[str, Any] = Field(default_factory=dict, description="Skill specific parameters")
    timeout_ms: int | None = Field(default=None, description="Best-effort timeout in milliseconds")


class SkillInvokeResponse(BaseModel):
    request_id: str
    ok: bool
    result: Any | None = None
    error: SkillError | None = None


class SkillDescriptor(BaseModel):
    id: str
    name: str
    version: str
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None

