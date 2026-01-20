from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException

from skills.contracts import (
    SkillDescriptor,
    SkillError,
    SkillErrorCode,
    SkillInvokeRequest,
    SkillInvokeResponse,
)
from skills.registry import get_registry


def get_skills_router() -> APIRouter:
    router = APIRouter(prefix="/api/v1/skills", tags=["skills"])
    reg = get_registry()

    @router.get("", response_model=list[SkillDescriptor])
    async def list_skills() -> list[SkillDescriptor]:
        return reg.list()

    @router.post("/{skill_id}:invoke", response_model=SkillInvokeResponse)
    async def invoke_skill(skill_id: str, req: SkillInvokeRequest) -> SkillInvokeResponse:
        skill = reg.get(skill_id)
        if skill is None:
            return SkillInvokeResponse(
                request_id=req.request_id,
                ok=False,
                error=SkillError(code=SkillErrorCode.SKILL_NOT_FOUND, message=f"skill not found: {skill_id}"),
            )
        started = time.monotonic()
        try:
            result = skill.invoke(dict(req.params or {}))
            if req.timeout_ms is not None:
                elapsed_ms = (time.monotonic() - started) * 1000
                if elapsed_ms > float(req.timeout_ms):
                    return SkillInvokeResponse(
                        request_id=req.request_id,
                        ok=False,
                        error=SkillError(code=SkillErrorCode.TIMEOUT, message="timeout"),
                    )
            return SkillInvokeResponse(request_id=req.request_id, ok=True, result=result)
        except HTTPException:
            raise
        except Exception as e:
            return SkillInvokeResponse(
                request_id=req.request_id,
                ok=False,
                error=SkillError(code=SkillErrorCode.SKILL_ERROR, message=str(e)),
            )

    return router

