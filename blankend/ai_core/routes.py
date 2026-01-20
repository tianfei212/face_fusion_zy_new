from __future__ import annotations

import time

from fastapi import APIRouter, Body
from pydantic import BaseModel

from .service import get_ai_core_service


class AiProcessRequest(BaseModel):
    enable: bool | None = None
    mode: str | None = None
    portrait_id: str | None = None
    scene_id: str | None = None
    similarity: float | None = None
    sharpness: float | None = None


class AiStatusResponse(BaseModel):
    status: dict


def get_ai_core_router() -> APIRouter:
    router = APIRouter(prefix="/api/v1/ai", tags=["ai_core"])
    svc = get_ai_core_service()

    @router.get("/status", response_model=AiStatusResponse)
    async def status() -> AiStatusResponse:
        return AiStatusResponse(status=svc.status())

    @router.post("/process", response_model=AiStatusResponse)
    async def process(req: AiProcessRequest) -> AiStatusResponse:
        similarity_threshold = None
        if req.similarity is not None:
            v = float(req.similarity)
            if v > 1.0:
                v = v / 100.0
            similarity_threshold = float(max(0.0, min(1.0, v)))
        portrait = req.portrait_id
        scene = req.scene_id
        st = svc.configure(
            enabled=req.enable,
            similarity_threshold=similarity_threshold,
            portrait_name=portrait,
            scene_name=scene,
        )
        return AiStatusResponse(status=st)

    @router.post("/frame/{client_id}")
    async def submit_frame(client_id: str, body: bytes = Body(...)) -> dict:
        ok = svc.submit(client_id.encode(), body, time.time())
        return {"ok": ok}

    return router
