from __future__ import annotations

import time

from fastapi import APIRouter, Body
from pydantic import BaseModel

from .service import get_realtime_pipeline_service


class PipelineStatusResponse(BaseModel):
    status: dict


class PipelineConfigRequest(BaseModel):
    enabled: bool | None = None
    force_cuda: bool | None = None
    latest_frame_only: bool | None = None
    output_jpeg_quality: int | None = None
    depth_fusion: bool | None = None
    depth_threshold: float | None = None
    depth_alpha_boost: float | None = None
    background_mode: str | None = None
    background_jpeg_path: str | None = None
    max_fps: float | None = None
    auto_degrade: bool | None = None
    degrade_depth_if_slow_ms: float | None = None


class RecordingStartRequest(BaseModel):
    client_id: str
    format: str = "mp4"
    fps: int = 30
    out_path: str | None = None


class RecordingStopRequest(BaseModel):
    client_id: str


class RecordingResponse(BaseModel):
    ok: bool
    path: str | None = None


def get_realtime_pipeline_router() -> APIRouter:
    router = APIRouter(prefix="/api/v1/realtime_pipeline", tags=["realtime_pipeline"])
    svc = get_realtime_pipeline_service()

    @router.get("/status", response_model=PipelineStatusResponse)
    async def status() -> PipelineStatusResponse:
        return PipelineStatusResponse(status=svc.status())

    @router.post("/config", response_model=PipelineStatusResponse)
    async def config(req: PipelineConfigRequest) -> PipelineStatusResponse:
        st = svc.configure(**req.model_dump(exclude_none=True))
        return PipelineStatusResponse(status=st)

    @router.post("/record/start", response_model=RecordingResponse)
    async def record_start(req: RecordingStartRequest) -> RecordingResponse:
        path = svc.start_recording(req.client_id.encode(), fmt=req.format, fps=req.fps, out_path=req.out_path)
        return RecordingResponse(ok=True, path=path)

    @router.post("/record/stop", response_model=RecordingResponse)
    async def record_stop(req: RecordingStopRequest) -> RecordingResponse:
        path = svc.stop_recording(req.client_id.encode())
        return RecordingResponse(ok=path is not None, path=path)

    @router.post("/frame/{client_id}", response_model=RecordingResponse)
    async def submit_frame(client_id: str, body: bytes = Body(...)) -> RecordingResponse:
        ok = svc.submit(client_id.encode(), body, time.time())
        return RecordingResponse(ok=ok, path=None)

    return router
