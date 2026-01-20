from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .gpu import is_gpu_usable, probe_gpu_status, select_onnx_execution_providers
from .codec import probe_hw_codecs
from .service import get_video_processing_service


class GpuStatusResponse(BaseModel):
    ts: float = Field(..., description="Unix timestamp (seconds)")
    gpu_available: bool = Field(..., description="Whether GPU mode is usable now")
    onnxruntime: dict = Field(..., description="onnxruntime install/providers summary")
    cuda: dict = Field(..., description="CUDA/cuDNN detected versions")
    metrics: dict = Field(..., description="GPU metrics (memory, temperature, utilization)")


class GpuUsableResponse(BaseModel):
    usable: bool
    reason: str


class ExecutionProvidersResponse(BaseModel):
    mode: str = Field(..., description="gpu|cpu (selected execution mode)")
    providers: list[str] = Field(..., description="onnxruntime execution providers order")
    reason: str = Field(..., description="selection reason or fallback reason")


class PipelineConfigRequest(BaseModel):
    enabled: bool | None = None
    queue_size: int | None = None
    decode_workers: int | None = None
    encode_workers: int | None = None
    reencode_enabled: bool | None = None
    jpeg_quality: int | None = None
    heartbeat_timeout_s: float | None = None


class PipelineStatusResponse(BaseModel):
    status: dict


class CodecStatusResponse(BaseModel):
    status: dict


def get_video_processing_router() -> APIRouter:
    router = APIRouter(prefix="/api/v1/video_processing", tags=["video_processing"])

    @router.get("/gpu/status", response_model=GpuStatusResponse)
    async def get_gpu_status() -> GpuStatusResponse:
        return GpuStatusResponse(**probe_gpu_status())

    @router.get("/gpu/usable", response_model=GpuUsableResponse)
    async def get_gpu_usable() -> GpuUsableResponse:
        ok, reason = is_gpu_usable()
        return GpuUsableResponse(usable=ok, reason=reason)

    @router.get("/gpu/providers", response_model=ExecutionProvidersResponse)
    async def get_execution_providers(prefer_gpu: bool = True) -> ExecutionProvidersResponse:
        return ExecutionProvidersResponse(**select_onnx_execution_providers(prefer_gpu=prefer_gpu))

    @router.get("/pipeline/status", response_model=PipelineStatusResponse)
    async def get_pipeline_status() -> PipelineStatusResponse:
        svc = get_video_processing_service()
        return PipelineStatusResponse(status=svc.status())

    @router.post("/pipeline/config", response_model=PipelineStatusResponse)
    async def update_pipeline_config(req: PipelineConfigRequest) -> PipelineStatusResponse:
        svc = get_video_processing_service()
        data = req.model_dump(exclude_none=True)
        if data:
            svc.configure(**data)
        return PipelineStatusResponse(status=svc.status())

    @router.get("/codec/status", response_model=CodecStatusResponse)
    async def get_codec_status() -> CodecStatusResponse:
        return CodecStatusResponse(status=probe_hw_codecs())

    return router
