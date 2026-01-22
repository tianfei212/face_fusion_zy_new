from __future__ import annotations

import time
import logging

from fastapi import APIRouter, Body
from pydantic import BaseModel
from typing import List

from .manager import get_ai_manager
from .protocol import CommandType
from .resources import list_assets

logger = logging.getLogger("blankend.ai_core.api")


class AiProcessRequest(BaseModel):
    enable: bool | None = None
    mode: str | None = None
    portrait_id: str | None = None
    scene_id: str | None = None
    similarity: float | None = None
    sharpness: float | None = None


class AiStatusResponse(BaseModel):
    status: dict


class AiCommandRequest(BaseModel):
    type: str
    payload: object | None = None
    worker_id: int | None = None


class AiCommandResponse(BaseModel):
    ok: bool
    request_id: str | None = None
    active_worker_id: int | None = None
    standby_worker_id: int | None = None
    status: dict | None = None


def get_ai_core_router() -> APIRouter:
    router = APIRouter(prefix="/api/v1/ai", tags=["ai_core"])
    mgr = get_ai_manager()

    @router.get("/status", response_model=AiStatusResponse)
    async def status() -> AiStatusResponse:
        return AiStatusResponse(status=mgr.status())

    @router.post("/process", response_model=AiStatusResponse)
    async def process(req: AiProcessRequest) -> AiStatusResponse:
        logger.info(
            "api_process_recv enable=%s mode=%s portrait_id=%s scene_id=%s similarity=%s sharpness=%s",
            req.enable,
            req.mode,
            req.portrait_id,
            req.scene_id,
            req.similarity,
            req.sharpness,
        )
        similarity_threshold = None
        if req.similarity is not None:
            v = float(req.similarity)
            if v > 1.0:
                v = v / 100.0
            similarity_threshold = float(max(0.0, min(1.0, v)))
        portrait = req.portrait_id
        scene = req.scene_id
        if req.enable is not None:
            mgr.configure(enabled=bool(req.enable))
        if similarity_threshold is not None:
            mgr.configure(similarity_threshold=float(similarity_threshold))
        if portrait is not None:
            mgr.set_portrait(portrait)
        if scene is not None:
            mgr.set_scene(scene)
        st = mgr.status()
        logger.info(
            "api_process_done enabled=%s active_worker_id=%s workers=%s",
            st.get("enabled"),
            st.get("active_worker_id"),
            (st.get("workers") or {}),
        )
        return AiStatusResponse(status=st)

    @router.post("/frame/{client_id}")
    async def submit_frame(client_id: str, body: bytes = Body(...)) -> dict:
        ok = mgr.submit_frame(client_id.encode(), body, time.time())
        return {"ok": ok}

    @router.get("/assets/list", response_model=List[str])
    async def get_asset_list(type: str) -> List[str]:
        return list_assets(type)

    @router.post("/command", response_model=AiCommandResponse)
    async def command(req: AiCommandRequest) -> AiCommandResponse:
        t = str(req.type or "").strip()
        try:
            ct = CommandType(t)
        except Exception:
            ct = None
        if ct is None:
            logger.info("api_command_reject type=%s payload=%s worker_id=%s", req.type, req.payload, req.worker_id)
            return AiCommandResponse(ok=False, status=mgr.status())

        logger.info("api_command_recv type=%s payload=%s worker_id=%s", ct.value, req.payload, req.worker_id)

        if ct in {CommandType.SET_FACE, CommandType.SET_FACE_IMAGE}:
            ok = mgr.set_portrait(str(req.payload or ""))
            logger.info("api_command_done type=%s ok=%s", ct.value, ok)
            return AiCommandResponse(ok=ok, status=mgr.status())
        if ct in {CommandType.SET_BG, CommandType.SET_BACKGROUND}:
            ok = mgr.set_scene(str(req.payload or ""))
            logger.info("api_command_done type=%s ok=%s", ct.value, ok)
            return AiCommandResponse(ok=ok, status=mgr.status())
        if ct in {CommandType.LOAD_DFM, CommandType.LOAD_MODEL, CommandType.SET_FACE_DFM}:
            request_id = mgr.load_dfm(str(req.payload or ""))
            st = mgr.status()
            standby_id = st.get("active_worker_id")
            if isinstance(standby_id, int):
                standby_id = 1 - int(standby_id)
            logger.info(
                "api_command_done type=%s request_id=%s active_worker_id=%s standby_worker_id=%s",
                ct.value,
                request_id,
                st.get("active_worker_id"),
                standby_id,
            )
            return AiCommandResponse(
                ok=True,
                request_id=request_id,
                active_worker_id=st.get("active_worker_id"),
                standby_worker_id=standby_id if isinstance(standby_id, int) else None,
                status=st,
            )
        if ct == CommandType.ACTIVATE:
            active = mgr.activate(req.worker_id)
            logger.info("api_command_done type=%s active_worker_id=%s", ct.value, active)
            return AiCommandResponse(ok=True, active_worker_id=active, status=mgr.status())
        if ct == CommandType.UPDATE_CONFIG:
            ok = mgr.update_config(req.payload if isinstance(req.payload, dict) else {})
            logger.info("api_command_done type=%s ok=%s", ct.value, ok)
            return AiCommandResponse(ok=ok, status=mgr.status())
        if ct == CommandType.UNLOAD_MODEL:
            ok = mgr.unload_model()
            logger.info("api_command_done type=%s ok=%s", ct.value, ok)
            return AiCommandResponse(ok=ok, status=mgr.status())
        return AiCommandResponse(ok=False, status=mgr.status())

    return router
