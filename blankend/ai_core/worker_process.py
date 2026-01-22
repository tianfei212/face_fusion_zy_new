from __future__ import annotations

import logging
import os
import queue
import time
from dataclasses import dataclass
from typing import Any

from .engine import UnifiedInferenceEngine
from .protocol import CommandType, WorkerCommand

logger = logging.getLogger("blankend.ai_core.worker_process")


@dataclass
class FrameJob:
    client_id: bytes
    seq: int
    ts: float
    jpeg: bytes


@dataclass
class WorkerOutput:
    kind: str
    worker_id: int
    ts: float
    client_id: bytes | None = None
    seq: int | None = None
    jpeg: bytes | None = None
    diag: dict[str, Any] | None = None


def worker_main(
    worker_id: int,
    device_id: int,
    cmd_q,
    frame_q,
    out_q,
) -> None:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)

    def emit(kind: str, diag: dict[str, Any] | None = None, **kwargs: Any) -> None:
        try:
            out_q.put_nowait(
                WorkerOutput(
                    kind=kind,
                    worker_id=int(worker_id),
                    ts=time.time(),
                    diag=diag or {},
                    **kwargs,
                )
            )
        except Exception:
            pass

    try:
        from .model_config import apply_models_env_from_config, log_ai_model_config

        cfg = apply_models_env_from_config()
        log_ai_model_config(cfg, prefix="startup_worker_models")
    except Exception:
        pass

    simple_forward = os.getenv("SIMPLE_FORWARD") == "1" or os.getenv("AI_WORKER_SIMPLE_FORWARD") == "1"
    if simple_forward:
        logger.warning(
            "worker_simple_forward_enabled worker_id=%s device_id=%s SIMPLE_FORWARD=%s AI_WORKER_SIMPLE_FORWARD=%s",
            int(worker_id),
            int(device_id),
            os.getenv("SIMPLE_FORWARD"),
            os.getenv("AI_WORKER_SIMPLE_FORWARD"),
        )
    engine: UnifiedInferenceEngine | None = None
    if not simple_forward:
        try:
            t0 = time.perf_counter()
            logger.info("engine_init_start worker_id=%s device_id=%s", int(worker_id), int(device_id))
            engine = UnifiedInferenceEngine(device_id=int(device_id), lazy_init=False)
            cost_ms = int((time.perf_counter() - t0) * 1000)
            providers = getattr(engine, "active_providers", None)
            logger.info(
                "engine_init_done worker_id=%s device_id=%s cost_ms=%s providers=%s",
                int(worker_id),
                int(device_id),
                cost_ms,
                providers,
            )
            dfm_model = os.getenv("DFM_MODEL")
            if dfm_model:
                t0 = time.perf_counter()
                ok = bool(engine.load_dfm_model(dfm_model))
                cost_ms = int((time.perf_counter() - t0) * 1000)
                logger.info(
                    "dfm_preload_done worker_id=%s device_id=%s ok=%s cost_ms=%s model=%s",
                    int(worker_id),
                    int(device_id),
                    ok,
                    cost_ms,
                    dfm_model,
                )
            emit(
                "status",
                diag={
                    "engine_ready": True,
                    "device_id": int(device_id),
                    "simple_forward": bool(simple_forward),
                    "providers": getattr(engine, "active_providers", {}),
                },
            )
        except Exception as e:
            logger.exception(
                "engine_init_failed worker_id=%s device_id=%s err=%s",
                int(worker_id),
                int(device_id),
                e,
            )
            engine = None
            simple_forward = True

    active_flag = False
    loaded_dfm: str | None = None
    last_error: str | None = None

    emit("status", diag={"active": active_flag, "device_id": int(device_id), "simple_forward": simple_forward})

    while True:
        try:
            while True:
                cmd: WorkerCommand = cmd_q.get_nowait()
                logger.info(
                    "worker_cmd_recv worker_id=%s device_id=%s type=%s request_id=%s payload=%s",
                    int(worker_id),
                    int(device_id),
                    cmd.type.value,
                    cmd.request_id,
                    cmd.payload,
                )
                if cmd.type in {CommandType.SET_FACE, CommandType.SET_FACE_IMAGE}:
                    ok = bool(engine.set_portrait(str(cmd.payload))) if engine else False
                    logger.info(
                        "worker_cmd_done worker_id=%s device_id=%s type=%s ok=%s",
                        int(worker_id),
                        int(device_id),
                        cmd.type.value,
                        ok,
                    )
                    emit("status", diag={"set_face": ok})
                elif cmd.type in {CommandType.SET_BG, CommandType.SET_BACKGROUND}:
                    ok = bool(engine.set_scene(str(cmd.payload))) if engine else False
                    logger.info(
                        "worker_cmd_done worker_id=%s device_id=%s type=%s ok=%s",
                        int(worker_id),
                        int(device_id),
                        cmd.type.value,
                        ok,
                    )
                    emit("status", diag={"set_bg": ok})
                elif cmd.type in {CommandType.LOAD_DFM, CommandType.LOAD_MODEL, CommandType.SET_FACE_DFM}:
                    name = str(cmd.payload) if cmd.payload is not None else ""
                    t0 = time.perf_counter()
                    ok = bool(engine.load_dfm_model(name)) if engine else False
                    cost_ms = int((time.perf_counter() - t0) * 1000)
                    logger.info(
                        "worker_dfm_load worker_id=%s device_id=%s ok=%s cost_ms=%s model=%s",
                        int(worker_id),
                        int(device_id),
                        ok,
                        cost_ms,
                        name,
                    )
                    if ok:
                        loaded_dfm = name
                        emit("ready", diag={"request_id": cmd.request_id, "dfm": loaded_dfm})
                    else:
                        emit("error", diag={"request_id": cmd.request_id, "dfm": name, "error": "load_dfm_failed"})
                elif cmd.type == CommandType.UNLOAD_MODEL:
                    loaded_dfm = None
                    if engine:
                        engine._dfm_session = None
                        engine._use_dfm = False
                    logger.info(
                        "worker_unload_model worker_id=%s device_id=%s",
                        int(worker_id),
                        int(device_id),
                    )
                    emit("status", diag={"unload_model": bool(engine is not None)})
                elif cmd.type == CommandType.ACTIVATE:
                    active_flag = bool((cmd.payload or {}).get("active")) if isinstance(cmd.payload, dict) else True
                    logger.info(
                        "worker_activate_flag worker_id=%s device_id=%s active=%s",
                        int(worker_id),
                        int(device_id),
                        bool(active_flag),
                    )
                    emit("status", diag={"active": active_flag})
                elif cmd.type == CommandType.UPDATE_CONFIG:
                    if engine and isinstance(cmd.payload, dict):
                        for k, v in cmd.payload.items():
                            if hasattr(engine.cfg, k):
                                if k == "matting_enabled":
                                    v = False
                                setattr(engine.cfg, k, v)
                    logger.info(
                        "worker_update_config worker_id=%s device_id=%s keys=%s",
                        int(worker_id),
                        int(device_id),
                        (list(cmd.payload.keys()) if isinstance(cmd.payload, dict) else []),
                    )
                    emit("status", diag={"config_updated": True})
        except queue.Empty:
            pass
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            emit("error", diag={"error": last_error})

        try:
            job: FrameJob = frame_q.get(timeout=0.2)
        except queue.Empty:
            continue
        except Exception:
            time.sleep(0.01)
            continue

        if simple_forward or not active_flag:
            if not active_flag and job.seq % 60 == 0:
                 logger.info("worker_inactive_bypass seq=%s device_id=%s", job.seq, device_id)
            
            emit(
                "frame",
                client_id=job.client_id,
                seq=job.seq,
                jpeg=job.jpeg,
                diag={"device_id": int(device_id), "mode": "simple_forward" if simple_forward else "inactive"},
            )
            continue

        if engine is None:
            emit("error", client_id=job.client_id, seq=job.seq, diag={"error": "engine_missing"})
            continue

        try:
            out_jpeg, diag = engine.process_frame(job.client_id, job.seq, job.jpeg)
            if not out_jpeg:
                raise RuntimeError("empty_output")
            diag = dict(diag or {})
            diag.update({"device_id": int(device_id), "active": active_flag, "dfm": loaded_dfm})
            emit("frame", client_id=job.client_id, seq=job.seq, jpeg=out_jpeg, diag=diag)
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            emit("error", client_id=job.client_id, seq=job.seq, diag={"error": last_error})
