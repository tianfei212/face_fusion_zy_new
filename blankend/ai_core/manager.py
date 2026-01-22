from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

import multiprocessing as mp

from ..video_processing.gpu import probe_gpu_status
from .protocol import CommandType, WorkerCommand
from .worker_process import FrameJob, WorkerOutput, worker_main

logger = logging.getLogger("blankend.ai_core.manager")


@dataclass
class ManagerConfig:
    enabled: bool = False
    similarity_threshold: float = 0.75
    depth_enabled: bool = True
    light_fusion_enabled: bool = True
    matting_enabled: bool = False
    jpeg_quality: int = 85
    max_inflight_per_worker: int = 8


class AiActiveStandbyManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cfg = ManagerConfig()
        self._active_worker_id = 0
        self._seq_by_client: dict[bytes, int] = {}
        self._started = False

        self._ctx = mp.get_context("spawn")
        self._procs: dict[int, mp.Process] = {}
        self._cmd_q: dict[int, Any] = {}
        self._frame_q: dict[int, Any] = {}
        self._out_q: Any | None = None

        self._sender: Callable[[bytes, bytes], None] | None = None
        self._broadcaster: Callable[[bytes], Any] | None = None

        self._stop = threading.Event()
        self._out_thread = threading.Thread(target=self._out_loop, name="ai-core-manager-out", daemon=True)

        self._worker_state: dict[int, dict[str, Any]] = {
            0: {"device_id": 0, "active": True},
            1: {"device_id": 1, "active": False},
        }

    def attach(self, sender: Callable[[bytes, bytes], None], broadcaster: Callable[[bytes], Any] | None) -> None:
        with self._lock:
            self._sender = sender
            self._broadcaster = broadcaster

    def ensure_started(self) -> None:
        with self._lock:
            started = bool(self._started)
        if started:
            return
        self.start()

    def start(self) -> None:
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            logging.basicConfig(level=logging.INFO)
        logger.setLevel(logging.INFO)

        with self._lock:
            self._started = True
            if self._out_q is None:
                self._out_q = self._ctx.Queue(maxsize=2048)
            for wid in (0, 1):
                if wid not in self._cmd_q:
                    self._cmd_q[wid] = self._ctx.Queue(maxsize=256)
                if wid not in self._frame_q:
                    self._frame_q[wid] = self._ctx.Queue(maxsize=int(self._cfg.max_inflight_per_worker))
            for wid in (0, 1):
                p = self._procs.get(wid)
                if p is not None and p.is_alive():
                    continue
                device_id = wid
                proc = self._ctx.Process(
                    target=worker_main,
                    args=(wid, device_id, self._cmd_q[wid], self._frame_q[wid], self._out_q),
                    name=f"ai-core-worker-{wid}",
                    daemon=True,
                )
                proc.start()
                self._procs[wid] = proc

            self._stop.clear()
            if not self._out_thread.is_alive():
                self._out_thread = threading.Thread(target=self._out_loop, name="ai-core-manager-out", daemon=True)
                self._out_thread.start()

            self._send_activate_flags()
            self._push_config_to_workers()

    def enabled(self) -> bool:
        with self._lock:
            return bool(self._cfg.enabled)

    def configure(self, **kwargs: Any) -> dict[str, Any]:
        self.ensure_started()
        if kwargs:
            try:
                logger.info("mgr_configure kwargs=%s", dict(kwargs))
            except Exception:
                pass
        with self._lock:
            for k, v in kwargs.items():
                if not hasattr(self._cfg, k):
                    continue
                setattr(self._cfg, k, v)
        self._push_config_to_workers()
        return self.status()

    def status(self) -> dict[str, Any]:
        self.ensure_started()
        with self._lock:
            procs = {
                wid: {
                    "alive": (self._procs.get(wid).is_alive() if self._procs.get(wid) else False),
                    "pid": (self._procs.get(wid).pid if self._procs.get(wid) else None),
                }
                for wid in (0, 1)
            }
            return {
                "enabled": self._cfg.enabled,
                "active_worker_id": self._active_worker_id,
                "config": self._cfg.__dict__,
                "workers": {wid: dict(self._worker_state.get(wid) or {}) for wid in (0, 1)},
                "processes": procs,
                "gpu": probe_gpu_status(),
            }

    def wait_for_workers(self, timeout_s: float = 120.0) -> dict[str, Any]:
        self.ensure_started()
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            with self._lock:
                ok = True
                for wid in (0, 1):
                    st = self._worker_state.get(wid) or {}
                    if st.get("last_status_at") is None and st.get("ready_at") is None:
                        ok = False
                        break
                if ok:
                    break
            time.sleep(0.05)
        return self.status()

    def set_portrait(self, filename: str) -> bool:
        self.ensure_started()
        logger.info("mgr_set_portrait filename=%s", filename)
        cmd = WorkerCommand(type=CommandType.SET_FACE, payload=str(filename), ts=time.time())
        return self._broadcast_cmd(cmd)

    def set_scene(self, filename: str) -> bool:
        self.ensure_started()
        logger.info("mgr_set_scene filename=%s", filename)
        cmd = WorkerCommand(type=CommandType.SET_BG, payload=str(filename), ts=time.time())
        return self._broadcast_cmd(cmd)

    def update_config(self, payload: dict[str, Any]) -> bool:
        self.ensure_started()
        cmd = WorkerCommand(type=CommandType.UPDATE_CONFIG, payload=dict(payload), ts=time.time())
        return self._broadcast_cmd(cmd)

    def load_dfm(self, filename: str) -> str:
        self.ensure_started()
        standby = self._standby_id()
        req_id = uuid.uuid4().hex
        logger.info("mgr_load_dfm_request standby_worker_id=%s request_id=%s filename=%s", standby, req_id, filename)
        cmd = WorkerCommand(type=CommandType.LOAD_DFM, payload=str(filename), request_id=req_id, ts=time.time())
        self._send_cmd(standby, cmd)
        with self._lock:
            st = self._worker_state.get(standby) or {}
            st["loading_dfm"] = str(filename)
            st["loading_request_id"] = req_id
            self._worker_state[standby] = st
        return req_id

    def unload_model(self) -> bool:
        self.ensure_started()
        logger.info("mgr_unload_model")
        cmd = WorkerCommand(type=CommandType.UNLOAD_MODEL, payload=None, ts=time.time())
        return self._broadcast_cmd(cmd)

    def activate(self, worker_id: int | None = None) -> int:
        self.ensure_started()
        prev = None
        with self._lock:
            prev = int(self._active_worker_id)
            if worker_id is None:
                worker_id = self._standby_id()
            self._active_worker_id = int(worker_id)
        logger.info("mgr_activate from=%s to=%s", prev, int(worker_id))
        self._send_activate_flags()
        return int(worker_id)

    def submit_frame(self, client_id: bytes, jpeg: bytes, ts: float | None = None) -> bool:
        self.ensure_started()
        wid = 0
        q = None
        seq = 0
        with self._lock:
            if not self._cfg.enabled:
                # Log only once every 60 frames to avoid spam
                s = int(self._seq_by_client.get(client_id, 0))
                if s % 60 == 0:
                    logger.warning("mgr_submit_frame_skipped disabled=True seq=%s", s)
                self._seq_by_client[client_id] = s + 1
                return False
            wid = int(self._active_worker_id)
            q = self._frame_q.get(wid)
            seq = int(self._seq_by_client.get(client_id, 0))
            self._seq_by_client[client_id] = seq + 1
            
            # TRACE LOG
            if seq % 30 == 0:
                logger.info(f"mgr_submit_frame client_id={client_id[:4]}... seq={seq} worker_id={wid} q_size={q.qsize() if hasattr(q, 'qsize') else '?'}")

        if q is None:
            logger.error("mgr_submit_frame_no_queue worker_id=%s", wid)
            return False
        job = FrameJob(client_id=client_id, seq=seq, ts=ts or time.time(), jpeg=jpeg)
        try:
            q.put_nowait(job)
            return True
        except Exception:
            try:
                q.get_nowait()
            except Exception:
                pass
            try:
                q.put_nowait(job)
                return True
            except Exception:
                logger.error("mgr_submit_frame_queue_full worker_id=%s seq=%s", wid, seq)
                return False

    def _broadcast_cmd(self, cmd: WorkerCommand) -> bool:
        logger.info("mgr_broadcast_cmd type=%s request_id=%s", cmd.type.value, cmd.request_id)
        ok = True
        for wid in (0, 1):
            ok = self._send_cmd(wid, cmd) and ok
        return ok

    def _send_cmd(self, worker_id: int, cmd: WorkerCommand) -> bool:
        q = self._cmd_q.get(int(worker_id))
        if q is None:
            return False
        try:
            q.put_nowait(cmd)
            logger.info("mgr_send_cmd worker_id=%s type=%s request_id=%s", int(worker_id), cmd.type.value, cmd.request_id)
            return True
        except Exception:
            return False

    def _standby_id(self) -> int:
        with self._lock:
            return 1 - int(self._active_worker_id)

    def _send_activate_flags(self) -> None:
        active = int(self._active_worker_id)
        for wid in (0, 1):
            self._send_cmd(wid, WorkerCommand(type=CommandType.ACTIVATE, payload={"active": wid == active}, ts=time.time()))
        with self._lock:
            for wid in (0, 1):
                st = self._worker_state.get(wid) or {}
                st["active"] = wid == active
                self._worker_state[wid] = st
        logger.info("mgr_active_worker active_worker_id=%s standby_worker_id=%s", active, 1 - active)

    def _push_config_to_workers(self) -> None:
        with self._lock:
            payload = {
                "similarity_threshold": float(self._cfg.similarity_threshold),
                "depth_enabled": bool(self._cfg.depth_enabled),
                "light_fusion_enabled": bool(self._cfg.light_fusion_enabled),
                "matting_enabled": bool(self._cfg.matting_enabled),
                "jpeg_quality": int(self._cfg.jpeg_quality),
            }
        self.update_config(payload)

    def _out_loop(self) -> None:
        while not self._stop.is_set():
            q = None
            with self._lock:
                q = self._out_q
            if q is None:
                time.sleep(0.05)
                continue
            try:
                out: WorkerOutput = q.get(timeout=0.2)
            except queue.Empty:
                continue
            except Exception:
                time.sleep(0.01)
                continue

            if out.kind == "frame":
                if out.client_id is None or out.jpeg is None:
                    continue
                sender = None
                broadcaster = None
                with self._lock:
                    sender = self._sender
                    broadcaster = self._broadcaster
                if sender is not None:
                    try:
                        sender(out.client_id, out.jpeg)
                    except Exception:
                        pass
                if broadcaster is not None:
                    try:
                        broadcaster(out.jpeg)
                    except Exception:
                        pass
            else:
                with self._lock:
                    st = self._worker_state.get(out.worker_id) or {}
                    diag = out.diag or {}
                    if out.kind == "ready":
                        st["ready_at"] = out.ts
                        if "dfm" in diag:
                            st["loaded_dfm"] = diag.get("dfm")
                            st.pop("loading_dfm", None)
                            st.pop("loading_request_id", None)
                        st["last_ready"] = diag
                        logger.info(
                            "mgr_worker_ready worker_id=%s diag=%s",
                            out.worker_id,
                            diag,
                        )
                    elif out.kind == "error":
                        st["last_error_at"] = out.ts
                        st["last_error"] = diag.get("error") if isinstance(diag, dict) else diag
                        logger.warning(
                            "mgr_worker_error worker_id=%s diag=%s",
                            out.worker_id,
                            diag,
                        )
                    else:
                        st["last_status_at"] = out.ts
                        st.update(diag if isinstance(diag, dict) else {})
                    self._worker_state[out.worker_id] = st
                    if out.kind == "status" and isinstance(diag, dict) and diag.get("providers") is not None:
                        logger.info(
                            "mgr_worker_providers worker_id=%s providers=%s",
                            out.worker_id,
                            diag.get("providers"),
                        )


_mgr: AiActiveStandbyManager | None = None


def get_ai_manager() -> AiActiveStandbyManager:
    global _mgr
    if _mgr is None:
        _mgr = AiActiveStandbyManager()
    return _mgr
