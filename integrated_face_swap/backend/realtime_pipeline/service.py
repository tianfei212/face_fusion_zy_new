from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

import logging

from ..video_processing.gpu import probe_gpu_status
from .models import DepthModel, RvmModel, load_default_models, RvmState
from .preprocess import chw_from_rgb_u8
from .recording import FfmpegRecorder, RecordingSpec

logger = logging.getLogger("blankend.realtime_pipeline")


try:
    from turbojpeg import TurboJPEG, TJPF_RGB  # type: ignore
except Exception:
    TurboJPEG = None  # type: ignore
    TJPF_RGB = None  # type: ignore


@dataclass
class PipelineConfig:
    enabled: bool = False
    force_cuda: bool = True
    target_max_latency_ms: float = 300.0
    latest_frame_only: bool = True
    output_jpeg_quality: int = 85
    depth_fusion: bool = True
    depth_threshold: float = 0.45
    depth_alpha_boost: float = 0.5
    background_mode: str = "original"
    background_jpeg_path: str | None = None
    max_fps: float = 60.0
    auto_degrade: bool = True
    degrade_depth_if_slow_ms: float = 40.0


@dataclass
class ClientSlot:
    client_id: bytes
    frame: bytes | None = None
    ts: float = 0.0
    has_new: bool = False
    busy: bool = False
    rvm_state: RvmState | None = None
    last_diag: dict[str, Any] | None = None
    last_out_ts: float = 0.0


class RealtimePipelineService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._work_q: queue.Queue[bytes] = queue.Queue(maxsize=2048)

        self._cfg = PipelineConfig()
        enabled_env = os.getenv("BLANKEND_REALTIME_PIPELINE_ENABLED")
        if enabled_env is not None:
            self._cfg.enabled = enabled_env.strip() not in {"0", "false", "False", "no", "NO", ""}

        self._jpeg = TurboJPEG() if TurboJPEG else None
        self._bg_jpeg_cache: tuple[str, bytes] | None = None
        self._bg_rgb_cache: tuple[str, np.ndarray] | None = None

        self._rvm: RvmModel | None = None
        self._depth: DepthModel | None = None

        self._clients: dict[bytes, ClientSlot] = {}
        self._recorders: dict[bytes, FfmpegRecorder] = {}
        self._sent = 0
        self._dropped = 0
        self._last_log = time.monotonic()

        self._sender: Callable[[bytes, bytes], None] | None = None
        self._broadcaster: Callable[[bytes], Any] | None = None

    def attach(self, sender: Callable[[bytes, bytes], None], broadcaster: Callable[[bytes], Any] | None = None) -> None:
        with self._lock:
            self._sender = sender
            self._broadcaster = broadcaster

    def start(self) -> None:
        with self._lock:
            self._threads = [t for t in self._threads if t.is_alive()]
            if self._threads:
                return
            self._stop.clear()
            workers = int(os.getenv("BLANKEND_REALTIME_PIPELINE_WORKERS") or "1")
            if workers < 1:
                workers = 1
            for i in range(workers):
                t = threading.Thread(target=self._worker_loop, name=f"realtime-pipeline-{i}", daemon=True)
                t.start()
                self._threads.append(t)
            logger.info(
                "realtime pipeline workers started workers=%s enabled=%s force_cuda=%s",
                workers,
                self._cfg.enabled,
                self._cfg.force_cuda,
            )

    def stop(self) -> None:
        self._stop.set()

    def configure(self, **kwargs: Any) -> dict[str, Any]:
        with self._lock:
            for k, v in kwargs.items():
                if not hasattr(self._cfg, k):
                    continue
                setattr(self._cfg, k, v)
            if self._cfg.output_jpeg_quality < 1 or self._cfg.output_jpeg_quality > 100:
                self._cfg.output_jpeg_quality = 85
            if self._cfg.max_fps < 1:
                self._cfg.max_fps = 1.0
            if self._cfg.background_jpeg_path:
                self._bg_jpeg_cache = None
                self._bg_rgb_cache = None
            logger.info("realtime pipeline configured %s", json.dumps(self.status(), ensure_ascii=False))
            return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            model_info = None
            if self._rvm is not None and self._depth is not None:
                model_info = {
                    "rvm": {"path": self._rvm.model_path, "providers_active": self._rvm.meta.get("providers_active"), "cuda_failed": self._rvm.meta.get("cuda_failed")},
                    "depth": {"path": self._depth.model_path, "providers_active": self._depth.meta.get("providers_active"), "cuda_failed": self._depth.meta.get("cuda_failed")},
                }
            return {
                "enabled": self._cfg.enabled,
                "force_cuda": self._cfg.force_cuda,
                "latest_frame_only": self._cfg.latest_frame_only,
                "depth_fusion": self._cfg.depth_fusion,
                "background_mode": self._cfg.background_mode,
                "background_jpeg_path": self._cfg.background_jpeg_path,
                "output_jpeg_quality": self._cfg.output_jpeg_quality,
                "clients": len(self._clients),
                "recording_clients": [k.decode("utf-8", errors="replace") for k in self._recorders.keys()],
                "sent": self._sent,
                "dropped": self._dropped,
                "turbojpeg_available": self._jpeg is not None,
                "gpu": probe_gpu_status(),
                "models": model_info,
            }

    def start_recording(self, client_id: bytes, fmt: str = "mp4", fps: int = 30, out_path: str | None = None) -> str:
        with self._lock:
            rec = self._recorders.get(client_id)
            if rec is None:
                rec = FfmpegRecorder(RecordingSpec(format=fmt, fps=int(fps), out_path=out_path))
                self._recorders[client_id] = rec
        return rec.start()

    def stop_recording(self, client_id: bytes) -> str | None:
        with self._lock:
            rec = self._recorders.pop(client_id, None)
        if rec is None:
            return None
        path = rec.output_path()
        rec.stop()
        return path

    def submit(self, client_id: bytes, frame: bytes, ts: float | None = None) -> bool:
        with self._lock:
            if not self._cfg.enabled:
                return False
            slot = self._clients.get(client_id)
            if slot is None:
                slot = ClientSlot(client_id=client_id)
                self._clients[client_id] = slot
            if self._cfg.latest_frame_only:
                slot.frame = frame
                slot.ts = ts or time.time()
                slot.has_new = True
                try:
                    self._work_q.put_nowait(client_id)
                except queue.Full:
                    pass
                return True
            if slot.has_new:
                self._dropped += 1
                slot.frame = frame
                slot.ts = ts or time.time()
                slot.has_new = True
                try:
                    self._work_q.put_nowait(client_id)
                except queue.Full:
                    pass
                return True
            slot.frame = frame
            slot.ts = ts or time.time()
            slot.has_new = True
            try:
                self._work_q.put_nowait(client_id)
            except queue.Full:
                pass
            return True

    def _ensure_models(self) -> None:
        if self._rvm is not None and self._depth is not None:
            return
        self._rvm, self._depth = load_default_models(force_cuda=self._cfg.force_cuda)
        logger.info("models loaded rvm=%s depth=%s providers=%s", self._rvm.model_path, self._depth.model_path, self._rvm.meta.get("providers_active"))

    def _decode_jpeg_rgb(self, data: bytes) -> np.ndarray:
        if self._jpeg is None:
            raise RuntimeError("TurboJPEG not available")
        if TJPF_RGB is not None:
            try:
                return self._jpeg.decode(data, pixel_format=TJPF_RGB)
            except Exception:
                pass
        img = self._jpeg.decode(data)
        if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3:
            return img[:, :, ::-1].copy()
        return img

    def _encode_jpeg(self, rgb: np.ndarray, quality: int) -> bytes:
        if self._jpeg is None:
            raise RuntimeError("TurboJPEG not available")
        return self._jpeg.encode(rgb, quality=int(quality))

    def _load_background_rgb(self) -> np.ndarray | None:
        path = self._cfg.background_jpeg_path
        if not path:
            return None
        if self._bg_rgb_cache is not None and self._bg_rgb_cache[0] == path:
            return self._bg_rgb_cache[1]
        data = open(path, "rb").read()
        rgb = self._decode_jpeg_rgb(data)
        self._bg_rgb_cache = (path, rgb)
        return rgb

    def _depth_mask(self, depth: np.ndarray, th: float) -> np.ndarray | None:
        d = depth
        if d is None:
            return None
        d = np.array(d)
        if d.ndim == 4:
            d = d.reshape(d.shape[-2], d.shape[-1])
        elif d.ndim == 3:
            d = d.reshape(d.shape[-2], d.shape[-1])
        dmin = float(np.min(d))
        dmax = float(np.max(d))
        rng = dmax - dmin
        if rng < 1e-6:
            return None
        norm = (d - dmin) / rng
        mask = (norm > float(th)).astype(np.float32)
        ratio = float(np.mean(mask))
        if ratio < 0.05 or ratio > 0.95:
            return None
        return mask

    def _compose(self, fgr: np.ndarray, alpha: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
        h, w = bg_rgb.shape[:2]
        if fgr.shape[0] != h or fgr.shape[1] != w:
            fgr = fgr[:h, :w]
        a = alpha.reshape(h, w, 1)
        a = np.clip(a, 0.0, 1.0).astype(np.float32)
        out = (a * fgr.astype(np.float32) + (1.0 - a) * bg_rgb.astype(np.float32)).astype(np.uint8)
        return out

    def _process_one(self, slot: ClientSlot) -> bytes | None:
        self._ensure_models()
        if self._rvm is None or self._depth is None:
            return None
        if slot.frame is None:
            return None
        rgb = self._decode_jpeg_rgb(slot.frame)
        h, w = rgb.shape[:2]
        rvm_dtype = self._rvm._input_dtype()
        rvm_in = chw_from_rgb_u8(rgb, rvm_dtype)
        slot.rvm_state = self._rvm.ensure_state(slot.rvm_state, w, h, downsample_ratio=0.25)
        alpha, fgr, slot.rvm_state, diag_rvm = self._rvm.run(rvm_in, slot.rvm_state)
        alpha_arr = np.array(alpha)
        if alpha_arr.ndim == 4:
            alpha_arr = alpha_arr.reshape(alpha_arr.shape[-2], alpha_arr.shape[-1])
        alpha_max = float(np.max(alpha_arr))
        if alpha_max < 0.01:
            return slot.frame
        if fgr is not None:
            f = np.array(fgr)
            if f.ndim == 4:
                c, hh, ww = int(f.shape[1]), int(f.shape[2]), int(f.shape[3])
                if c == 3:
                    f = f.reshape(3, hh, ww).transpose(1, 2, 0)
            fgr_rgb = np.clip(f * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            fgr_rgb = rgb

        depth_mask = None
        diag_depth: dict[str, Any] = {}
        if self._cfg.depth_fusion:
            depth_dtype = self._depth._input_dtype()
            depth_in = chw_from_rgb_u8(rgb, depth_dtype)
            depth_out, diag_depth = self._depth.run(depth_in)
            depth_mask = self._depth_mask(depth_out, self._cfg.depth_threshold)
            if depth_mask is not None:
                if depth_mask.shape[0] != h or depth_mask.shape[1] != w:
                    depth_mask = depth_mask[:h, :w]
                alpha_arr = alpha_arr * depth_mask
                alpha_arr = np.clip(alpha_arr * (1.0 - self._cfg.depth_alpha_boost) + alpha_arr * self._cfg.depth_alpha_boost, 0.0, 1.0)

        bg = rgb
        if self._cfg.background_mode == "image":
            bg2 = self._load_background_rgb()
            if bg2 is not None:
                bg = bg2
        out_rgb = self._compose(fgr_rgb, alpha_arr, bg)
        out_jpg = self._encode_jpeg(out_rgb, self._cfg.output_jpeg_quality)
        diag = {"w": w, "h": h, **diag_rvm, **diag_depth}
        slot.last_diag = diag
        return out_jpg

    def _worker_loop(self) -> None:
        last_tick = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                enabled = self._cfg.enabled
                sender = self._sender
                broadcaster = self._broadcaster
                max_fps = float(self._cfg.max_fps)
                auto_degrade = bool(self._cfg.auto_degrade)
                degrade_depth_if_slow_ms = float(self._cfg.degrade_depth_if_slow_ms)
            if not enabled or sender is None:
                time.sleep(0.01)
                continue

            now = time.monotonic()
            min_interval = 1.0 / max(1.0, max_fps)
            if now - last_tick < min_interval:
                time.sleep(0.001)
                continue
            last_tick = now

            try:
                client_id = self._work_q.get(timeout=0.2)
            except queue.Empty:
                continue
            with self._lock:
                slot = self._clients.get(client_id)
                if slot is None or slot.busy or not slot.has_new:
                    continue
                slot.busy = True
                frame = slot.frame
                ts = slot.ts
                slot.has_new = False
            if frame is None:
                with self._lock:
                    slot.busy = False
                continue
            started = time.perf_counter()
            try:
                out = self._process_one(slot)
            except Exception as e:
                out = frame
                slot.last_diag = {"error": f"{type(e).__name__}: {e}"}
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latency_ms = max(0.0, (time.time() - ts) * 1000.0)
            if auto_degrade and self._cfg.depth_fusion and elapsed_ms > degrade_depth_if_slow_ms:
                with self._lock:
                    self._cfg.depth_fusion = False
                    logger.warning("auto degrade: disable depth_fusion due to slow frame ms=%.2f", elapsed_ms)

            if out is not None:
                sender(slot.client_id, out)
                if broadcaster is not None:
                    try:
                        broadcaster(out)
                    except Exception:
                        pass
                with self._lock:
                    rec = self._recorders.get(slot.client_id)
                if rec is not None:
                    try:
                        rec.write_jpeg(out)
                    except Exception:
                        pass
                with self._lock:
                    self._sent += 1
                    slot.last_out_ts = time.time()

            with self._lock:
                slot.busy = False

            now2 = time.monotonic()
            if now2 - self._last_log >= 1.0:
                with self._lock:
                    diag = slot.last_diag or {}
                    logger.info(
                        "realtime pipeline frames=%s clients=%s last_frame_ms=%.2f latency_ms=%.2f diag=%s",
                        self._sent,
                        len(self._clients),
                        elapsed_ms,
                        latency_ms,
                        json.dumps(diag, ensure_ascii=False)[:512],
                    )
                    self._last_log = now2


_svc: RealtimePipelineService | None = None


def get_realtime_pipeline_service() -> RealtimePipelineService:
    global _svc
    if _svc is None:
        _svc = RealtimePipelineService()
        _svc.start()
    return _svc
