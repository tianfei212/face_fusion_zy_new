from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from ..realtime_pipeline.models import DepthModel, RvmModel, RvmState
from ..realtime_pipeline.onnx_runtime import OrtProvidersConfig, resolve_checkpoint_path
from ..realtime_pipeline.preprocess import chw_from_rgb_u8
from ..video_processing.gpu import probe_gpu_status
from .resources import default_asset_paths, resolve_asset_file

logger = logging.getLogger("blankend.ai_core")


@dataclass
class AiCoreConfig:
    enabled: bool = False
    similarity_threshold: float = 0.75
    portrait_name: str | None = None
    scene_name: str | None = None
    depth_fusion: bool = True
    output_jpeg_quality: int = 85
    max_fps: float = 20.0
    min_fps: float = 15.0


@dataclass
class DeviceHealth:
    device_id: int
    healthy: bool = True
    last_error: str | None = None
    last_error_at: float | None = None
    frames: int = 0
    last_frame_ms: float | None = None
    fps_ema: float = 0.0
    rvm_providers: list[str] | None = None
    depth_providers: list[str] | None = None
    swap_providers: list[str] | None = None


@dataclass
class FrameJob:
    client_id: bytes
    seq: int
    ts: float
    jpeg: bytes


@dataclass
class FrameResult:
    client_id: bytes
    seq: int
    ts: float
    jpeg: bytes
    diag: dict[str, Any]


class _DeviceWorker:
    def __init__(self, device_id: int, output_q: queue.Queue[FrameResult]) -> None:
        self.device_id = int(device_id)
        self.output_q = output_q
        self.input_q: queue.Queue[FrameJob] = queue.Queue(maxsize=256)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, name=f"ai-core-gpu{device_id}", daemon=True)
        self.health = DeviceHealth(device_id=device_id)
        self._lock = threading.RLock()

        self._jpeg = None
        try:
            from turbojpeg import TurboJPEG  # type: ignore

            self._jpeg = TurboJPEG()
        except Exception:
            self._jpeg = None

        self._rvm: RvmModel | None = None
        self._depth: DepthModel | None = None
        self._swapper = None
        self._face_app = None
        self._source_emb: np.ndarray | None = None
        self._scene_rgb: np.ndarray | None = None
        self._rvm_state_by_client: dict[bytes, RvmState] = {}

        self._depth_enabled = True
        self._swap_stride = 1
        self._rvm_downsample_ratio = 0.25
        self._jpeg_quality = 85
        self._similarity_threshold = 0.75
        self._depth_threshold = float(os.getenv("DEPTH_THRESHOLD") or "0.5")

        self._last_fps_ts = time.monotonic()
        self._last_fps_frames = 0

    def start(self) -> None:
        if not self.thread.is_alive():
            self.thread.start()

    def set_runtime(
        self,
        source_emb: np.ndarray | None,
        scene_rgb: np.ndarray | None,
        similarity_threshold: float,
        depth_enabled: bool,
        swap_stride: int,
        rvm_downsample_ratio: float,
        jpeg_quality: int,
    ) -> None:
        with self._lock:
            self._source_emb = source_emb
            self._scene_rgb = scene_rgb
            self._similarity_threshold = float(similarity_threshold)
            self._depth_enabled = bool(depth_enabled)
            self._swap_stride = int(max(1, swap_stride))
            self._rvm_downsample_ratio = float(rvm_downsample_ratio)
            self._jpeg_quality = int(max(30, min(95, jpeg_quality)))

    def submit(self, job: FrameJob) -> bool:
        if not self.health.healthy:
            return False
        try:
            self.input_q.put_nowait(job)
            return True
        except queue.Full:
            return False

    def maybe_restart(self, cooldown_s: float = 5.0) -> None:
        if self.health.healthy:
            return
        t = self.health.last_error_at
        if not t:
            return
        if (time.time() - float(t)) < float(cooldown_s):
            return
        with self._lock:
            self._rvm = None
            self._depth = None
            self._swapper = None
            self._face_app = None
            self._rvm_state_by_client.clear()
        self.health.healthy = True
        self.health.last_error = None
        self.health.last_error_at = None

    def _ensure_models(self) -> None:
        if self._rvm is None or self._depth is None:
            rvm_path = resolve_checkpoint_path(os.getenv("RVM_MODEL") or "rvm_mobilenetv3_fp32.onnx")
            depth_path = resolve_checkpoint_path(os.getenv("DEPTH_MODEL") or "depth_anything_v2_vits_fp32.onnx")
            cfg = OrtProvidersConfig(device_id=self.device_id)
            self._rvm = RvmModel(rvm_path, force_cuda=True, providers_cfg=cfg, use_tensorrt=False)
            self._depth = DepthModel(depth_path, force_cuda=True, providers_cfg=cfg, use_tensorrt=False)
            try:
                self.health.rvm_providers = list(self._rvm.meta.get("providers_active") or [])
            except Exception:
                self.health.rvm_providers = None
            try:
                self.health.depth_providers = list(self._depth.meta.get("providers_active") or [])
            except Exception:
                self.health.depth_providers = None
        if self._face_app is None:
            from insightface.app import FaceAnalysis  # type: ignore

            providers = [("CUDAExecutionProvider", {"device_id": int(self.device_id)}), "CPUExecutionProvider"]
            try:
                self._face_app = FaceAnalysis(name=os.getenv("INSIGHTFACE_APP") or "buffalo_l", providers=providers)
            except Exception:
                self._face_app = FaceAnalysis(name=os.getenv("INSIGHTFACE_APP") or "buffalo_l")
            self._face_app.prepare(ctx_id=0, det_size=(640, 640))
        if self._swapper is None:
            from insightface.model_zoo import get_model  # type: ignore

            swapper_path = resolve_checkpoint_path(os.getenv("SWAP_MODEL") or "inswapper_128_fp16.onnx")
            providers = [("CUDAExecutionProvider", {"device_id": int(self.device_id)}), "CPUExecutionProvider"]
            try:
                self._swapper = get_model(swapper_path, providers=providers)
            except Exception:
                self._swapper = get_model(swapper_path)
            try:
                import onnxruntime as ort  # type: ignore

                if hasattr(self._swapper, "session") and isinstance(getattr(self._swapper, "session"), ort.InferenceSession):
                    self.health.swap_providers = list(self._swapper.session.get_providers())
            except Exception:
                self.health.swap_providers = None

    def _decode(self, jpeg_bytes: bytes) -> np.ndarray | None:
        if self._jpeg is not None:
            try:
                rgb = self._jpeg.decode(jpeg_bytes)
                return rgb[:, :, ::-1].copy()
            except Exception:
                pass
        try:
            import cv2  # type: ignore

            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return bgr
        except Exception:
            return None

    def _encode(self, bgr: np.ndarray) -> bytes:
        if self._jpeg is not None:
            try:
                rgb = bgr[:, :, ::-1].copy()
                return self._jpeg.encode(rgb, quality=self._jpeg_quality)
            except Exception:
                pass
        import cv2  # type: ignore

        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self._jpeg_quality)])
        if not ok:
            return b""
        return bytes(buf)

    def _swap_face(self, bgr: np.ndarray, seq: int) -> tuple[np.ndarray, dict[str, Any]]:
        diag: dict[str, Any] = {"swap_applied": False, "swap_similarity": None}
        if self._face_app is None or self._swapper is None:
            return bgr, diag
        source = self._source_emb
        if source is None:
            return bgr, diag
        if (seq % self._swap_stride) != 0:
            return bgr, diag
        faces = self._face_app.get(bgr)
        if not faces:
            return bgr, diag
        face = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
        try:
            sim = float(np.dot(face.normed_embedding, source))
        except Exception:
            sim = None
        diag["swap_similarity"] = sim
        if sim is None or sim < float(self._similarity_threshold):
            return bgr, diag
        src = SimpleNamespace(normed_embedding=source)
        out = self._swapper.get(bgr, face, src, paste_back=True)
        diag["swap_applied"] = True
        return out, diag

    def _matting_and_compose(self, client_id: bytes, bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        if self._rvm is None or self._depth is None:
            return bgr, {"matting": False}
        rgb0 = bgr[:, :, ::-1].copy()
        h0, w0 = rgb0.shape[:2]
        pad_h = (64 - (h0 % 64)) % 64
        pad_w = (64 - (w0 % 64)) % 64
        if pad_h or pad_w:
            import cv2  # type: ignore

            rgb = cv2.copyMakeBorder(rgb0, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            rgb = rgb0
        h, w = rgb.shape[:2]
        prev = self._rvm_state_by_client.get(client_id)
        st = self._rvm.ensure_state(prev, w, h, downsample_ratio=self._rvm_downsample_ratio)
        rvm_in = chw_from_rgb_u8(rgb, self._rvm._input_dtype())
        alpha, fgr, st, diag_rvm = self._rvm.run(rvm_in, st)
        self._rvm_state_by_client[client_id] = st
        diag: dict[str, Any] = {"matting": True, "rvm": diag_rvm}

        alpha_arr = np.array(alpha)
        if alpha_arr.ndim == 4:
            alpha_arr = alpha_arr.reshape(alpha_arr.shape[-2], alpha_arr.shape[-1])
        elif alpha_arr.ndim == 3:
            alpha_arr = alpha_arr.reshape(alpha_arr.shape[-2], alpha_arr.shape[-1])
        alpha_arr = alpha_arr.astype(np.float32)
        alpha_arr = np.clip(alpha_arr, 0.0, 1.0)

        if fgr is not None:
            f = np.array(fgr)
            if f.ndim == 4:
                c, hh, ww = int(f.shape[1]), int(f.shape[2]), int(f.shape[3])
                if c == 3:
                    f = f.reshape(3, hh, ww).transpose(1, 2, 0)
            fgr_rgb = np.clip(f * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            fgr_rgb = rgb
        if self._depth_enabled and self._depth is not None:
            import cv2  # type: ignore

            size = int(os.getenv("DEPTH_FUSION_SIZE") or "518")
            if size <= 0:
                size = 518
            rgb_small = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
            depth_in = chw_from_rgb_u8(rgb_small, np.float32)
            depth_out, diag_depth = self._depth.run(depth_in)
            diag["depth"] = diag_depth
            m = self._depth_mask(depth_out, self._depth_threshold)
            if m is not None:
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                alpha_arr = alpha_arr * m
        bg = self._scene_rgb
        if bg is None:
            bg = np.zeros_like(rgb)
        else:
            if bg.shape[0] != h or bg.shape[1] != w:
                import cv2  # type: ignore

                bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
        a = alpha_arr.reshape(h, w, 1)
        out = (fgr_rgb.astype(np.float32) * a + bg.astype(np.float32) * (1.0 - a)).astype(np.uint8)
        if pad_h or pad_w:
            out = out[:h0, :w0, :]
        return out[:, :, ::-1].copy(), diag

    def _depth_mask(self, depth: np.ndarray, th: float) -> np.ndarray | None:
        d = np.array(depth)
        if d.ndim == 4:
            d = d.reshape(d.shape[-2], d.shape[-1])
        elif d.ndim == 3:
            d = d.reshape(d.shape[-2], d.shape[-1])
        if d.ndim != 2:
            return None
        dmin = float(np.min(d))
        dmax = float(np.max(d))
        rng = dmax - dmin
        if rng < 1e-6:
            return None
        norm = (d - dmin) / rng
        mask = (norm > float(th)).astype(np.float32)
        ratio = float(np.mean(mask))
        if ratio < 0.02 or ratio > 0.98:
            return None
        return mask

    def _tick_fps(self) -> None:
        now = time.monotonic()
        self._last_fps_frames += 1
        if now - self._last_fps_ts < 1.0:
            return
        fps = float(self._last_fps_frames) / max(1e-6, (now - self._last_fps_ts))
        self._last_fps_frames = 0
        self._last_fps_ts = now
        self.health.fps_ema = (self.health.fps_ema * 0.8) + (fps * 0.2)

    def _loop(self) -> None:
        while not self.stop.is_set():
            try:
                job = self.input_q.get(timeout=0.2)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            diag: dict[str, Any] = {"device_id": self.device_id}
            try:
                self._ensure_models()
                bgr = self._decode(job.jpeg)
                if bgr is None:
                    raise RuntimeError("decode_failed")
                bgr2, swap_diag = self._swap_face(bgr, job.seq)
                diag.update(swap_diag)
                out_bgr, mat_diag = self._matting_and_compose(job.client_id, bgr2)
                diag.update(mat_diag)
                out_jpeg = self._encode(out_bgr)
                if not out_jpeg:
                    raise RuntimeError("encode_failed")
                self.output_q.put(FrameResult(client_id=job.client_id, seq=job.seq, ts=job.ts, jpeg=out_jpeg, diag=diag))
                self.health.frames += 1
                self._tick_fps()
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                self.health.healthy = False
                self.health.last_error = msg
                self.health.last_error_at = time.time()
                logger.exception("device worker failed device=%s err=%s", self.device_id, msg)
            finally:
                self.health.last_frame_ms = (time.perf_counter() - t0) * 1000.0


class AiCoreService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cfg = AiCoreConfig()
        self._assets = default_asset_paths()

        self._sender: Callable[[bytes, bytes], None] | None = None
        self._broadcaster: Callable[[bytes], Any] | None = None

        self._output_q: queue.Queue[FrameResult] = queue.Queue(maxsize=2048)
        self._workers = [_DeviceWorker(0, self._output_q), _DeviceWorker(1, self._output_q)]
        self._dispatch_thread = threading.Thread(target=self._dispatch_loop, name="ai-core-dispatch", daemon=True)
        self._send_thread = threading.Thread(target=self._send_loop, name="ai-core-send", daemon=True)
        self._stop = threading.Event()

        self._seq_by_client: dict[bytes, int] = {}
        self._next_send_seq: dict[bytes, int] = {}
        self._pending: dict[tuple[bytes, int], FrameResult] = {}

        self._in_q: queue.Queue[FrameJob] = queue.Queue(maxsize=2048)

        self._portrait_cache: dict[str, np.ndarray] = {}
        self._scene_cache: dict[str, np.ndarray] = {}
        self._active_source_emb: np.ndarray | None = None
        self._active_scene_rgb: np.ndarray | None = None

        self._degrade_level = 0
        self._last_degrade_check = time.monotonic()
        self._out_frames = 0
        self._out_fps_ema = 0.0

    def attach(self, sender: Callable[[bytes, bytes], None], broadcaster: Callable[[bytes], Any] | None) -> None:
        with self._lock:
            self._sender = sender
            self._broadcaster = broadcaster

    def start(self) -> None:
        with self._lock:
            if self._dispatch_thread.is_alive():
                return
            self._stop.clear()
            for w in self._workers:
                w.start()
            self._dispatch_thread.start()
            self._send_thread.start()

    def configure(
        self,
        enabled: bool | None = None,
        similarity_threshold: float | None = None,
        portrait_name: str | None = None,
        scene_name: str | None = None,
        depth_fusion: bool | None = None,
        output_jpeg_quality: int | None = None,
        max_fps: float | None = None,
        min_fps: float | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if enabled is not None:
                self._cfg.enabled = bool(enabled)
            if similarity_threshold is not None:
                self._cfg.similarity_threshold = float(max(0.0, min(1.0, similarity_threshold)))
            if depth_fusion is not None:
                self._cfg.depth_fusion = bool(depth_fusion)
            if output_jpeg_quality is not None:
                self._cfg.output_jpeg_quality = int(max(30, min(95, output_jpeg_quality)))
            if max_fps is not None:
                self._cfg.max_fps = float(max(1.0, max_fps))
            if min_fps is not None:
                self._cfg.min_fps = float(max(1.0, min_fps))
        if portrait_name is not None:
            self.set_portrait(portrait_name)
        if scene_name is not None:
            self.set_scene(scene_name)
        return self.status()

    def enabled(self) -> bool:
        with self._lock:
            return bool(self._cfg.enabled)

    def set_portrait(self, portrait_name: str) -> bool:
        name = str(portrait_name)
        p = resolve_asset_file(self._assets.portraits_dir, name)
        if p is None:
            return False
        try:
            import cv2  # type: ignore

            img = cv2.imdecode(np.frombuffer(p.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return False
        except Exception:
            return False
        with self._lock:
            self._portrait_cache[name] = img
            self._cfg.portrait_name = name
        self._refresh_active_assets()
        return True

    def set_scene(self, scene_name: str) -> bool:
        name = str(scene_name)
        p = resolve_asset_file(self._assets.scenes_dir, name)
        if p is None:
            return False
        try:
            import cv2  # type: ignore

            bgr = cv2.imdecode(np.frombuffer(p.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                return False
            rgb = bgr[:, :, ::-1].copy()
        except Exception:
            return False
        with self._lock:
            self._scene_cache[name] = rgb
            self._cfg.scene_name = name
        self._refresh_active_assets()
        return True

    def _refresh_active_assets(self) -> None:
        with self._lock:
            portrait_name = self._cfg.portrait_name
            scene_name = self._cfg.scene_name
            portrait_bgr = self._portrait_cache.get(portrait_name or "") if portrait_name else None
            scene_rgb = self._scene_cache.get(scene_name or "") if scene_name else None

        emb = None
        if portrait_bgr is not None:
            try:
                from insightface.app import FaceAnalysis  # type: ignore

                providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
                app = FaceAnalysis(name=os.getenv("INSIGHTFACE_APP") or "buffalo_l", providers=providers)
                app.prepare(ctx_id=0, det_size=(640, 640))
                faces = app.get(portrait_bgr)
                if faces:
                    face = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
                    emb = np.array(face.normed_embedding, dtype=np.float32).reshape((512,))
            except Exception:
                emb = None

        with self._lock:
            self._active_source_emb = emb
            self._active_scene_rgb = scene_rgb

    def submit(self, client_id: bytes, jpeg: bytes, ts: float | None = None) -> bool:
        with self._lock:
            if not self._cfg.enabled:
                return False
            seq = self._seq_by_client.get(client_id, 0)
            self._seq_by_client[client_id] = seq + 1
        job = FrameJob(client_id=client_id, seq=seq, ts=ts or time.time(), jpeg=jpeg)
        try:
            self._in_q.put_nowait(job)
            return True
        except queue.Full:
            return False

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self._cfg.enabled,
                "similarity_threshold": self._cfg.similarity_threshold,
                "portrait_name": self._cfg.portrait_name,
                "scene_name": self._cfg.scene_name,
                "depth_fusion": self._cfg.depth_fusion,
                "output_jpeg_quality": self._cfg.output_jpeg_quality,
                "max_fps": self._cfg.max_fps,
                "min_fps": self._cfg.min_fps,
                "degrade_level": self._degrade_level,
                "out_fps_ema": self._out_fps_ema,
                "devices": [w.health.__dict__ for w in self._workers],
                "gpu": probe_gpu_status(),
            }

    def _pick_worker(self, seq: int) -> _DeviceWorker | None:
        for w in self._workers:
            w.maybe_restart()
        healthy = [w for w in self._workers if w.health.healthy]
        if not healthy:
            return None
        return healthy[seq % len(healthy)]

    def _dispatch_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            with self._lock:
                emb = self._active_source_emb
                scene = self._active_scene_rgb
                depth_enabled = bool(self._cfg.depth_fusion)
                sim = float(self._cfg.similarity_threshold)
                jpeg_q = int(self._cfg.output_jpeg_quality)
                max_fps = float(self._cfg.max_fps)
            w = self._pick_worker(job.seq)
            if w is None:
                continue
            depth_enabled, swap_stride, rvm_ratio, jpeg_q = self._apply_degrade(depth_enabled, jpeg_q)
            w.set_runtime(
                source_emb=emb,
                scene_rgb=scene,
                similarity_threshold=sim,
                depth_enabled=depth_enabled,
                swap_stride=swap_stride,
                rvm_downsample_ratio=rvm_ratio,
                jpeg_quality=jpeg_q,
            )
            ok = w.submit(job)
            if not ok:
                time.sleep(0.001)
                try:
                    self._in_q.put_nowait(job)
                except queue.Full:
                    pass
            if max_fps > 0:
                time.sleep(max(0.0, (1.0 / max_fps) * 0.15))

    def _apply_degrade(self, depth_enabled: bool, jpeg_q: int) -> tuple[bool, int, float, int]:
        with self._lock:
            level = int(self._degrade_level)
        swap_stride = 1
        rvm_ratio = 0.25
        if level >= 1:
            depth_enabled = False
        if level >= 2:
            swap_stride = 2
        if level >= 3:
            rvm_ratio = 0.5
            jpeg_q = min(jpeg_q, 75)
        return depth_enabled, swap_stride, rvm_ratio, jpeg_q

    def _send_loop(self) -> None:
        last_tick = time.monotonic()
        while not self._stop.is_set():
            try:
                res = self._output_q.get(timeout=0.2)
            except queue.Empty:
                self._maybe_degrade()
                continue
            key = (res.client_id, res.seq)
            with self._lock:
                self._pending[key] = res
            self._flush_client(res.client_id)
            self._out_frames += 1
            now = time.monotonic()
            if now - last_tick >= 1.0:
                fps = float(self._out_frames) / max(1e-6, (now - last_tick))
                self._out_fps_ema = (self._out_fps_ema * 0.8) + (fps * 0.2)
                self._out_frames = 0
                last_tick = now
            self._maybe_degrade()

    def _flush_client(self, client_id: bytes) -> None:
        sender = None
        broadcaster = None
        with self._lock:
            sender = self._sender
            broadcaster = self._broadcaster
            want = self._next_send_seq.get(client_id, 0)
        while True:
            k = (client_id, want)
            with self._lock:
                res = self._pending.pop(k, None)
                if res is None:
                    return
                self._next_send_seq[client_id] = want + 1
                want = want + 1
            if sender is not None:
                try:
                    sender(client_id, res.jpeg)
                except Exception:
                    pass
            if broadcaster is not None:
                try:
                    broadcaster(res.jpeg)
                except Exception:
                    pass

    def _maybe_degrade(self) -> None:
        now = time.monotonic()
        with self._lock:
            if now - self._last_degrade_check < 2.0:
                return
            self._last_degrade_check = now
            fps = float(sum(w.health.fps_ema for w in self._workers if w.health.healthy))
            min_fps = float(self._cfg.min_fps)
            if fps > 0.1 and fps < min_fps:
                self._degrade_level = min(3, int(self._degrade_level) + 1)
                logger.warning("auto degrade level=%s fps=%.2f min_fps=%.2f", self._degrade_level, fps, min_fps)
            elif fps > (min_fps + 3.0) and self._degrade_level > 0:
                self._degrade_level = max(0, int(self._degrade_level) - 1)


_svc: AiCoreService | None = None


def get_ai_core_service() -> AiCoreService:
    global _svc
    if _svc is None:
        _svc = AiCoreService()
        _svc.start()
    return _svc
