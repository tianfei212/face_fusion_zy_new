from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
import logging
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable


def _load_config() -> dict:
    path = Path(__file__).resolve().parent.parent / "config.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

logger = logging.getLogger("blankend.video_processing")


@dataclass
class PipelineConfig:
    enabled: bool = False
    queue_size: int = 8
    decode_workers: int = 1
    encode_workers: int = 1
    reencode_enabled: bool = False
    jpeg_quality: int = 85
    heartbeat_timeout_s: float = 3.0


@dataclass
class ThreadHealth:
    name: str
    last_heartbeat: float = 0.0
    last_error: str | None = None
    frames: int = 0


class VideoProcessingService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._broadcast_bytes: Callable[[bytes], Any] | None = None
        self._running = False
        self._stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None

        self._cfg = PipelineConfig()
        self._in_q: queue.Queue[tuple[float, bytes]] = queue.Queue(maxsize=self._cfg.queue_size)
        self._mid_q: queue.Queue[tuple[float, bytes]] = queue.Queue(maxsize=self._cfg.queue_size)

        self._decode_threads: list[threading.Thread] = []
        self._encode_threads: list[threading.Thread] = []
        self._thread_stop: dict[str, threading.Event] = {}
        self._health: dict[str, ThreadHealth] = {}
        self._dropped = 0

        self._jpeg = None
        try:
            from turbojpeg import TurboJPEG  # type: ignore

            self._jpeg = TurboJPEG()
        except Exception:
            self._jpeg = None

        self.reload_from_config_file()

    def reload_from_config_file(self) -> None:
        cfg = _load_config()
        stream_cfg = cfg.get("stream", {}) if isinstance(cfg, dict) else {}
        proc_cfg = stream_cfg.get("video_processing", {}) if isinstance(stream_cfg, dict) else {}

        enabled_env = os.getenv("BLANKEND_VIDEO_PROCESSING_ENABLED")
        if enabled_env is not None:
            enabled = enabled_env.strip() not in {"0", "false", "False", "no", "NO", ""}
        else:
            enabled = bool(proc_cfg.get("enabled", False))

        with self._lock:
            self._cfg.enabled = enabled
            self._cfg.queue_size = int(proc_cfg.get("queue_size", self._cfg.queue_size) or self._cfg.queue_size)
            self._cfg.decode_workers = int(proc_cfg.get("decode_workers", self._cfg.decode_workers) or self._cfg.decode_workers)
            self._cfg.encode_workers = int(proc_cfg.get("encode_workers", self._cfg.encode_workers) or self._cfg.encode_workers)
            self._cfg.reencode_enabled = bool(proc_cfg.get("reencode_enabled", self._cfg.reencode_enabled))
            self._cfg.jpeg_quality = int(proc_cfg.get("jpeg_quality", self._cfg.jpeg_quality) or self._cfg.jpeg_quality)
            self._cfg.heartbeat_timeout_s = float(proc_cfg.get("heartbeat_timeout_s", self._cfg.heartbeat_timeout_s) or self._cfg.heartbeat_timeout_s)

            if self._cfg.queue_size < 1:
                self._cfg.queue_size = 1
            if self._cfg.decode_workers < 1:
                self._cfg.decode_workers = 1
            if self._cfg.encode_workers < 1:
                self._cfg.encode_workers = 1
            if self._cfg.jpeg_quality < 1 or self._cfg.jpeg_quality > 100:
                self._cfg.jpeg_quality = 85

            self._resize_queues_locked()

    def _resize_queues_locked(self) -> None:
        self._in_q = queue.Queue(maxsize=self._cfg.queue_size)
        self._mid_q = queue.Queue(maxsize=self._cfg.queue_size)

    def attach(self, loop: asyncio.AbstractEventLoop, broadcast_bytes: Callable[[bytes], Any]) -> None:
        with self._lock:
            self._loop = loop
            self._broadcast_bytes = broadcast_bytes

    def configure(self, **kwargs: Any) -> PipelineConfig:
        with self._lock:
            for k, v in kwargs.items():
                if not hasattr(self._cfg, k):
                    continue
                setattr(self._cfg, k, v)
            if self._cfg.queue_size < 1:
                self._cfg.queue_size = 1
            if self._cfg.decode_workers < 1:
                self._cfg.decode_workers = 1
            if self._cfg.encode_workers < 1:
                self._cfg.encode_workers = 1
            if self._cfg.jpeg_quality < 1 or self._cfg.jpeg_quality > 100:
                self._cfg.jpeg_quality = 85
            self._resize_queues_locked()
            if self._running:
                self._ensure_workers_locked()
            logger.info(
                "video_processing reconfigured enabled=%s queue_size=%s decode_workers=%s encode_workers=%s reencode=%s jpeg_quality=%s",
                self._cfg.enabled,
                self._cfg.queue_size,
                self._cfg.decode_workers,
                self._cfg.encode_workers,
                self._cfg.reencode_enabled,
                self._cfg.jpeg_quality,
            )
            return PipelineConfig(**self._cfg.__dict__)

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._stop.clear()
            self._ensure_workers_locked()
            self._monitor_thread = threading.Thread(target=self._monitor_loop, name="vp-monitor", daemon=True)
            self._monitor_thread.start()
            logger.info(
                "video_processing started enabled=%s queue_size=%s decode_workers=%s encode_workers=%s reencode=%s jpeg_quality=%s",
                self._cfg.enabled,
                self._cfg.queue_size,
                self._cfg.decode_workers,
                self._cfg.encode_workers,
                self._cfg.reencode_enabled,
                self._cfg.jpeg_quality,
            )

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._stop.set()

    def enabled(self) -> bool:
        with self._lock:
            return bool(self._cfg.enabled)

    def submit(self, ts: float, data: bytes) -> bool:
        with self._lock:
            if not self._cfg.enabled:
                return False
            try:
                self._in_q.put_nowait((ts, data))
                return True
            except queue.Full:
                self._dropped += 1
                try:
                    _ = self._in_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._in_q.put_nowait((ts, data))
                    return True
                except queue.Full:
                    self._dropped += 1
                    return False

    def status(self) -> dict[str, Any]:
        with self._lock:
            health = {k: dict(v.__dict__) for k, v in self._health.items()}
            return {
                "enabled": self._cfg.enabled,
                "queue_size": self._cfg.queue_size,
                "decode_workers": self._cfg.decode_workers,
                "encode_workers": self._cfg.encode_workers,
                "reencode_enabled": self._cfg.reencode_enabled,
                "jpeg_quality": self._cfg.jpeg_quality,
                "heartbeat_timeout_s": self._cfg.heartbeat_timeout_s,
                "in_queue_len": self._in_q.qsize(),
                "mid_queue_len": self._mid_q.qsize(),
                "dropped": self._dropped,
                "threads": health,
                "turbojpeg_available": self._jpeg is not None,
            }

    def _ensure_workers_locked(self) -> None:
        self._decode_threads = [t for t in self._decode_threads if t.is_alive()]
        self._encode_threads = [t for t in self._encode_threads if t.is_alive()]

        while len(self._decode_threads) > self._cfg.decode_workers:
            t = self._decode_threads.pop()
            stop_evt = self._thread_stop.get(t.name)
            if stop_evt is not None:
                stop_evt.set()

        while len(self._encode_threads) > self._cfg.encode_workers:
            t = self._encode_threads.pop()
            stop_evt = self._thread_stop.get(t.name)
            if stop_evt is not None:
                stop_evt.set()

        while len(self._decode_threads) < self._cfg.decode_workers:
            idx = len(self._decode_threads)
            name = f"vp-decode-{idx}"
            stop_evt = threading.Event()
            self._thread_stop[name] = stop_evt
            th = threading.Thread(target=self._decode_loop, name=name, daemon=True, args=(name, stop_evt))
            self._decode_threads.append(th)
            self._health.setdefault(name, ThreadHealth(name=name, last_heartbeat=time.monotonic()))
            th.start()

        while len(self._encode_threads) < self._cfg.encode_workers:
            idx = len(self._encode_threads)
            name = f"vp-encode-{idx}"
            stop_evt = threading.Event()
            self._thread_stop[name] = stop_evt
            th = threading.Thread(target=self._encode_loop, name=name, daemon=True, args=(name, stop_evt))
            self._encode_threads.append(th)
            self._health.setdefault(name, ThreadHealth(name=name, last_heartbeat=time.monotonic()))
            th.start()

    def _heartbeat(self, name: str) -> None:
        with self._lock:
            h = self._health.get(name)
            if h is None:
                h = ThreadHealth(name=name)
                self._health[name] = h
            h.last_heartbeat = time.monotonic()

    def _set_error(self, name: str, err: Exception) -> None:
        with self._lock:
            h = self._health.get(name)
            if h is None:
                h = ThreadHealth(name=name)
                self._health[name] = h
            h.last_error = f"{type(err).__name__}: {err}"

    def _inc_frames(self, name: str) -> None:
        with self._lock:
            h = self._health.get(name)
            if h is None:
                h = ThreadHealth(name=name)
                self._health[name] = h
            h.frames += 1

    def _decode_loop(self, name: str, stop_evt: threading.Event) -> None:
        while not self._stop.is_set() and not stop_evt.is_set():
            self._heartbeat(name)
            try:
                ts, data = self._in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if self._cfg.reencode_enabled and self._jpeg is not None:
                    decoded = self._jpeg.decode(data)
                    data = self._jpeg.encode(decoded, quality=self._cfg.jpeg_quality)
                self._mid_q.put((ts, data), timeout=0.5)
                self._inc_frames(name)
            except Exception as e:
                self._set_error(name, e)
                time.sleep(0.05)

    def _encode_loop(self, name: str, stop_evt: threading.Event) -> None:
        while not self._stop.is_set() and not stop_evt.is_set():
            self._heartbeat(name)
            try:
                ts, data = self._mid_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                loop = self._loop
                cb = self._broadcast_bytes
                if loop is not None and cb is not None:
                    asyncio.run_coroutine_threadsafe(cb(data), loop)
                self._inc_frames(name)
            except Exception as e:
                self._set_error(name, e)
                time.sleep(0.05)

    def _monitor_loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(0.5)
            with self._lock:
                if not self._running:
                    continue
                now = time.monotonic()
                timeout = float(self._cfg.heartbeat_timeout_s)
                for name, h in list(self._health.items()):
                    if now - float(h.last_heartbeat) > timeout:
                        h.last_error = h.last_error or "heartbeat timeout"
                self._ensure_workers_locked()


_service: VideoProcessingService | None = None


def get_video_processing_service() -> VideoProcessingService:
    global _service
    if _service is None:
        _service = VideoProcessingService()
    return _service
