from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import logging

logger = logging.getLogger("blankend.video_processing.hwdecode")


@dataclass(frozen=True)
class HwDecodeConfig:
    codec: str
    prefer_gpu: bool = True
    width: int | None = None
    height: int | None = None
    mjpeg_quality: int = 5


def _build_ffmpeg_cmd(cfg: HwDecodeConfig, use_gpu: bool) -> list[str]:
    exe = os.getenv("FFMPEG") or "ffmpeg"
    codec = cfg.codec.lower().strip()
    if codec in {"h264", "avc"}:
        demux = "h264"
        cuvid = "h264_cuvid"
    elif codec in {"hevc", "h265"}:
        demux = "hevc"
        cuvid = "hevc_cuvid"
    else:
        demux = codec
        cuvid = codec

    cmd = [exe, "-hide_banner", "-loglevel", "error"]
    if use_gpu:
        cmd += ["-hwaccel", "cuda", "-c:v", cuvid]
    cmd += ["-f", demux, "-i", "pipe:0"]
    if cfg.width and cfg.height:
        cmd += ["-vf", f"scale={int(cfg.width)}:{int(cfg.height)}"]
    cmd += ["-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", str(int(cfg.mjpeg_quality)), "pipe:1"]
    return cmd


def _extract_jpegs(buffer: bytearray) -> list[bytes]:
    out: list[bytes] = []
    while True:
        start = buffer.find(b"\xff\xd8")
        if start < 0:
            if len(buffer) > 2 * 1024 * 1024:
                del buffer[:-4096]
            return out
        if start > 0:
            del buffer[:start]
        end = buffer.find(b"\xff\xd9", 2)
        if end < 0:
            return out
        frame = bytes(buffer[: end + 2])
        del buffer[: end + 2]
        out.append(frame)


class FfmpegHwDecoder:
    def __init__(self, cfg: HwDecodeConfig) -> None:
        self.cfg = cfg
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._out_q: queue.Queue[bytes] = queue.Queue(maxsize=64)
        self._last_err: str | None = None
        self._use_gpu = False

    def start(self) -> None:
        with self._lock:
            if self._proc is not None:
                return
            ok, reason = self._try_start(prefer_gpu=self.cfg.prefer_gpu)
            if not ok:
                raise RuntimeError(reason)

    def _try_start(self, prefer_gpu: bool) -> tuple[bool, str]:
        attempts: list[tuple[bool, str]] = []
        if prefer_gpu:
            attempts.append((True, "gpu"))
        attempts.append((False, "cpu"))
        last_err = "unknown"
        for use_gpu, mode in attempts:
            cmd = _build_ffmpeg_cmd(self.cfg, use_gpu=use_gpu)
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )
            except Exception as e:
                last_err = f"{mode} spawn failed: {type(e).__name__}: {e}"
                continue
            time.sleep(0.05)
            if proc.poll() is not None:
                try:
                    err = (proc.stderr.read(2048) if proc.stderr else b"") if hasattr(proc.stderr, "read") else b""
                    if isinstance(err, (bytes, bytearray)):
                        err_s = err.decode("utf-8", errors="replace").strip()
                    else:
                        err_s = str(err).strip()
                except Exception:
                    err_s = ""
                last_err = f"{mode} ffmpeg exited rc={proc.returncode} {err_s}".strip()
                continue
            self._proc = proc
            self._use_gpu = use_gpu
            self._stop.clear()
            self._stdout_thread = threading.Thread(target=self._read_stdout_loop, name="ffmpeg-stdout", daemon=True)
            self._stderr_thread = threading.Thread(target=self._read_stderr_loop, name="ffmpeg-stderr", daemon=True)
            self._stdout_thread.start()
            self._stderr_thread.start()
            logger.info("ffmpeg hwdecode started mode=%s codec=%s", mode, self.cfg.codec)
            return True, "ok"
        return False, last_err

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            proc = self._proc
            self._proc = None
        if proc is not None:
            try:
                if proc.stdin:
                    try:
                        proc.stdin.close()
                    except Exception:
                        pass
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def feed(self, data: bytes) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            return
        try:
            proc.stdin.write(data)
        except Exception:
            return

    def frames(self) -> Iterable[bytes]:
        while not self._stop.is_set():
            try:
                yield self._out_q.get(timeout=0.25)
            except queue.Empty:
                proc = self._proc
                if proc is not None and proc.poll() is not None:
                    return
                continue

    def status(self) -> dict:
        proc = self._proc
        return {
            "running": proc is not None and proc.poll() is None,
            "mode": "gpu" if self._use_gpu else "cpu",
            "codec": self.cfg.codec,
            "last_error": self._last_err,
        }

    def _read_stdout_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        buf = bytearray()
        while not self._stop.is_set():
            try:
                chunk = proc.stdout.read(4096)
            except Exception:
                break
            if not chunk:
                break
            buf.extend(chunk)
            frames = _extract_jpegs(buf)
            for f in frames:
                try:
                    self._out_q.put_nowait(f)
                except queue.Full:
                    try:
                        _ = self._out_q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._out_q.put_nowait(f)
                    except queue.Full:
                        pass

    def _read_stderr_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        while not self._stop.is_set():
            try:
                line = proc.stderr.readline()
            except Exception:
                break
            if not line:
                break
            s = line.decode("utf-8", errors="replace").strip() if isinstance(line, (bytes, bytearray)) else str(line).strip()
            if s:
                self._last_err = s


def run_decode_session(
    decoder: FfmpegHwDecoder,
    stop_evt: threading.Event,
    broadcast: Callable[[bytes], None],
) -> None:
    last = time.monotonic()
    frames = 0
    while not stop_evt.is_set():
        for frame in decoder.frames():
            if stop_evt.is_set():
                return
            broadcast(frame)
            frames += 1
            now = time.monotonic()
            if now - last >= 1.0:
                logger.info("ffmpeg decoded fps=%s mode=%s", frames, decoder.status().get("mode"))
                frames = 0
                last = now
        return
