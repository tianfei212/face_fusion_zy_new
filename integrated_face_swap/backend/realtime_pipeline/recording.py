from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecordingSpec:
    format: str = "mp4"
    fps: int = 30
    width: int | None = None
    height: int | None = None
    out_path: str | None = None


class FfmpegRecorder:
    def __init__(self, spec: RecordingSpec) -> None:
        self.spec = spec
        self._proc: subprocess.Popen | None = None
        self._started_at = 0.0

    def start(self) -> str:
        if self._proc is not None:
            return self.output_path()
        exe = os.getenv("FFMPEG") or "ffmpeg"
        fmt = (self.spec.format or "mp4").lower()
        out = self.spec.out_path
        if not out:
            out_dir = Path(os.getenv("RECORDINGS_DIR") or "outputs")
            out_dir.mkdir(parents=True, exist_ok=True)
            suffix = "mp4" if fmt == "mp4" else "gif"
            out = str(out_dir / f"recording_{int(time.time()*1000)}.{suffix}")
        self.spec = RecordingSpec(
            format=fmt,
            fps=int(self.spec.fps),
            width=self.spec.width,
            height=self.spec.height,
            out_path=out,
        )

        cmd = [exe, "-hide_banner", "-loglevel", "error", "-y"]
        cmd += ["-f", "mjpeg", "-r", str(int(self.spec.fps)), "-i", "pipe:0"]
        if fmt == "gif":
            cmd += ["-vf", "fps=15,scale=iw:-1:flags=lanczos", out]
        else:
            cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", out]

        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._started_at = time.time()
        return out

    def write_jpeg(self, jpeg_bytes: bytes) -> None:
        if self._proc is None or self._proc.stdin is None:
            return
        try:
            self._proc.stdin.write(jpeg_bytes)
        except Exception:
            return

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
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
            proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def running(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    def output_path(self) -> str:
        return self.spec.out_path or ""
