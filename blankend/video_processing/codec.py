from __future__ import annotations

import os
import subprocess
import time
from typing import Any


def _run(cmd: list[str], timeout_s: float = 1.5) -> tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout_s, text=True)
        return 0, out
    except subprocess.CalledProcessError as e:
        return int(e.returncode or 1), str(e.output or "")
    except Exception as e:
        return 1, str(e)


def probe_ffmpeg() -> dict[str, Any]:
    exe = os.getenv("FFMPEG") or "ffmpeg"
    ok, out = _run([exe, "-hide_banner", "-version"])
    if ok != 0:
        return {"installed": False, "exe": exe, "error": out.strip()}
    first = out.splitlines()[0].strip() if out else ""
    return {"installed": True, "exe": exe, "version_line": first}


def probe_hw_codecs() -> dict[str, Any]:
    exe = os.getenv("FFMPEG") or "ffmpeg"
    ffmpeg = probe_ffmpeg()
    if not ffmpeg.get("installed"):
        return {"ts": time.time(), "ffmpeg": ffmpeg, "hwaccels": [], "encoders": [], "decoders": []}

    _, hw = _run([exe, "-hide_banner", "-hwaccels"])
    hwaccels: list[str] = []
    for line in (hw or "").splitlines():
        s = line.strip()
        if not s or s.lower().startswith("hardware acceleration"):
            continue
        hwaccels.append(s)

    _, enc = _run([exe, "-hide_banner", "-encoders"])
    encoders: list[str] = []
    for line in (enc or "").splitlines():
        s = line.strip()
        if not s or s.startswith("--") or s.lower().startswith("encoders"):
            continue
        parts = s.split()
        if len(parts) >= 2:
            encoders.append(parts[1])

    _, dec = _run([exe, "-hide_banner", "-decoders"])
    decoders: list[str] = []
    for line in (dec or "").splitlines():
        s = line.strip()
        if not s or s.startswith("--") or s.lower().startswith("decoders"):
            continue
        parts = s.split()
        if len(parts) >= 2:
            decoders.append(parts[1])

    def has(name: str, arr: list[str]) -> bool:
        return name in set(arr)

    return {
        "ts": time.time(),
        "ffmpeg": ffmpeg,
        "hwaccels": hwaccels,
        "encoders": encoders,
        "decoders": decoders,
        "nvidia": {
            "h264_nvenc": has("h264_nvenc", encoders),
            "hevc_nvenc": has("hevc_nvenc", encoders),
            "h264_cuvid": has("h264_cuvid", decoders),
            "hevc_cuvid": has("hevc_cuvid", decoders),
            "cuda_hwaccel": "cuda" in set(hwaccels),
        },
    }

