from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _get(base: str, path: str) -> tuple[int, dict]:
    with urllib.request.urlopen(base + path, timeout=2) as f:
        return f.status, json.loads(f.read().decode("utf-8"))


def main() -> int:
    port = int(os.getenv("PORT", "8125"))
    base = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    env.setdefault("BLANKEND_RELAY_ENABLED", "0")
    env.setdefault("BLANKEND_VIDEO_PROCESSING_ENABLED", "0")

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "blankend.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        for _ in range(20):
            try:
                code, body = _get(base, "/")
                if code == 200 and body.get("status") == "ok":
                    break
            except Exception:
                time.sleep(0.2)
        else:
            raise RuntimeError("server did not become ready")

        code, body = _get(base, "/api/v1/video_processing/gpu/status")
        assert code == 200 and "onnxruntime" in body

        code, body = _get(base, "/api/v1/video_processing/pipeline/status")
        assert code == 200 and "status" in body

        code, body = _get(base, "/api/v1/skills")
        assert code == 200 and isinstance(body, list)

        print("smoke: OK")
        return 0
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

