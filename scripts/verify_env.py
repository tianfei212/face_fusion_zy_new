from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CheckResult:
    ok: bool
    details: dict[str, Any]


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return 0, out
    except subprocess.CalledProcessError as e:
        return int(e.returncode or 1), str(e.output or "")
    except Exception as e:
        return 1, str(e)


def check_conda_env(expected: str = "ai_face_change") -> CheckResult:
    env = os.getenv("CONDA_DEFAULT_ENV") or ""
    prefix = os.getenv("CONDA_PREFIX") or ""
    ok = env == expected
    return CheckResult(
        ok=ok,
        details={
            "expected": expected,
            "CONDA_DEFAULT_ENV": env,
            "CONDA_PREFIX": prefix,
            "python": sys.executable,
        },
    )


def check_import(name: str) -> CheckResult:
    try:
        __import__(name)
        return CheckResult(ok=True, details={"module": name})
    except Exception as e:
        return CheckResult(ok=False, details={"module": name, "error": f"{type(e).__name__}: {e}"})


def check_onnxruntime_gpu(min_version: str = "1.12.0") -> CheckResult:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as e:
        return CheckResult(ok=False, details={"installed": False, "error": f"{type(e).__name__}: {e}"})

    ver = getattr(ort, "__version__", None)
    try:
        providers = list(ort.get_available_providers())
    except Exception:
        providers = []
    cuda_ep = "CUDAExecutionProvider" in providers

    return CheckResult(
        ok=bool(cuda_ep),
        details={
            "installed": True,
            "version": ver,
            "min_required": min_version,
            "available_providers": providers,
            "cuda_ep_available": cuda_ep,
        },
    )


def pip_install(pkgs: list[str]) -> CheckResult:
    code, out = _run([sys.executable, "-m", "pip", "install", *pkgs])
    return CheckResult(ok=code == 0, details={"cmd": "pip install " + " ".join(pkgs), "output": out})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true", help="Auto install missing dependencies via pip")
    ap.add_argument("--json", action="store_true", help="Print json output")
    args = ap.parse_args()

    report: dict[str, Any] = {
        "platform": {"python": sys.version, "os": platform.platform()},
        "checks": {},
        "install": {},
    }

    conda = check_conda_env()
    report["checks"]["conda_env"] = asdict(conda)

    deps = {
        "pyzmq": "zmq",
        "turbojpeg": "turbojpeg",
        "pynvml": "pynvml",
        "httpx": "httpx",
    }
    for key, mod in deps.items():
        report["checks"][key] = asdict(check_import(mod))

    ort = check_onnxruntime_gpu()
    report["checks"]["onnxruntime_gpu"] = asdict(ort)

    code, out = _run([os.getenv("FFMPEG") or "ffmpeg", "-hide_banner", "-version"])
    report["checks"]["ffmpeg"] = asdict(CheckResult(ok=code == 0, details={"output": (out.splitlines()[:1] or [""])[0]}))

    missing: list[str] = []
    if not report["checks"]["onnxruntime_gpu"]["ok"]:
        missing.append("onnxruntime-gpu>=1.12.0")
    if not report["checks"]["httpx"]["ok"]:
        missing.append("httpx>=0.24.0")
    if not report["checks"]["pynvml"]["ok"]:
        missing.append("pynvml>=11.5.0")

    report["missing_suggested"] = missing

    if args.install and missing:
        report["install"]["result"] = asdict(pip_install(missing))
        report["checks"]["onnxruntime_gpu_after"] = asdict(check_onnxruntime_gpu())

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for k, v in report["checks"].items():
            ok = "OK" if v["ok"] else "FAIL"
            print(f"{k}: {ok}")
        if missing:
            print("missing_suggested:", ", ".join(missing))

    return 0 if all(v["ok"] for v in report["checks"].values() if isinstance(v, dict) and "ok" in v) else 1


if __name__ == "__main__":
    raise SystemExit(main())
