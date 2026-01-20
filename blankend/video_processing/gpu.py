from __future__ import annotations

import ctypes
import json
import os
import subprocess
import time
from dataclasses import dataclass
import logging
from typing import Any

logger = logging.getLogger("blankend.video_processing.gpu")

@dataclass(frozen=True)
class CudaVersions:
    cuda_runtime: str | None
    cudnn: str | None


def _safe_int_version(v: int | None) -> str | None:
    if v is None:
        return None
    major = v // 1000
    minor = (v % 1000) // 10
    return f"{major}.{minor}"


def _probe_cuda_runtime_version() -> str | None:
    candidates = [
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.11.0",
        "libcudart.so.10.2",
    ]
    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
        except Exception:
            continue
        try:
            fn = lib.cudaRuntimeGetVersion
            fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
            fn.restype = ctypes.c_int
            out = ctypes.c_int()
            rc = fn(ctypes.byref(out))
            if rc != 0:
                continue
            return _safe_int_version(int(out.value))
        except Exception:
            continue
    return None


def _probe_cudnn_version() -> str | None:
    candidates = [
        "libcudnn.so",
        "libcudnn.so.9",
        "libcudnn.so.8",
        "libcudnn.so.7",
    ]
    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
        except Exception:
            continue
        try:
            fn = lib.cudnnGetVersion
            fn.argtypes = []
            fn.restype = ctypes.c_size_t
            v = int(fn())
            major = v // 1000
            minor = (v % 1000) // 100
            patch = v % 100
            return f"{major}.{minor}.{patch}"
        except Exception:
            continue
    return None


def probe_cuda_versions() -> CudaVersions:
    return CudaVersions(cuda_runtime=_probe_cuda_runtime_version(), cudnn=_probe_cudnn_version())


def probe_onnxruntime() -> dict[str, Any]:
    info: dict[str, Any] = {
        "installed": False,
        "version": None,
        "device": None,
        "available_providers": [],
        "cuda_ep_available": False,
    }
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        return info
    try:
        info["installed"] = True
        info["version"] = getattr(ort, "__version__", None)
        try:
            info["device"] = ort.get_device()
        except Exception:
            info["device"] = None
        try:
            providers = ort.get_available_providers()
            info["available_providers"] = list(providers) if providers else []
            info["cuda_ep_available"] = "CUDAExecutionProvider" in info["available_providers"]
        except Exception:
            info["available_providers"] = []
            info["cuda_ep_available"] = False
        return info
    except Exception:
        return info


def _probe_nvidia_smi() -> dict[str, Any] | None:
    exe = os.getenv("NVIDIA_SMI") or "nvidia-smi"
    try:
        out = subprocess.check_output(
            [
                exe,
                "--query-gpu=index,name,temperature.gpu,memory.used,memory.total,utilization.gpu,driver_version",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
            timeout=1.5,
            text=True,
        )
    except Exception:
        return None
    gpus: list[dict[str, Any]] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        idx, name, temp, mem_used, mem_total, util, driver = parts[:7]
        try:
            gpus.append(
                {
                    "index": int(idx),
                    "name": name,
                    "temperature_c": float(temp),
                    "memory_used_mb": float(mem_used),
                    "memory_total_mb": float(mem_total),
                    "utilization_gpu_percent": float(util),
                    "driver_version": driver,
                }
            )
        except Exception:
            continue
    return {"gpus": gpus}


def _probe_nvidia_smi_versions() -> dict[str, Any] | None:
    exe = os.getenv("NVIDIA_SMI") or "nvidia-smi"
    try:
        out = subprocess.check_output([exe], stderr=subprocess.STDOUT, timeout=1.5, text=True)
    except Exception:
        return None
    driver = None
    cuda = None
    for line in out.splitlines():
        if "Driver Version" in line and "CUDA Version" in line:
            s = line
            try:
                driver = s.split("Driver Version:")[1].split()[0].strip()
            except Exception:
                driver = None
            try:
                cuda = s.split("CUDA Version:")[1].split()[0].strip()
            except Exception:
                cuda = None
            break
    return {"driver_version": driver, "cuda_version": cuda}


def _probe_pynvml() -> dict[str, Any] | None:
    try:
        import pynvml  # type: ignore
    except Exception:
        return None

    try:
        pynvml.nvmlInit()
    except Exception:
        return None

    try:
        count = int(pynvml.nvmlDeviceGetCount())
        gpus: list[dict[str, Any]] = []
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                util_gpu = float(util.gpu)
            except Exception:
                util_gpu = None
            try:
                temp = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                temp = None
            gpus.append(
                {
                    "index": i,
                    "name": name,
                    "temperature_c": temp,
                    "memory_used_mb": float(mem.used) / (1024 * 1024),
                    "memory_total_mb": float(mem.total) / (1024 * 1024),
                    "utilization_gpu_percent": util_gpu,
                }
            )
        return {"gpus": gpus}
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def probe_gpu_metrics() -> dict[str, Any]:
    metrics = _probe_pynvml()
    if metrics is not None:
        return {"source": "pynvml", **metrics}
    metrics = _probe_nvidia_smi()
    if metrics is not None:
        return {"source": "nvidia-smi", **metrics}
    return {"source": None, "gpus": []}


def probe_gpu_status() -> dict[str, Any]:
    ort = probe_onnxruntime()
    cuda_versions = probe_cuda_versions()
    metrics = probe_gpu_metrics()
    smi_v = _probe_nvidia_smi_versions() or {}
    return {
        "ts": time.time(),
        "onnxruntime": ort,
        "cuda": {
            "runtime": cuda_versions.cuda_runtime,
            "cudnn": cuda_versions.cudnn,
            "driver_version": smi_v.get("driver_version"),
            "cuda_version": smi_v.get("cuda_version"),
        },
        "metrics": metrics,
        "gpu_available": bool(ort.get("cuda_ep_available")) and (
            len(metrics.get("gpus", [])) > 0
            or str(os.getenv("CUDA_VISIBLE_DEVICES", "")).strip() not in {"", "-1", "none", "None"}
        ),
    }


def is_gpu_usable() -> tuple[bool, str]:
    status = probe_gpu_status()
    if not status["onnxruntime"]["installed"]:
        return False, "onnxruntime not installed"
    if not status["onnxruntime"]["cuda_ep_available"]:
        return False, "CUDAExecutionProvider not available"
    if status["metrics"]["gpus"]:
        return True, "ok"
    if status["gpu_available"]:
        return True, "cuda provider available (gpu metrics unavailable)"
    return False, "no NVIDIA GPU detected"


def log_gpu_summary(prefix: str = "gpu") -> None:
    try:
        status = probe_gpu_status()
        ort = status.get("onnxruntime") or {}
        cuda = status.get("cuda") or {}
        metrics = status.get("metrics") or {}
        gpus = metrics.get("gpus") or []
        logger.info(
            "%s available=%s ort_installed=%s ort_version=%s cuda_ep=%s cuda_runtime=%s cudnn=%s gpus=%s",
            prefix,
            status.get("gpu_available"),
            ort.get("installed"),
            ort.get("version"),
            ort.get("cuda_ep_available"),
            cuda.get("runtime"),
            cuda.get("cudnn"),
            len(gpus),
        )
    except Exception:
        return


def select_onnx_execution_providers(prefer_gpu: bool = True) -> dict[str, Any]:
    ort = probe_onnxruntime()
    if not ort["installed"]:
        return {
            "mode": "cpu",
            "providers": ["CPUExecutionProvider"],
            "reason": "onnxruntime not installed",
        }
    if prefer_gpu:
        ok, reason = is_gpu_usable()
        if ok:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            available = set(ort.get("available_providers") or [])
            providers = [p for p in providers if p in available] + ["CPUExecutionProvider"]
            return {"mode": "gpu", "providers": providers, "reason": reason}
        return {"mode": "cpu", "providers": ["CPUExecutionProvider"], "reason": reason}
    return {"mode": "cpu", "providers": ["CPUExecutionProvider"], "reason": "prefer_gpu=false"}
