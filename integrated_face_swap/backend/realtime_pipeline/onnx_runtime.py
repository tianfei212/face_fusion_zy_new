from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OrtProvidersConfig:
    device_id: int = 0
    arena_extend_strategy: str = "kNextPowerOfTwo"
    cudnn_conv_algo_search: str = "EXHAUSTIVE"
    do_copy_in_default_stream: bool = True
    trt_engine_cache_enable: bool = True
    trt_engine_cache_path: str | None = None


def build_providers(force_cuda: bool = True, cfg: OrtProvidersConfig | None = None, use_tensorrt: bool = False) -> list[Any]:
    if not force_cuda:
        return ["CPUExecutionProvider"]
    c = cfg or OrtProvidersConfig()
    cuda = (
        "CUDAExecutionProvider",
        {
            "device_id": int(c.device_id),
            "arena_extend_strategy": str(c.arena_extend_strategy),
            "cudnn_conv_algo_search": str(c.cudnn_conv_algo_search),
            "do_copy_in_default_stream": bool(c.do_copy_in_default_stream),
        },
    )
    cpu = "CPUExecutionProvider"
    if not use_tensorrt:
        return [cuda, cpu]
    trt_opts: dict[str, Any] = {"device_id": int(c.device_id)}
    if c.trt_engine_cache_enable:
        trt_opts["trt_engine_cache_enable"] = True
        if c.trt_engine_cache_path:
            trt_opts["trt_engine_cache_path"] = str(c.trt_engine_cache_path)
    trt = ("TensorrtExecutionProvider", trt_opts)
    return [trt, cuda, cpu]


def create_inference_session(
    model_path: str,
    force_cuda: bool = True,
    cfg: OrtProvidersConfig | None = None,
    use_tensorrt: bool = False,
) -> tuple[Any, dict[str, Any]]:
    import onnxruntime as ort  # type: ignore

    providers = build_providers(force_cuda=force_cuda, cfg=cfg, use_tensorrt=use_tensorrt)
    sess_options = ort.SessionOptions()
    try:
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    except Exception:
        pass
    try:
        sess = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        meta = {
            "model_path": model_path,
            "providers_requested": providers,
            "providers_active": sess.get_providers() if hasattr(sess, "get_providers") else None,
            "created_at": time.time(),
            "cuda_failed": None,
        }
        return sess, meta
    except Exception as e:
        if not force_cuda:
            raise
        cpu_sess = ort.InferenceSession(model_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
        meta = {
            "model_path": model_path,
            "providers_requested": providers,
            "providers_active": cpu_sess.get_providers() if hasattr(cpu_sess, "get_providers") else None,
            "created_at": time.time(),
            "cuda_failed": f"{type(e).__name__}: {e}",
        }
        return cpu_sess, meta


def resolve_checkpoint_path(filename: str) -> str:
    root = os.getenv("CHECKPOINT_ROOT") or str((__import__("pathlib").Path(__file__).resolve().parents[2] / "checkpoints"))
    return str(__import__("pathlib").Path(root) / filename)
