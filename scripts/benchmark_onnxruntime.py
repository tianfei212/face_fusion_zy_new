from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _pick_dim(dim: Any, default: int) -> int:
    if isinstance(dim, int) and dim > 0:
        return dim
    return default


def _make_dummy_array(dtype: str, shape: list[int]):
    import numpy as np  # type: ignore

    if dtype in {"tensor(float)", "float"}:
        return np.zeros(shape, dtype=np.float32)
    if dtype in {"tensor(float16)"}:
        return np.zeros(shape, dtype=np.float16)
    if dtype in {"tensor(int64)"}:
        return np.zeros(shape, dtype=np.int64)
    if dtype in {"tensor(int32)"}:
        return np.zeros(shape, dtype=np.int32)
    return np.zeros(shape, dtype=np.float32)


def benchmark(model_path: Path, providers: list[str], runs: int, warmup: int) -> dict[str, Any]:
    import onnxruntime as ort  # type: ignore

    sess = ort.InferenceSession(str(model_path), providers=providers)
    inputs = sess.get_inputs()
    feed: dict[str, Any] = {}
    for inp in inputs:
        dims = list(inp.shape)
        shape: list[int] = []
        for i, d in enumerate(dims):
            if i in (2, 3):
                shape.append(_pick_dim(d, 256))
            else:
                shape.append(_pick_dim(d, 1))
        feed[inp.name] = _make_dummy_array(inp.type, shape)

    for _ in range(max(0, warmup)):
        _ = sess.run(None, feed)

    t0 = time.perf_counter()
    for _ in range(max(1, runs)):
        _ = sess.run(None, feed)
    t1 = time.perf_counter()

    return {
        "providers": providers,
        "inputs": [{"name": i.name, "type": i.type, "shape": list(i.shape)} for i in inputs],
        "outputs": [o.name for o in sess.get_outputs()],
        "runs": runs,
        "warmup": warmup,
        "total_s": float(t1 - t0),
        "avg_ms": float((t1 - t0) * 1000 / max(1, runs)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/depth_anything_v2_vits_fp32.onnx")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    model = Path(args.model)
    if not model.exists():
        raise SystemExit(f"model not found: {model}")

    try:
        import onnxruntime as ort  # type: ignore
    except Exception as e:
        raise SystemExit(f"onnxruntime not available: {e}")

    providers = list(getattr(ort, "get_available_providers")() or [])
    results: dict[str, Any] = {"model": str(model), "available_providers": providers, "bench": {}}

    results["bench"]["cpu"] = benchmark(model, ["CPUExecutionProvider"], runs=args.runs, warmup=args.warmup)
    if "CUDAExecutionProvider" in providers:
        results["bench"]["gpu"] = benchmark(
            model, ["CUDAExecutionProvider", "CPUExecutionProvider"], runs=args.runs, warmup=args.warmup
        )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print("model:", model)
        print("available_providers:", providers)
        print("cpu avg_ms:", results["bench"]["cpu"]["avg_ms"])
        if "gpu" in results["bench"]:
            print("gpu avg_ms:", results["bench"]["gpu"]["avg_ms"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

