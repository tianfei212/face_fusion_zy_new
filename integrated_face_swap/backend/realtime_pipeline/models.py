from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .onnx_runtime import OrtProvidersConfig, create_inference_session, resolve_checkpoint_path
from .preprocess import pick_output, pick_src_input_name, resolve_dims


@dataclass
class RvmState:
    tensors: dict[str, np.ndarray]
    width: int
    height: int


class RvmModel:
    def __init__(
        self,
        model_path: str,
        force_cuda: bool = True,
        providers_cfg: OrtProvidersConfig | None = None,
        use_tensorrt: bool = False,
    ) -> None:
        self.model_path = model_path
        self.force_cuda = force_cuda
        self.sess, self.meta = create_inference_session(
            model_path,
            force_cuda=force_cuda,
            cfg=providers_cfg,
            use_tensorrt=use_tensorrt,
        )
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.src_name = pick_src_input_name(self.input_names)
        self.state_names = [n for n in self.input_names if n != self.src_name]

    def _input_dtype(self) -> Any:
        try:
            t = self.sess.get_inputs()[0].type
            if "float16" in str(t):
                return np.float16
        except Exception:
            pass
        return np.float32

    def ensure_state(self, prev: RvmState | None, width: int, height: int, downsample_ratio: float = 0.25) -> RvmState:
        if prev is not None and prev.width == width and prev.height == height:
            return prev
        base_h = int(max(1, int(float(height) * float(downsample_ratio))))
        base_w = int(max(1, int(float(width) * float(downsample_ratio))))
        tensors: dict[str, np.ndarray] = {}
        div_by_ch = {16: 2, 20: 4, 40: 8, 64: 16}
        for name in self.state_names:
            if "downsample" in name.lower():
                continue
            meta = next((i for i in self.sess.get_inputs() if i.name == name), None)
            ch = 1
            try:
                if meta is not None and isinstance(getattr(meta, "shape", None), (list, tuple)):
                    s = list(getattr(meta, "shape"))
                    if len(s) >= 2 and isinstance(s[1], int) and s[1] > 0:
                        ch = int(s[1])
            except Exception:
                ch = 1
            div = div_by_ch.get(ch, 4)
            h2 = int(max(1, base_h // int(div)))
            w2 = int(max(1, base_w // int(div)))
            dims = [1, int(ch), int(h2), int(w2)]
            tensors[name] = np.zeros(tuple(dims), dtype=np.float32)
        return RvmState(tensors=tensors, width=width, height=height)

    def run(self, src_chw: np.ndarray, state: RvmState, downsample_ratio: float = 0.25) -> tuple[np.ndarray, np.ndarray | None, RvmState, dict[str, Any]]:
        feeds: dict[str, Any] = {self.src_name: src_chw}
        for name in self.state_names:
            if "downsample" in name.lower():
                feeds[name] = np.array([downsample_ratio], dtype=np.float32)
            else:
                feeds[name] = state.tensors[name]
        t0 = time.perf_counter()
        out = self.sess.run(None, feeds)
        t1 = time.perf_counter()
        outputs = {self.output_names[i]: out[i] for i in range(len(out))}
        for n in self.state_names:
            if n in outputs and "downsample" not in n.lower():
                state.tensors[n] = outputs[n]
        alpha = pick_output(outputs, ["pha", "alpha", "mask"])
        fgr = outputs.get("fgr")
        if fgr is None:
            fgr = outputs.get("foreground")
        diag = {"rvm_ms": (t1 - t0) * 1000.0}
        return alpha, fgr, state, diag


class DepthModel:
    def __init__(
        self,
        model_path: str,
        force_cuda: bool = True,
        providers_cfg: OrtProvidersConfig | None = None,
        use_tensorrt: bool = False,
    ) -> None:
        self.model_path = model_path
        self.force_cuda = force_cuda
        self.sess, self.meta = create_inference_session(
            model_path,
            force_cuda=force_cuda,
            cfg=providers_cfg,
            use_tensorrt=use_tensorrt,
        )
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.src_name = pick_src_input_name(self.input_names)

    def _input_dtype(self) -> Any:
        try:
            t = self.sess.get_inputs()[0].type
            if "float16" in str(t):
                return np.float16
        except Exception:
            pass
        return np.float32

    def run(self, src_chw: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        t0 = time.perf_counter()
        out = self.sess.run(None, {self.src_name: src_chw})
        t1 = time.perf_counter()
        outputs = {self.output_names[i]: out[i] for i in range(len(out))}
        depth = pick_output(outputs, ["depth", "pred", "output", "disp"])
        diag = {"depth_ms": (t1 - t0) * 1000.0}
        return depth, diag


def load_default_models(force_cuda: bool = True) -> tuple[RvmModel, DepthModel]:
    rvm_path = resolve_checkpoint_path("rvm_mobilenetv3_fp32.onnx")
    depth_path = resolve_checkpoint_path("depth_anything_v2_vits_fp32.onnx")
    return RvmModel(rvm_path, force_cuda=force_cuda), DepthModel(depth_path, force_cuda=force_cuda)
