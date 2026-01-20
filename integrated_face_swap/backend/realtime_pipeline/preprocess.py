from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ImageTensor:
    chw: np.ndarray
    rgb_u8: np.ndarray
    width: int
    height: int


def chw_from_rgb_u8(rgb: np.ndarray, dtype: Any) -> np.ndarray:
    h, w, _ = rgb.shape
    x = rgb.astype(np.float32) / 255.0
    chw = np.transpose(x, (2, 0, 1)).reshape(1, 3, h, w)
    if dtype == np.float16:
        return chw.astype(np.float16, copy=False)
    return chw.astype(np.float32, copy=False)


def pick_src_input_name(names: list[str]) -> str:
    for n in names:
        if "src" in n.lower():
            return n
    for n in names:
        if "input" in n.lower():
            return n
    return names[0]


def pick_output(outputs: dict, keys: list[str]):
    for k in keys:
        if k in outputs:
            return outputs[k]
    if outputs:
        return outputs[next(iter(outputs.keys()))]
    return None


def resolve_dims(dims: list[Any], width: int, height: int) -> list[int]:
    out: list[int] = []
    for i, d in enumerate(dims):
        if isinstance(d, int) and d > 0:
            out.append(d)
        elif isinstance(d, str):
            s = d.strip().lower()
            if s.startswith("h") and i == 2:
                try:
                    factor = int(s[1:]) if len(s) > 1 else 1
                    factor = max(1, factor)
                    out.append(int(max(1, height // factor)))
                except Exception:
                    out.append(int(height))
            elif s.startswith("w") and i == 3:
                try:
                    factor = int(s[1:]) if len(s) > 1 else 1
                    factor = max(1, factor)
                    out.append(int(max(1, width // factor)))
                except Exception:
                    out.append(int(width))
            else:
                out.append(1)
        elif i == 2:
            out.append(int(height))
        elif i == 3:
            out.append(int(width))
        else:
            out.append(1)
    return out
