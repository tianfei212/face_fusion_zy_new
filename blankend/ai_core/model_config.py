from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import logging

logger = logging.getLogger("blankend.ai_core.model_config")


@dataclass(frozen=True)
class AiModelConfig:
    rvm_model: str = "rvm_mobilenetv3_fp32.onnx"
    depth_model: str = "depth_anything_v2_vits_fp32.onnx"
    swap_model: str = "inswapper_128_fp16.onnx"
    insightface_app: str = "buffalo_l"
    preload_dfm: bool = False
    dfm_model: str = ""


def _load_blankend_config() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "config.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_ai_model_config() -> AiModelConfig:
    cfg = _load_blankend_config()
    ai = cfg.get("ai", {}) if isinstance(cfg, dict) else {}
    models = ai.get("models", {}) if isinstance(ai, dict) else {}

    def _get_str(key: str, default: str) -> str:
        v = models.get(key, default) if isinstance(models, dict) else default
        if v is None:
            return default
        return str(v).strip() or default

    preload = ai.get("preload", {}) if isinstance(ai, dict) else {}
    preload_dfm = bool(preload.get("dfm", False)) if isinstance(preload, dict) else False
    dfm_model = ""
    if isinstance(preload, dict) and preload.get("dfm_model") is not None:
        dfm_model = str(preload.get("dfm_model") or "").strip()

    return AiModelConfig(
        rvm_model=_get_str("rvm", AiModelConfig.rvm_model),
        depth_model=_get_str("depth", AiModelConfig.depth_model),
        swap_model=_get_str("swap", AiModelConfig.swap_model),
        insightface_app=_get_str("insightface_app", AiModelConfig.insightface_app),
        preload_dfm=preload_dfm,
        dfm_model=dfm_model,
    )


def apply_models_env_from_config() -> AiModelConfig:
    cfg = load_ai_model_config()
    os.environ.setdefault("RVM_MODEL", cfg.rvm_model)
    os.environ.setdefault("DEPTH_MODEL", cfg.depth_model)
    os.environ.setdefault("SWAP_MODEL", cfg.swap_model)
    os.environ.setdefault("INSIGHTFACE_APP", cfg.insightface_app)
    if cfg.preload_dfm and cfg.dfm_model:
        os.environ.setdefault("DFM_MODEL", cfg.dfm_model)
    return cfg


def log_ai_model_config(cfg: AiModelConfig, prefix: str = "ai_models") -> None:
    logger.info(
        "%s rvm=%s depth=%s swap=%s insightface_app=%s preload_dfm=%s dfm_model=%s",
        prefix,
        cfg.rvm_model,
        cfg.depth_model,
        cfg.swap_model,
        cfg.insightface_app,
        bool(cfg.preload_dfm),
        cfg.dfm_model,
    )

