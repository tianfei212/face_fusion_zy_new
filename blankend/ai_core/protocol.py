from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

class CommandType(Enum):
    SET_FACE = "SET_FACE"  # Payload: filename
    SET_BG = "SET_BG"  # Payload: filename
    LOAD_DFM = "LOAD_DFM"  # Payload: filename
    ACTIVATE = "ACTIVATE"  # Payload: {"active": bool} | None
    READY = "READY"  # Payload: {"request_id": str, "detail": ...}
    UPDATE_CONFIG = "UPDATE_CONFIG"  # Payload: dict (update parameters)

    SET_FACE_IMAGE = "SET_FACE_IMAGE"  # Payload: filename
    SET_FACE_DFM = "SET_FACE_DFM"  # Payload: filename
    SET_BACKGROUND = "SET_BACKGROUND"  # Payload: filename
    LOAD_MODEL = "LOAD_MODEL"  # Payload: filename (preload model)
    UNLOAD_MODEL = "UNLOAD_MODEL"  # Payload: None (release VRAM)

@dataclass
class WorkerCommand:
    type: CommandType
    payload: Any | None = None
    request_id: str | None = None
    ts: float | None = None
