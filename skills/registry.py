from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable

from .contracts import SkillDescriptor


@dataclass(frozen=True)
class Skill:
    descriptor: SkillDescriptor
    invoke: Callable[[dict[str, Any]], Any]


class SkillRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        with self._lock:
            self._skills[skill.descriptor.id] = skill

    def list(self) -> list[SkillDescriptor]:
        with self._lock:
            return [s.descriptor for s in self._skills.values()]

    def get(self, skill_id: str) -> Skill | None:
        with self._lock:
            return self._skills.get(skill_id)


_registry: SkillRegistry | None = None


def get_registry() -> SkillRegistry:
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
        _register_builtin(_registry)
    return _registry


def _register_builtin(reg: SkillRegistry) -> None:
    from .contracts import SkillDescriptor

    reg.register(
        Skill(
            descriptor=SkillDescriptor(
                id="echo",
                name="Echo",
                version="1.0.0",
                description="Return params as-is. Useful for integration testing.",
            ),
            invoke=lambda params: params,
        )
    )

    def _gpu_status(_: dict[str, Any]) -> Any:
        from blankend.video_processing.gpu import probe_gpu_status

        return probe_gpu_status()

    reg.register(
        Skill(
            descriptor=SkillDescriptor(
                id="gpu_status",
                name="GPU Status",
                version="1.0.0",
                description="Return GPU availability, CUDA/cuDNN versions and metrics.",
            ),
            invoke=_gpu_status,
        )
    )

    def _pipeline_status(_: dict[str, Any]) -> Any:
        from blankend.video_processing.service import get_video_processing_service

        return get_video_processing_service().status()

    reg.register(
        Skill(
            descriptor=SkillDescriptor(
                id="video_pipeline_status",
                name="Video Pipeline Status",
                version="1.0.0",
                description="Return producer-consumer pipeline status and thread health.",
            ),
            invoke=_pipeline_status,
        )
    )

