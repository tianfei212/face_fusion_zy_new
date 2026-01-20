from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AssetPaths:
    portraits_dir: Path
    scenes_dir: Path


def default_asset_paths() -> AssetPaths:
    base = Path(__file__).resolve().parents[1] / "assets"
    return AssetPaths(portraits_dir=base / "human_pic", scenes_dir=base / "m_blank")


def resolve_asset_file(dir_path: Path, name: str) -> Path | None:
    p = dir_path / name
    if p.is_file():
        return p
    stem = Path(name).stem
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        c = dir_path / f"{stem}{ext}"
        if c.is_file():
            return c
    return None

