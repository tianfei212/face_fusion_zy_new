from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("blankend.ai_core.asset_manager")

@dataclass(frozen=True)
class AssetPaths:
    portraits_dir: Path
    scenes_dir: Path
    dfm_dir: Path

def default_asset_paths() -> AssetPaths:
    base = Path(__file__).resolve().parents[1] / "assets"
    repo_root = Path(__file__).resolve().parents[2]
    checkpoints_root = Path(os.getenv("CHECKPOINT_ROOT") or (repo_root / "checkpoints"))
    dfm_dir = checkpoints_root / "DFM"
    if not dfm_dir.exists():
        dfm_dir = base / "DFM"
    return AssetPaths(
        portraits_dir=base / "human_pic",
        scenes_dir=base / "m_blank",
        dfm_dir=dfm_dir,
    )

def resolve_asset_file(dir_path: Path, name: str) -> Optional[Path]:
    """
    Resolves a file in a directory.
    If 'name' has no extension, tries common image extensions.
    """
    if not name:
        return None
        
    p = dir_path / name
    if p.is_file():
        return p
    
    # Try exact match first
    # Then try extensions if no extension provided
    if not p.suffix:
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".onnx"):
            c = dir_path / f"{name}{ext}"
            if c.is_file():
                return c
    
    # Fallback: try matching stem if name has extension but file not found
    # e.g. name="001" (from frontend id) -> "001.png"
    # or name="001.jpg" -> "001.png" (if ext mismatch)
    stem = Path(name).stem
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".onnx"):
        c = dir_path / f"{stem}{ext}"
        if c.is_file():
            return c
            
    return None

class AssetManager:
    """
    Manages access to static assets (Portraits, Backgrounds, DFM Models).
    """
    def __init__(self, paths: Optional[AssetPaths] = None):
        self.paths = paths or default_asset_paths()
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Ensures that asset directories exist."""
        for p in [self.paths.portraits_dir, self.paths.scenes_dir, self.paths.dfm_dir]:
            if not p.exists():
                try:
                    p.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to create asset dir {p}: {e}")

    def list_assets(self, asset_type: str) -> List[str]:
        """
        Lists assets of a given type.
        
        Args:
            asset_type: 'face', 'bg', or 'dfm'
        
        Returns:
            List of filenames sorted alphabetically.
        """
        target_dir = None
        extensions = {".png", ".jpg", ".jpeg", ".webp"}

        if asset_type == "face":
            target_dir = self.paths.portraits_dir
        elif asset_type == "bg":
            target_dir = self.paths.scenes_dir
        elif asset_type == "dfm":
            target_dir = self.paths.dfm_dir
            extensions = {".onnx"}
        else:
            return []

        if not target_dir.exists():
            return []

        files = []
        try:
            for p in target_dir.iterdir():
                if p.is_file() and p.suffix.lower() in extensions:
                    files.append(p.name)
        except Exception as e:
            logger.error(f"Error listing assets in {target_dir}: {e}")
            return []
    
        return sorted(files)

    def resolve_path(self, asset_type: str, filename: str) -> Optional[Path]:
        """
        Resolves the absolute path for an asset.
        """
        target_dir = None
        if asset_type == "face":
            target_dir = self.paths.portraits_dir
        elif asset_type == "bg":
            target_dir = self.paths.scenes_dir
        elif asset_type == "dfm":
            target_dir = self.paths.dfm_dir
        
        if target_dir:
            return resolve_asset_file(target_dir, filename)
        return None

# Singleton instance for convenience if needed, but class usage is preferred for DI
_default_manager = None

def get_asset_manager() -> AssetManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = AssetManager()
    return _default_manager


def list_assets(asset_type: str) -> List[str]:
    return get_asset_manager().list_assets(asset_type)
