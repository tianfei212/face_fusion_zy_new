from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional, Tuple, Dict, List

import cv2
import numpy as np

from ..realtime_pipeline.models import DepthModel, RvmModel, RvmState
from ..realtime_pipeline.onnx_runtime import OrtProvidersConfig, resolve_checkpoint_path
from ..realtime_pipeline.preprocess import chw_from_rgb_u8
from .resources import default_asset_paths, resolve_asset_file
from blankend.modules.face_swapper.model_manager import ModelManager
from blankend.modules.face_swapper.core_inference import CoreInference, Face

logger = logging.getLogger("blankend.ai_core.engine")


@dataclass
class EngineConfig:
    similarity_threshold: float = 0.20
    depth_enabled: bool = True
    light_fusion_enabled: bool = False
    matting_enabled: bool = False
    swap_stride: int = 1
    rvm_downsample_ratio: float = 0.25
    jpeg_quality: int = 85
    depth_threshold: float = 0.5
    depth_fusion_size: int = 518


class UnifiedInferenceEngine:
    def __init__(self, device_id: int, lazy_init: bool = False):
        self.device_id = device_id
        self.cfg = EngineConfig()

        # Assets & State
        self._assets = default_asset_paths()
        self._source_face: Optional[Face] = None
        self._portrait_bgr: Optional[np.ndarray] = None
        self._scene_rgb: Optional[np.ndarray] = None
        self._rvm_state_by_client: Dict[bytes, RvmState] = {}
        
        # Face Detection State
        self._last_face = None
        self._detect_interval = 5

        # Models
        self._rvm: Optional[RvmModel] = None
        self._depth: Optional[DepthModel] = None
        
        # New Face Swapper Modules
        self._model_manager = None
        self._inference = None

        # Codec
        self._jpeg = None
        try:
            from turbojpeg import TurboJPEG
            self._jpeg = TurboJPEG()
        except ImportError:
            pass

        self.active_providers: Dict[str, List[str]] = {}

        if not lazy_init:
            self._ensure_models()

    def _ensure_models(self) -> None:
        # Load Face Swapper Models
        if self._model_manager is None:
            # Checkpoints are in /home/ubuntu/codes/ai_face_change/checkpoints
            # We can use absolute path or resolve relative to repo root
            repo_root = self._assets.dfm_dir.parent.parent # blankend/assets/DFM -> blankend/assets -> blankend -> repo
            # Actually default_asset_paths uses: repo_root = Path(__file__).resolve().parents[2]
            # let's just use the absolute path we know works
            checkpoints_dir = "/home/ubuntu/codes/ai_face_change/checkpoints"
            
            try:
                self._model_manager = ModelManager(checkpoints_dir)
                self._model_manager.load_models()
                self._inference = CoreInference(self._model_manager)
                logger.info("Face Swapper models loaded successfully.")
                self.active_providers["swapper"] = ["CUDAExecutionProvider"] # Assuming force CUDA
            except Exception as e:
                logger.error(f"Failed to load Face Swapper models: {e}")

        # Load RVM/Depth only when matting is enabled.
        if self.cfg.matting_enabled and (self._rvm is None or self._depth is None):
            t0 = time.perf_counter()
            rvm_path = resolve_checkpoint_path(os.getenv("RVM_MODEL") or "rvm_mobilenetv3_fp32.onnx")
            depth_path = resolve_checkpoint_path(os.getenv("DEPTH_MODEL") or "depth_anything_v2_vits_fp32.onnx")
            cfg = OrtProvidersConfig(device_id=self.device_id)
            self._rvm = RvmModel(rvm_path, force_cuda=True, providers_cfg=cfg, use_tensorrt=False)
            self._depth = DepthModel(depth_path, force_cuda=True, providers_cfg=cfg, use_tensorrt=False)
            
            self.active_providers["rvm"] = self._rvm.meta.get("providers_active", [])
            self.active_providers["depth"] = self._depth.meta.get("providers_active", [])
            cost_ms = int((time.perf_counter() - t0) * 1000)
            logger.info("models_loaded_rvm_depth device_id=%s cost_ms=%s", int(self.device_id), cost_ms)

    def set_portrait(self, filename: str) -> bool:
        self._ensure_models()
        p = resolve_asset_file(self._assets.portraits_dir, filename)
        if not p:
            logger.warning("portrait_not_found device_id=%s name=%s", int(self.device_id), filename)
            return False
        try:
            img = cv2.imdecode(np.frombuffer(p.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                logger.error("portrait_decode_failed device_id=%s name=%s", int(self.device_id), filename)
                return False
            self._portrait_bgr = img
            
            # Detect and get embedding
            faces = self._inference.detect_faces(img)
            if faces:
                # Pick largest face
                face = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
                self._inference.get_embedding(img, face)
                self._source_face = face
                logger.info("portrait_set_success device_id=%s name=%s faces_found=%d", int(self.device_id), filename, len(faces))
                return True
            
            logger.warning("portrait_no_face device_id=%s name=%s", int(self.device_id), filename)
            return False
        except Exception as e:
            logger.exception("portrait_set_error device_id=%s name=%s err=%s", int(self.device_id), filename, e)
            return False

    def set_scene(self, filename: str) -> bool:
        self._ensure_models()
        p = resolve_asset_file(self._assets.scenes_dir, filename)
        if not p:
            logger.warning("scene_not_found device_id=%s name=%s", int(self.device_id), filename)
            return False
        try:
            bgr = cv2.imdecode(np.frombuffer(p.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                return False
            self._scene_rgb = bgr[:, :, ::-1].copy()
            return True
        except Exception:
            return False

    # DFM removed or kept as placeholder if needed (removing to clean up)

    def _decode(self, jpeg_bytes: bytes) -> Optional[np.ndarray]:
        if self._jpeg is not None:
            try:
                width, height, _, _ = self._jpeg.decode_header(jpeg_bytes)
                scale = (1, 1)
                if width > 3000: scale = (1, 4)
                elif width > 1500: scale = (1, 2)
                rgb = self._jpeg.decode(jpeg_bytes, scaling_factor=scale)
                return rgb[:, :, ::-1].copy()
            except Exception:
                pass
        try:
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                h, w = bgr.shape[:2]
                if w > 1920:
                    new_w = 1280
                    new_h = int(h * (1280 / w))
                    bgr = cv2.resize(bgr, (new_w, new_h))
            return bgr
        except Exception:
            return None

    def _encode(self, bgr: np.ndarray) -> bytes:
        if self._jpeg is not None:
            try:
                rgb = bgr[:, :, ::-1].copy()
                return self._jpeg.encode(rgb, quality=self.cfg.jpeg_quality)
            except Exception:
                pass
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.jpeg_quality)])
        return bytes(buf) if ok else b""

    def _light_fusion(self, source_face: np.ndarray, target_face: np.ndarray) -> np.ndarray:
        try:
            s_lab = cv2.cvtColor(source_face, cv2.COLOR_BGR2LAB).astype(np.float32)
            t_lab = cv2.cvtColor(target_face, cv2.COLOR_BGR2LAB).astype(np.float32)

            s_mean, s_std = cv2.meanStdDev(s_lab)
            t_mean, t_std = cv2.meanStdDev(t_lab)

            s_mean = s_mean.reshape((1, 1, 3))
            s_std = s_std.reshape((1, 1, 3))
            t_mean = t_mean.reshape((1, 1, 3))
            t_std = t_std.reshape((1, 1, 3))
            
            s_std = np.maximum(s_std, 1e-6)

            res_lab = (s_lab - s_mean) / s_std * t_std + t_mean
            res_lab = np.clip(res_lab, 0, 255).astype(np.uint8)
            
            return cv2.cvtColor(res_lab, cv2.COLOR_LAB2BGR)
        except Exception:
            return source_face

    def _swap_face(self, bgr: np.ndarray, seq: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        diag: Dict[str, Any] = {"swap_applied": False}
        
        if self._inference is None or self._source_face is None:
            return bgr, diag
        
        if (seq % self.cfg.swap_stride) != 0:
            return bgr, diag

        # Step 4: Business Logic - Best Match Strategy & Force Swap
        
        # Detect faces
        faces = self._inference.detect_faces(bgr)
        if not faces:
            if seq % 30 == 0:
                logger.info("swap_step_no_face device_id=%s seq=%s", int(self.device_id), seq)
            return bgr, diag

        # Calculate Similarity for all faces
        best_face = None
        max_sim = -1.0
        
        source_emb = self._source_face.normed_embedding
        
        for face in faces:
            # We need embedding for target faces to compare
            # Optim: If we have many faces, getting embedding for all is slow.
            # But "Best Match" requires it.
            # Let's get embedding for all faces.
            try:
                self._inference.get_embedding(bgr, face)
                if face.normed_embedding is not None and source_emb is not None:
                    sim = float(np.dot(face.normed_embedding, source_emb))
                    if sim > max_sim:
                        max_sim = sim
                        best_face = face
            except Exception:
                pass
        
        # If no embedding extraction worked, fallback to largest face
        if best_face is None and faces:
             best_face = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
             max_sim = 0.0 # Unknown
        
        # Debug Output
        if seq % 10 == 0:
            logger.info(f"Swapping face with similarity: {max_sim:.3f}")

        # Force Swap
        # Unless no face is detected (handled above), ALWAYS execute swap
        if best_face:
            try:
                t0 = time.perf_counter()
                swapped_bgr = self._inference.swap_face(self._source_face, best_face, bgr)
                t_cost = (time.perf_counter() - t0) * 1000
                
                # Apply Light Fusion (Optional but recommended)
                if self.cfg.light_fusion_enabled:
                    # Get crops
                    bbox = best_face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    h, w = bgr.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        target_crop = bgr[y1:y2, x1:x2]
                        # We need the swapped crop at the same location.
                        # Since swap_face returns full image, we crop it.
                        swapped_crop = swapped_bgr[y1:y2, x1:x2]
                        corrected_crop = self._light_fusion(swapped_crop, target_crop)
                        swapped_bgr[y1:y2, x1:x2] = corrected_crop

                diag["swap_applied"] = True
                diag["swap_sim"] = max_sim
                if seq % 10 == 0:
                    logger.info("swap_success device_id=%s seq=%s sim=%.3f cost=%.1fms", int(self.device_id), seq, max_sim, t_cost)
                return swapped_bgr, diag
            except Exception as e:
                logger.error("swap_failed device_id=%s seq=%s err=%s", int(self.device_id), seq, e)
                return bgr, diag
        
        return bgr, diag

    def _matting_and_compose(self, client_id: bytes, bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self._rvm is None or self._depth is None:
            return bgr, {"matting": False}
        
        rgb0 = bgr[:, :, ::-1].copy()
        h0, w0 = rgb0.shape[:2]
        
        pad_h = (64 - (h0 % 64)) % 64
        pad_w = (64 - (w0 % 64)) % 64
        if pad_h or pad_w:
            rgb = cv2.copyMakeBorder(rgb0, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            rgb = rgb0
        
        h, w = rgb.shape[:2]
        
        prev = self._rvm_state_by_client.get(client_id)
        st = self._rvm.ensure_state(prev, w, h, downsample_ratio=self.cfg.rvm_downsample_ratio)
        rvm_in = chw_from_rgb_u8(rgb, self._rvm._input_dtype())
        alpha, fgr, st, diag_rvm = self._rvm.run(rvm_in, st)
        self._rvm_state_by_client[client_id] = st
        diag: Dict[str, Any] = {"matting": True, "rvm": diag_rvm}

        alpha_arr = np.array(alpha).squeeze()
        if alpha_arr.ndim == 3: alpha_arr = alpha_arr[0]
        alpha_arr = np.clip(alpha_arr, 0.0, 1.0).astype(np.float32)

        if fgr is not None:
             f = np.array(fgr).squeeze()
             if f.ndim == 3 and f.shape[0] == 3:
                 f = f.transpose(1, 2, 0)
             fgr_rgb = np.clip(f * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
             fgr_rgb = rgb

        if self.cfg.depth_enabled and self._depth is not None:
            size = self.cfg.depth_fusion_size
            rgb_small = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
            depth_in = chw_from_rgb_u8(rgb_small, np.float32)
            depth_out, diag_depth = self._depth.run(depth_in)
            diag["depth"] = diag_depth
            
            d = np.array(depth_out).squeeze()
            dmin, dmax = d.min(), d.max()
            if (dmax - dmin) > 1e-6:
                norm = (d - dmin) / (dmax - dmin)
                mask = (norm > self.cfg.depth_threshold).astype(np.float32)
                ratio = np.mean(mask)
                if 0.02 < ratio < 0.98:
                    m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                    alpha_arr = alpha_arr * m

        bg = self._scene_rgb
        if bg is None:
            bg = np.zeros_like(rgb)
        else:
            if bg.shape[0] != h or bg.shape[1] != w:
                bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
        
        a = alpha_arr.reshape(h, w, 1)
        out = (fgr_rgb.astype(np.float32) * a + bg.astype(np.float32) * (1.0 - a)).astype(np.uint8)
        
        if pad_h or pad_w:
            out = out[:h0, :w0, :]
            
        return out[:, :, ::-1].copy(), diag

    def process_frame(self, client_id: bytes, seq: int, frame_bytes: bytes) -> Tuple[bytes, Dict[str, Any]]:
        # TRACE LOG
        if seq % 30 == 0:
            logger.info(f"process_frame_entry device_id={self.device_id} seq={seq} bytes={len(frame_bytes)}")

        diag: Dict[str, Any] = {"device_id": self.device_id}
        self._ensure_models()
        
        t0 = time.perf_counter()
        bgr = self._decode(frame_bytes)
        t1 = time.perf_counter()
        if bgr is None:
            if seq % 30 == 0:
                logger.error(f"process_frame_decode_fail device_id={self.device_id} seq={seq}")
            return b"", diag

        bgr_swapped, swap_diag = self._swap_face(bgr, seq)
        t2 = time.perf_counter()
        diag.update(swap_diag)

        if self.cfg.matting_enabled:
            bgr_final, mat_diag = self._matting_and_compose(client_id, bgr_swapped)
            t3 = time.perf_counter()
            diag.update(mat_diag)
        else:
            bgr_final = bgr_swapped
            t3 = time.perf_counter()
            diag.update({"matting": False})

        out_bytes = self._encode(bgr_final)
        t4 = time.perf_counter()

        diag["decode_ms"] = (t1 - t0) * 1000.0
        diag["swap_ms"] = (t2 - t1) * 1000.0
        diag["matting_ms"] = (t3 - t2) * 1000.0
        diag["encode_ms"] = (t4 - t3) * 1000.0

        return out_bytes, diag
