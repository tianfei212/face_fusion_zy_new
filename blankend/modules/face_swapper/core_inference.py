import cv2
import numpy as np
import onnxruntime
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from insightface.utils import face_align

@dataclass
class Face:
    bbox: np.ndarray
    kps: np.ndarray
    det_score: float
    embedding: Optional[np.ndarray] = None
    normed_embedding: Optional[np.ndarray] = None

class CoreInference:
    def __init__(self, model_manager):
        self.detector_sess = model_manager.face_detector_session
        self.recognizer_sess = model_manager.face_recognizer_session
        self.swapper_sess = model_manager.face_swapper_session
        self.input_size = (640, 640) # Default for YOLOv8-Face
        
        # ArcFace mean/std
        self.arcface_mean = 127.5
        self.arcface_std = 128.0
        self._swap_target_size = 128
        self._swap_has_mask = False
        self._prev_kps: Optional[np.ndarray] = None
        self._smooth_alpha: float = 0.6
        try:
            for inp in self.swapper_sess.get_inputs():
                s = list(inp.shape or [])
                if len(s) == 4 and int(s[1]) == 3:
                    self._swap_target_size = int(s[2])
            self._swap_has_mask = any(o.name == "mask" for o in self.swapper_sess.get_outputs())
        except Exception:
            pass


    def detect_faces(self, frame: np.ndarray) -> List[Face]:
        # Preprocess
        img_height, img_width = frame.shape[:2]
        input_img = cv2.resize(frame, self.input_size)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.transpose(2, 0, 1).astype(np.float32)
        input_img /= 255.0
        input_img = input_img[np.newaxis, ...]

        # Inference
        input_name = self.detector_sess.get_inputs()[0].name
        outputs = self.detector_sess.run(None, {input_name: input_img})
        
        # Post-process YOLOv8-Face (Simplified)
        # Output shape usually: (1, 4 + 1 + 10, 8400) -> bbox, score, kps
        # This is a best-effort implementation assuming standard YOLOv8-Face output
        # If the model is different, this might need adjustment.
        # However, for now, we focus on the structure.
        
        # Note: Implementing full YOLOv8-Face NMS here is complex.
        # Given the constraints, if this fails, we might fallback or need more info.
        # But let's assume standard output format.
        
        # Placeholder for actual detection logic if complex.
        # For now, let's try to parse it.
        output = outputs[0][0] # (15, 8400)
        # transpose to (8400, 15)
        output = output.transpose()
        
        boxes = output[:, :4]
        scores = output[:, 4]
        kps = output[:, 5:]
        
        # Filter by score
        indices = np.where(scores > 0.5)[0]
        boxes = boxes[indices]
        scores = scores[indices]
        kps = kps[indices]
        
        if len(boxes) == 0:
            return []

        # Convert boxes from cx,cy,w,h to x1,y1,x2,y2 and rescale
        # Input was 640x640. Rescale to img_width, img_height
        scale_x = img_width / self.input_size[0]
        scale_y = img_height / self.input_size[1]
        
        final_faces = []
        # NMS should be applied here (using cv2.dnn.NMSBoxes)
        # ... (Skipping NMS implementation details for brevity, assuming top faces)
        # For strict implementation, we should use NMS.
        
        # Basic NMS
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        w = boxes[:, 2]
        h = boxes[:, 3]
        
        # Rescale
        x1 *= scale_x
        y1 *= scale_y
        w *= scale_x
        h *= scale_y
        
        boxes_xywh = np.stack([x1, y1, w, h], axis=1)
        indices_nms = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), 0.5, 0.4)
        
        for i in indices_nms:
            # Handle different OpenCV versions/return types
            if isinstance(i, (tuple, list, np.ndarray)):
                idx = int(i[0])
            else:
                idx = int(i)
                
            box = boxes_xywh[idx]
            bx, by, bw, bh = box
            bbox = np.array([bx, by, bx+bw, by+bh])
            
            # KPS
            # kps[idx] shape is likely (10,) for 5 landmarks (x,y) flattened
            # or (15,) if it includes confidence?
            # YOLOv8-Face output typically: x,y,w,h,conf, kps...
            # If output was 4+1+10=15, then kps is 10 elements.
            # If output was 4+1+15=20?
            # Let's check size
            raw_kps = kps[idx]
            if raw_kps.size == 10:
                 kp = raw_kps.reshape(-1, 2)
            elif raw_kps.size == 15:
                 # x, y, conf for each of 5 landmarks
                 kp = raw_kps.reshape(-1, 3)[:, :2]
            else:
                 # Fallback, try to reshape to (N, 2) or (N, 3)
                 if raw_kps.size % 2 == 0:
                     kp = raw_kps.reshape(-1, 2)
                 elif raw_kps.size % 3 == 0:
                     kp = raw_kps.reshape(-1, 3)[:, :2]
                 else:
                     # Fallback to zeros if unknown
                     kp = np.zeros((5, 2), dtype=np.float32)

            kp[:, 0] *= scale_x
            kp[:, 1] *= scale_y
            
            final_faces.append(Face(bbox=bbox, kps=kp, det_score=scores[idx]))
            
        return final_faces

    def get_embedding(self, frame: np.ndarray, face: Face) -> np.ndarray:
        # Align face
        # Ensure kps is valid
        if face.kps is None or len(face.kps) != 5:
             # Fallback if no kps, cannot align properly for ArcFace
             # Or detect again on crop?
             # For now, just return None or zero
             return np.zeros(512, dtype=np.float32)

        aligned_face = face_align.norm_crop(frame, landmark=face.kps)
        
        # Preprocess for ArcFace (Strict Clone of FaceFusion/InsightFace)
        # 1. BGR to RGB? No, ArcFace usually trained on BGR or RGB?
        # InsightFace models (arcface_w600k) expect BGR or RGB?
        # InsightFace library uses cv2.imread (BGR) -> input_blob
        # Let's check insightface source: 
        #   blob = cv2.dnn.blobFromImages(imgs, 1.0/127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        #   swapRB=True means BGR -> RGB.
        # So input should be RGB.
        
        input_img = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = np.expand_dims(input_img, axis=0)
        input_img = (input_img.astype(np.float32) - self.arcface_mean) / self.arcface_std
        
        # Inference
        input_name = self.recognizer_sess.get_inputs()[0].name
        embedding = self.recognizer_sess.run(None, {input_name: input_img})[0]

        raw = embedding.flatten().astype(np.float32)
        norm = float(np.linalg.norm(raw))
        if norm > 0:
            normed = (raw / norm).astype(np.float32)
        else:
            normed = raw

        face.embedding = raw
        face.normed_embedding = normed
        return normed

    def swap_face(self, source_face: Face, target_face: Face, frame: np.ndarray) -> np.ndarray:
        # Step 3: Strict Implementation Logic (Clone FaceFusion)
        
        # Embedding Prep
        if source_face.embedding is None:
             raise ValueError("Source face has no embedding")
        
        use_raw = bool(getattr(self, "_swap_has_mask", False) or int(getattr(self, "_swap_target_size", 128) or 128) >= 256)
        if use_raw:
            embedding = source_face.embedding
        else:
            embedding = source_face.normed_embedding if source_face.normed_embedding is not None else source_face.embedding
        if embedding is None:
            raise ValueError("Source face has no embedding")
        embedding = np.asarray(embedding, dtype=np.float32).reshape((1, 512))
        
        crop_size = int(getattr(self, "_swap_target_size", 128) or 128)

        kps_float = np.ascontiguousarray(target_face.kps).astype(np.float32)
        if self._prev_kps is not None and self._prev_kps.shape == kps_float.shape:
            try:
                mean_dist = float(np.linalg.norm(kps_float - self._prev_kps, axis=1).mean())
                if mean_dist < 20.0:
                    a = float(self._smooth_alpha)
                    kps_float = (self._prev_kps * a + kps_float * (1.0 - a)).astype(np.float32)
            except Exception:
                pass
        self._prev_kps = kps_float

        crop, M = face_align.norm_crop2(frame, landmark=kps_float, image_size=crop_size, mode="arcface")
        
        # Tensor Prep (CRITICAL)
        # Convert crop to float32.
        # Divide by 255.0 (Range must be 0.0-1.0).
        # Convert BGR to RGB via [:, :, ::-1].
        # Transpose to (1, 3, crop_size, crop_size) (NCHW).
        
        if getattr(self, "_swap_has_mask", False) or crop_size >= 256:
            test_crop = crop.astype(np.float32) / 127.5 - 1.0
        else:
            test_crop = crop.astype(np.float32) / 255.0
        test_crop = test_crop[:, :, ::-1]
        test_crop = np.transpose(test_crop, (2, 0, 1))
        test_crop = np.expand_dims(test_crop, axis=0)
        
        # Inference
        feeds: Dict[str, np.ndarray] = {}
        for inp in self.swapper_sess.get_inputs():
            shape = list(inp.shape or [])
            if len(shape) == 2 and int(shape[-1]) == 512:
                feeds[inp.name] = embedding.astype(np.float32)
            else:
                feeds[inp.name] = test_crop.astype(np.float32)
        outputs = self.swapper_sess.run(None, feeds)
        output = outputs[0]
        out_mask = None
        if getattr(self, "_swap_has_mask", False) and len(outputs) >= 2:
            out_mask = outputs[1]
        
        # Post-Processing
        # Output is RGB. Convert back to BGR.
        # Multiply by 255.0 and cast to uint8.
        # Paste back using the inverse affine matrix.
        
        out0 = output[0]
        if float(out0.min()) < 0.0:
            out0 = (out0 + 1.0) * 0.5
        res_crop = out0.transpose((1, 2, 0))[:, :, ::-1]
        res_crop = np.clip(res_crop * 255.0, 0, 255).astype(np.uint8)
        
        IM = cv2.invertAffineTransform(M)
        
        if out_mask is not None:
            alpha = np.clip(out_mask[0, 0], 0.0, 1.0).astype(np.float32)
            alpha = cv2.GaussianBlur(alpha, (0, 0), 2.0)

            c = crop_size // 2
            face_soft = np.zeros((crop_size, crop_size), dtype=np.float32)
            ax = int(round(crop_size * 0.44))
            ay = int(round(crop_size * 0.54))
            cv2.ellipse(face_soft, (c, c), (ax, ay), 0, 0, 360, 1.0, -1)
            face_soft = cv2.GaussianBlur(face_soft, (0, 0), 6.0)
            alpha = np.power(alpha, 0.70).astype(np.float32)
            alpha = np.clip(alpha * face_soft, 0.0, 1.0)
            bin_mask = (alpha > 0.35).astype(np.uint8) * 255
        else:
            bin_mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
            c = crop_size // 2
            ax = int(round(crop_size * 0.34))
            ay = int(round(crop_size * 0.40))
            cv2.ellipse(bin_mask, (c, c), (ax, ay), 0, 0, 360, 255, -1)
            feather = float(max(10.0, crop_size * 0.055))
            dist = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 5).astype(np.float32)
            alpha = np.clip(dist / feather, 0.0, 1.0)
            alpha = cv2.GaussianBlur(alpha, (0, 0), 2.0)
        alpha3 = np.dstack([alpha, alpha, alpha])
        img_mask = cv2.warpAffine(alpha3, IM, (frame.shape[1], frame.shape[0]), borderValue=0.0)
        img_mask = np.clip(img_mask, 0.0, 1.0).astype(np.float32)
        img_mask[img_mask < 0.01] = 0.0
        
        # Warped result
        img_white = cv2.warpAffine(
            res_crop,
            IM,
            (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderValue=0.0,
        ).astype(np.float32)
        
        # Masking (simple paste for now as per instructions "Paste back using the inverse affine matrix")
        # To do it properly we usually blend. But instruction says "Paste back".
        # Let's do a simple blend to look better, or strict paste.
        # "Paste back using the inverse affine matrix." -> implies using the mask to overwrite.
        
        # Strict paste:
        # We need a mask of the cropped area in the original image.
        # But `warpAffine` doesn't just paste, it transforms. 
        # So we blend: Result = Original * (1-Mask) + Swapped * Mask
        
        base = frame.astype(np.float32)
        blended = np.clip(base * (1 - img_mask) + img_white * img_mask, 0, 255).astype(np.uint8)

        if out_mask is not None:
            try:
                region = (img_mask[:, :, 0] > 0.60).astype(np.uint8)
                if int(region.max()) > 0:
                    blurred = cv2.GaussianBlur(blended, (0, 0), 1.2)
                    sharp = cv2.addWeighted(blended, 1.6, blurred, -0.6, 0)
                    blended[region > 0] = sharp[region > 0]
            except Exception:
                pass
            return blended

        try:
            k = max(3, int(crop_size // 40))
            m0 = cv2.erode(bin_mask, np.ones((k, k), np.uint8), iterations=1)
            m = cv2.warpAffine(m0, IM, (frame.shape[1], frame.shape[0]), borderValue=0)
            _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
            if int(m.max()) > 0:
                ys, xs = np.where(m > 0)
                if xs.size > 0 and ys.size > 0:
                    cx = int(np.clip(xs.mean(), 0, frame.shape[1] - 1))
                    cy = int(np.clip(ys.mean(), 0, frame.shape[0] - 1))
                    src = np.clip(img_white, 0, 255).astype(np.uint8)
                    dst = frame
                    out = cv2.seamlessClone(src, dst, m, (cx, cy), cv2.NORMAL_CLONE)
                    return out
        except Exception:
            pass

        return blended
