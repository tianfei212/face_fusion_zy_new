import os
import onnxruntime
from pathlib import Path

class ModelManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.face_detector_session = None
        self.face_recognizer_session = None
        self.face_swapper_session = None

    def load_models(self):
        # Paths to models
        detector_path = self.checkpoint_dir / "yoloface_8n.onnx"
        recognizer_path = self.checkpoint_dir / "arcface_w600k_r50.onnx"
        swapper_path_256 = self.checkpoint_dir / "hyperswap_1a_256.onnx"
        swapper_path_128 = self.checkpoint_dir / "inswapper_128.onnx"

        # Load Face Detector
        if detector_path.exists():
            self.face_detector_session = onnxruntime.InferenceSession(
                str(detector_path), providers=self.providers
            )
        else:
            raise FileNotFoundError(f"Face detector model not found at {detector_path}")

        # Load Face Recognizer (Embedding)
        if recognizer_path.exists():
            self.face_recognizer_session = onnxruntime.InferenceSession(
                str(recognizer_path), providers=self.providers
            )
        else:
            raise FileNotFoundError(f"Face recognizer model not found at {recognizer_path}")

        # Load Face Swapper (prefer 256 if available)
        if swapper_path_256.exists():
            self.face_swapper_session = onnxruntime.InferenceSession(
                str(swapper_path_256), providers=self.providers
            )
        elif swapper_path_128.exists():
            self.face_swapper_session = onnxruntime.InferenceSession(
                str(swapper_path_128), providers=self.providers
            )
        else:
            raise FileNotFoundError("Face swapper model not found (256/128)")

        return self.face_detector_session, self.face_recognizer_session, self.face_swapper_session
