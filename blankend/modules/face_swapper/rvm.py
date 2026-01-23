import os
import cv2
import numpy as np
import onnxruntime

class RVMInference:
    def __init__(self, model_path: str, target_size: tuple = (640, 384)):
        """
        :param model_path: 模型路径
        :param target_size: 推理分辨率 (width, height)，必须能被32整除
        """
        self.session = None
        self.has_downsample_input = False
        
        # 接收外部传入的分辨率
        self.infer_w, self.infer_h = target_size
        
        # 预计算状态尺寸 (MobileNet strides: 2, 4, 8, 16)
        # 动态计算，不再写死
        s1_h, s1_w = self.infer_h // 2,  self.infer_w // 2
        s2_h, s2_w = self.infer_h // 4,  self.infer_w // 4
        s3_h, s3_w = self.infer_h // 8,  self.infer_w // 8
        s4_h, s4_w = self.infer_h // 16, self.infer_w // 16
        
        # 生成固定尺寸的零状态
        self.zero_r1 = np.zeros((1, 16, s1_h, s1_w), dtype=np.float32)
        self.zero_r2 = np.zeros((1, 20, s2_h, s2_w), dtype=np.float32)
        self.zero_r3 = np.zeros((1, 40, s3_h, s3_w), dtype=np.float32)
        self.zero_r4 = np.zeros((1, 64, s4_h, s4_w), dtype=np.float32)
        
        print(f"--- RVM: Loading Model from {model_path} ---")
        
        if os.path.exists(model_path):
            try:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.session = onnxruntime.InferenceSession(model_path, providers=providers)
                input_names = [inp.name for inp in self.session.get_inputs()]
                self.has_downsample_input = ('downsample_ratio' in input_names)
                print(f"--- RVM: Engine Started! (Resolution: {self.infer_w}x{self.infer_h}) ---")
            except Exception as e:
                print(f"!!! RVM CRASH DURING INIT: {e} !!!")
                self.session = None
        else:
            print(f"!!! RVM MODEL FILE MISSING: {model_path} !!!")

    def process(self, img_bgr: np.ndarray, green_bg: bool = True) -> np.ndarray:
        if self.session is None:
            return img_bgr

        # 1. 保存原始尺寸
        orig_h, orig_w = img_bgr.shape[:2]

        # 2. 手动缩放到配置的尺寸
        img_small = cv2.resize(img_bgr, (self.infer_w, self.infer_h))

        # 3. 预处理
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        img_tensor = img_rgb.astype(np.float32) / 255.0
        img_tensor = np.transpose(img_tensor, (2, 0, 1))[np.newaxis, ...]

        # 4. 构造输入 (downsample_ratio = 1.0, 因为我们已经手动resize了)
        inputs = {
            'src': img_tensor,
            'r1i': self.zero_r1,
            'r2i': self.zero_r2,
            'r3i': self.zero_r3,
            'r4i': self.zero_r4,
        }
        if self.has_downsample_input:
            inputs['downsample_ratio'] = np.array([1.0], dtype=np.float32)

        # 5. 推理
        try:
            results = self.session.run(None, inputs)
            fgr, pha = results[0], results[1]
            
            # 6. 后处理
            alpha_small = pha[0][0] 
            
            # 放大回原图尺寸
            alpha = cv2.resize(alpha_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            alpha_3c = np.stack([alpha]*3, axis=-1)

            if green_bg:
                bg = np.zeros_like(img_bgr)
                bg[:, :] = (0, 255, 0)
                
                foreground = img_bgr.astype(np.float32)
                background = bg.astype(np.float32)
                
                comp = foreground * alpha_3c + background * (1.0 - alpha_3c)
                return np.clip(comp, 0, 255).astype(np.uint8)
            else:
                return img_bgr

        except Exception as e:
            # 简化报错日志，防止刷屏
            # print(f"!!! RVM RUNTIME ERROR: {e} !!!")
            return img_bgr
