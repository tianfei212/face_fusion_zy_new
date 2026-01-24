```markdown
# 任务：基于 MatAnyone 模型重构推理代码

## 📋 任务概述

**目标：** 生成两个完整的 Python 文件

1. **`rvm_inference.py`** - 使用 MatAnyone ONNX 模型的抠图推理类（全新实现）
2. **`core_inference.py`** - 人脸交换推理类（完全保持不变，直接复制）

---

## 🎯 MatAnyone 模型规格

### 模型文件
```python
model_path = "matanyone_fixed.onnx"
```

### ONNX 输入

| 参数名 | 形状 | 数据类型 | 值范围 | 说明 |
|--------|------|----------|--------|------|
| `image` | `[batch, 3, 512, 512]` | float32 | [0.0, 1.0] | RGB 图像，已归一化 |
| `ref_sensory` | `[batch, 1, 256, 32, 32]` | float32 | 任意 | 参考感知特征图 |
| `ref_mask` | `[batch, 1, 512, 512]` | float32 | [0.0, 1.0] | 参考遮罩图 |

### ONNX 输出

| 参数名 | 形状 | 数据类型 | 说明 |
|--------|------|----------|------|
| `alpha` | `[batch, 2, 512, 512]` | float32 | 通道 0 = 背景概率<br>通道 1 = 前景概率 |

### 关键特性

1. **固定分辨率：** 模型内部强制 512×512，无法动态调整
2. **无 Recurrent 状态：** 不需要 r1/r2/r3/r4 这类循环状态
3. **参考帧机制：**
   - **首帧初始化：** `ref_sensory` 和 `ref_mask` 使用零初始化
   - **后续帧传递：** 可选择是否更新参考帧（建议保持首帧不变以获得稳定效果）

---

## 📄 文件 1：`rvm_inference.py`

### 类设计要求

```python
class RVMInference:
    """
    MatAnyone ONNX 模型推理类
    用于视频/图像前景分割（抠图）
    """
```

### 初始化方法 `__init__`

#### 参数
```python
def __init__(self, model_path: str, target_size: tuple = (512, 512)):
    """
    :param model_path: ONNX 模型路径
    :param target_size: 推理分辨率 (width, height)
                       注意：MatAnyone 固定为 (512, 512)，此参数仅为兼容性保留
    """
```

#### 需要初始化的属性

```python
self.session           # ONNX Runtime 会话对象
self.infer_w          # 推理宽度：512
self.infer_h          # 推理高度：512
self.ref_sensory      # 参考感知特征（首帧后保存）
self.ref_mask         # 参考遮罩（首帧后保存）
self.is_first_frame   # 是否为首帧标志
```

#### 实现逻辑

1. **验证 target_size**
   - 如果传入的不是 (512, 512)，打印警告：`"!!! MatAnyone only supports 512x512, ignoring target_size !!!"`
   - 强制设置 `self.infer_w = 512`, `self.infer_h = 512`

2. **加载模型**
   - 检查文件是否存在
   - 使用 `onnxruntime.InferenceSession`
   - 优先使用 `CUDAExecutionProvider`，回退到 `CPUExecutionProvider`
   - 打印：`"--- MatAnyone: Loading Model from {model_path} ---"`

3. **初始化参考帧**
   ```python
   self.ref_sensory = None
   self.ref_mask = None
   self.is_first_frame = True
   ```

4. **错误处理**
   - 文件不存在：打印 `"!!! MatAnyone MODEL FILE MISSING: {model_path} !!!"`
   - 加载失败：打印 `"!!! MatAnyone CRASH DURING INIT: {e} !!!"`，设置 `self.session = None`

5. **成功提示**
   - 打印：`"--- MatAnyone: Engine Started! (Resolution: 512x512) ---"`

---

### 推理方法 `process`

#### 方法签名
```python
def process(self, img_bgr: np.ndarray, green_bg: bool = True) -> np.ndarray:
    """
    处理单帧图像
    
    :param img_bgr: 输入图像 (BGR 格式)
    :param green_bg: 是否使用绿幕背景合成
    :return: 处理后的图像 (BGR 格式)
    """
```

#### 实现逻辑

##### 1. 安全检查
```python
if self.session is None:
    return img_bgr
```

##### 2. 保存原始尺寸
```python
orig_h, orig_w = img_bgr.shape[:2]
```

##### 3. 图像预处理
```python
# Resize 到 512×512
img_small = cv2.resize(img_bgr, (512, 512))

# BGR → RGB
img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

# 归一化到 [0, 1]
img_tensor = img_rgb.astype(np.float32) / 255.0

# 转换为 NCHW 格式：(H, W, C) → (1, C, H, W)
img_tensor = np.transpose(img_tensor, (2, 0, 1))[np.newaxis, ...]
```

##### 4. 首帧处理
```python
if self.is_first_frame:
    # 初始化参考特征（零值）
    self.ref_sensory = np.zeros((1, 1, 256, 32, 32), dtype=np.float32)
    self.ref_mask = np.zeros((1, 1, 512, 512), dtype=np.float32)
    self.is_first_frame = False
```

##### 5. 构造输入字典
```python
inputs = {
    'image': img_tensor,
    'ref_sensory': self.ref_sensory,
    'ref_mask': self.ref_mask
}
```

##### 6. ONNX 推理
```python
try:
    results = self.session.run(None, inputs)
    alpha_output = results[0]  # shape: (1, 2, 512, 512)
    
except Exception as e:
    # 静默失败，避免刷屏
    # print(f"!!! MatAnyone RUNTIME ERROR: {e} !!!")
    return img_bgr
```

##### 7. 提取前景概率
```python
# 通道 1 = 前景概率
alpha_small = alpha_output[0, 1]  # shape: (512, 512)
```

##### 8. 可选：更新参考遮罩（用于下一帧）
```python
# 方案 A：保持首帧参考不变（推荐，效果更稳定）
# 不更新 self.ref_mask

# 方案 B：使用当前帧作为参考（可能导致漂移）
# self.ref_mask = alpha_output[:, 1:2, :, :]  # shape: (1, 1, 512, 512)
```

##### 9. Resize 回原始尺寸
```python
alpha = cv2.resize(alpha_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
```

##### 10. 后处理合成
```python
if green_bg:
    # 扩展到 3 通道
    alpha_3c = np.stack([alpha] * 3, axis=-1)
    
    # 创建绿色背景
    bg = np.zeros_like(img_bgr)
    bg[:, :] = (0, 255, 0)  # BGR 格式的绿色
    
    # Alpha 混合
    foreground = img_bgr.astype(np.float32)
    background = bg.astype(np.float32)
    comp = foreground * alpha_3c + background * (1.0 - alpha_3c)
    
    return np.clip(comp, 0, 255).astype(np.uint8)
else:
    return img_bgr
```

---

### 附加方法（可选）

#### 重置方法
```python
def reset(self):
    """
    重置参考帧状态（用于处理新视频/场景切换）
    """
    self.ref_sensory = None
    self.ref_mask = None
    self.is_first_frame = True
```

#### 获取原始 Alpha 遮罩方法
```python
def get_alpha(self, img_bgr: np.ndarray) -> np.ndarray:
    """
    仅返回 Alpha 遮罩，不进行合成
    
    :param img_bgr: 输入图像 (BGR 格式)
    :return: Alpha 遮罩 (单通道 float32, 范围 [0, 1])
    """
    if self.session is None:
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)
    
    orig_h, orig_w = img_bgr.shape[:2]
    img_small = cv2.resize(img_bgr, (512, 512))
    img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    img_tensor = img_rgb.astype(np.float32) / 255.0
    img_tensor = np.transpose(img_tensor, (2, 0, 1))[np.newaxis, ...]
    
    if self.is_first_frame:
        self.ref_sensory = np.zeros((1, 1, 256, 32, 32), dtype=np.float32)
        self.ref_mask = np.zeros((1, 1, 512, 512), dtype=np.float32)
        self.is_first_frame = False
    
    inputs = {
        'image': img_tensor,
        'ref_sensory': self.ref_sensory,
        'ref_mask': self.ref_mask
    }
    
    try:
        results = self.session.run(None, inputs)
        alpha_output = results[0]
        alpha_small = alpha_output[0, 1]
        alpha = cv2.resize(alpha_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return alpha
    except Exception:
        return np.zeros((orig_h, orig_w), dtype=np.float32)
```

---

### 代码风格要求

1. **导入语句**
```python
import os
import cv2
import numpy as np
import onnxruntime
```

2. **类型注解**
   - 使用 `np.ndarray` 表示 NumPy 数组
   - 使用 `tuple` 表示元组
   - 使用 `str`, `bool`, `float` 等基本类型

3. **打印格式**
   - 成功信息：`--- MatAnyone: xxx ---`
   - 错误信息：`!!! MatAnyone xxx !!!`
   - 警告信息：`!!! MatAnyone only supports 512x512, ignoring target_size !!!`

4. **注释风格**
   - 步骤注释：`# 1. 验证参数`, `# 2. 加载模型`
   - 行内注释：`alpha_small = alpha_output[0, 1]  # 前景概率`
   - Docstring：Google 风格

5. **异常处理**
   - 初始化阶段：打印错误并设置 `self.session = None`
   - 推理阶段：静默失败返回原图（避免刷屏）

6. **变量命名**
   - 遵循 snake_case
   - 使用描述性名称：`orig_h`, `img_small`, `alpha_output`

---

## 📄 文件 2：`core_inference.py`

### 要求

**完全保持原代码不变**，直接复制原始的 `core_inference.py` 文件内容。

包含：
- 所有 import 语句
- `@dataclass Face` 定义
- `CoreInference` 类完整实现
- 所有方法：
  - `__init__`
  - `detect_faces`
  - `get_embedding`
  - `_lab_color_transfer`
  - `_get_landmark_mask`
  - `_balance_embedding`
  - `_create_soft_mask`
  - `swap_face`
- 所有注释和格式

**不做任何修改！**

---

## ✅ 输出要求

### 格式

请按以下格式输出：

````markdown
# 生成的完整代码

## 文件 1：`rvm_inference.py`

```python
# 在这里输出完整的 rvm_inference.py 代码
import os
import cv2
import numpy as np
import onnxruntime

class RVMInference:
    ...
```

---

## 文件 2：`core_inference.py`

```python
# 在这里输出完整的 core_inference.py 代码（原样复制）
import cv2
import numpy as np
...
```
````

### 检查清单

在输出代码前，确认以下内容：

**`rvm_inference.py`**
- [ ] 删除了所有 r1/r2/r3/r4 相关代码
- [ ] 删除了 downsample_ratio 相关代码
- [ ] 添加了 ref_sensory 和 ref_mask 管理
- [ ] 固定推理分辨率为 512×512
- [ ] 输出使用 `alpha_output[0, 1]`（前景通道）
- [ ] 包含 `reset()` 和 `get_alpha()` 方法
- [ ] 错误处理完整
- [ ] 打印格式统一
- [ ] 代码可以直接运行

**`core_inference.py`**
- [ ] 完全保持原样
- [ ] 没有任何修改
- [ ] 格式和缩进正确

---

## 🚀 使用示例

生成的代码应该可以这样使用：

```python
from rvm_inference import RVMInference
from core_inference import CoreInference, Face

# 初始化 MatAnyone 抠图模型
matting = RVMInference("matanyone_fixed.onnx")

# 处理视频帧
import cv2
cap = cv2.VideoCapture("input.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 抠图
    result = matting.process(frame, green_bg=True)
    
    cv2.imshow("MatAnyone", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 场景切换时重置
matting.reset()
```

---

## 📌 特别注意

1. **不要生成测试代码或 main 函数**，只生成类定义
2. **不要添加额外的依赖库**，只使用 `os`, `cv2`, `numpy`, `onnxruntime`
3. **保持代码简洁专业**，注释精炼准确
4. **确保代码可以直接复制粘贴使用**
5. **MatAnyone 模型路径硬编码为 `"matanyone_fixed.onnx"`**（可在初始化时传入）

---

## 🎯 开始生成

请现在生成两个完整的 Python 文件。

确保：
1. ✅ `rvm_inference.py` 完全基于 MatAnyone 规格实现
2. ✅ `core_inference.py` 原样保持不变
3. ✅ 代码格式专业规范
4. ✅ 可以直接投入生产使用
```

---

## 使用说明

将上述 Markdown 内容：
1. 复制到文本文件（如 `prompt.md`）
2. 直接发送给 GPT-4o/GPT-4/Claude
3. 或者直接在对话框中粘贴

提示词已包含：
- ✅ 完整的技术规格
- ✅ 详细的实现逻辑
- ✅ 代码风格要求
- ✅ 边界情况处理
- ✅ 输出格式要求
- ✅ 使用示例

 
