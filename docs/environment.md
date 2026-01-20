# 环境配置说明

## Conda 环境
- 期望环境名：`ai_face_change`
- 推荐在该环境内运行前端与后端，避免依赖版本漂移。

## 依赖检测与自动安装
- 运行环境验证脚本：
  - `python scripts/verify_env.py`
- 自动安装缺失依赖（会执行 pip install）：
  - `python scripts/verify_env.py --install`

重点依赖：
- `onnxruntime-gpu>=1.12.0`（GPU 执行提供者与加速能力）
- `pynvml`（GPU 监控；如缺失会自动回退到 `nvidia-smi` 或返回空指标）

视频硬解码（可选）：
- 需要系统安装 `ffmpeg`，并编译/启用 `cuda` hwaccel 与 `h264_cuvid/hevc_cuvid`（解码）能力。
- 可通过 `GET /api/v1/video_processing/codec/status` 检查 `cuda_hwaccel/h264_cuvid/h264_nvenc` 等能力是否可用。
