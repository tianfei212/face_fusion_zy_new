# 部署指南（后端）

## 启动
- 参考根目录脚本：`start_backend.sh`
- 主要服务：
  - 端口 8001：基础/资源类接口
  - 端口 8100：视频流 WebSocket（`/video_in`）

## 新增 API 端点
- Video Processing（状态/监控/配置）：`/api/v1/video_processing/*`
- Skills（注册/调用）：`/api/v1/skills`
- 新增视频硬解码入口（保持旧协议不变）：WebSocket `/video_codec?codec=h264&prefer_gpu=1`

## 配置项
- 后端配置文件：`blankend/config.json`
- 可选环境变量：
  - `BLANKEND_VIDEO_PROCESSING_ENABLED=1`：启用生产者-消费者流水线（默认关闭，保持现有行为）
  - `BLANKEND_HW_TRANSCODE=1`：启用 TurboJPEG 硬件相关重编码（现有逻辑）
  - `FFMPEG=/path/to/ffmpeg`：指定 ffmpeg 可执行文件路径（用于硬解码）
