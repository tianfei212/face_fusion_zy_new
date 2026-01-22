# blankend 主备（Active/Standby）无缝切换设计与模块定义

## 目标
- 单用户 VIP 级实时换脸直播：日常单卡工作，另一张卡待命
- 毫秒级切换：人脸图片、背景直接在当前 active 上更新
- 秒级切换：DFM 模型在 standby 后台加载，加载完毕后下一帧切路由，无黑屏无卡顿

## 现有 blankend 目录结构与职责边界
- [main.py](file:///home/ubuntu/codes/ai_face_change/blankend/main.py)：FastAPI 入口与静态资源挂载（/files/*）
- [streaming/routes.py](file:///home/ubuntu/codes/ai_face_change/blankend/streaming/routes.py)：WebSocket /video_in、/video_codec 与 ZMQ relay；负责把进入的帧交给后端处理链路
- realtime_pipeline：旧的“抠像/深度融合”链路（线程模型）
- ai_core：换脸 + 光影融合 + 抠像 + 合成（本次主备改造聚焦）
- video_processing：可选 GPU/ffmpeg 解码与重编码

## 核心架构（Active/Standby）
### Manager（教练，主进程）
- 持有 `active_worker_id`（0/1）
- 拉起两个 OS 子进程 Worker（固定 GPU）
- 接收帧后只把帧投递给 active Worker
- 接收 Worker 输出并通过 relay 发回上游（或广播到前端 WS）

### Worker（主力/备胎，子进程）
- 固定绑定 `device_id`，只在该 GPU 上加载 ONNX/InsightFace/RVM/Depth
- 内部持有 `UnifiedInferenceEngine(device_id)`
- 命令队列：处理 SET_FACE/SET_BG/LOAD_DFM/UPDATE_CONFIG/ACTIVATE
- 帧队列：处理当前 active 分配的 JPEG 帧

## 数据流
1. ZmqRelay 收到 (client_id, jpeg_bytes)
2. streaming 路由调用 `AiActiveStandbyManager.submit_frame(client_id, jpeg)`
3. Manager 将帧投递到 active worker 的 frame_q
4. Worker 执行推理并把结果放入 out_q
5. Manager out_loop 取出结果并调用 `relay.enqueue(client_id, jpeg)` 发回

## 推理引擎定义（UnifiedInferenceEngine）
实现位于 [engine.py](file:///home/ubuntu/codes/ai_face_change/blankend/ai_core/engine.py)
- 输入：JPEG bytes
- 输出：JPEG bytes + diag
- Pipeline（顺序固定）：Decode -> Swap -> LightFusion -> Matting -> Composite -> Encode
- LightFusion：在换脸后、抠像前执行，可通过 `light_fusion_enabled` 开关控制
- DFM：支持 LOAD_DFM 加载模型并切换（推理 I/O 依赖具体 DFM 模型定义）

## 通信协议（Manager <-> Worker）
实现位于 [protocol.py](file:///home/ubuntu/codes/ai_face_change/blankend/ai_core/protocol.py)
- SET_FACE(payload=filename)
- SET_BG(payload=filename)
- LOAD_DFM(payload=filename, request_id=uuid)
- ACTIVATE(payload={"active": bool})
- UPDATE_CONFIG(payload=dict)
- READY（worker->manager，表示 standby 已加载完成）

## 资产与模型目录约定
### 静态资产（前端可直接访问）
- `blankend/assets/human_pic` -> `/files/human_pic/*`
- `blankend/assets/m_blank` -> `/files/m_blank/*`

### 模型（checkpoints）
路径解析函数：`resolve_checkpoint_path()`（见 [onnx_runtime.py](file:///home/ubuntu/codes/ai_face_change/blankend/realtime_pipeline/onnx_runtime.py)）
- 默认根目录：`repo_root/checkpoints`，或由 `CHECKPOINT_ROOT` 覆盖
- 推荐目录：
  - `checkpoints/rvm_mobilenetv3_fp32.onnx`
  - `checkpoints/depth_anything_v2_vits_fp32.onnx`
  - `checkpoints/inswapper_128_fp16.onnx`
  - `checkpoints/DFM/*.onnx`

DFM listing 默认扫描 `checkpoints/DFM`，若不存在则回退 `blankend/assets/DFM`（见 [asset_manager.py](file:///home/ubuntu/codes/ai_face_change/blankend/ai_core/asset_manager.py)）

## 后端 API（FastAPI）
实现位于 [ai_core/routes.py](file:///home/ubuntu/codes/ai_face_change/blankend/ai_core/routes.py)
- `GET /api/v1/ai/status`：Manager 状态、active_worker_id、各 worker 最近 READY/ERROR、GPU 信息
- `POST /api/v1/ai/process`：兼容旧接口（启停、设置 portrait/scene、相似度）
- `GET /api/v1/ai/assets/list?type=face|bg|dfm`：返回文件名列表
- `POST /api/v1/ai/command`：统一指令入口（SET_FACE/SET_BG/LOAD_DFM/ACTIVATE/UPDATE_CONFIG）

## 无缝切换时序（推荐）
1. 前端选择 DFM：调用 `POST /api/v1/ai/command {type: LOAD_DFM, payload: \"xxx.onnx\"}`
2. 前端轮询 `GET /api/v1/ai/status` 直到 standby worker 出现 `last_ready` 且 loaded_dfm=xxx.onnx
3. 前端调用 `POST /api/v1/ai/command {type: ACTIVATE}`（或带 worker_id）
4. 下一帧开始路由到新 active，无中断

## 本地验证（无模型旁路）
可设置 `SIMPLE_FORWARD=1` 或 `AI_WORKER_SIMPLE_FORWARD=1`，worker 将直接回传输入帧 bytes，用于验证主备进程/路由/READY/ACTIVATE 的控制面不依赖模型可先跑通。

