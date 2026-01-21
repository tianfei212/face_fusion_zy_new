# 视频流处理现状（后端：blankend）

本文基于当前仓库代码静态检查整理，目标是回答：

- 现在视频流怎么处理（输入/输出/格式/路由）
- 处理流程是什么（数据流转路径）
- 启动了多少线程（显式创建的 Python 线程 + 每连接额外线程）

后端主目录：`blankend/`，FastAPI 应用入口：`blankend/main.py`。

---

## 1. 视频流入口与协议

### 1.1 WebSocket：`/video_in`（直接 JPEG 帧）

- 路由：`blankend/streaming/routes.py` 的 `@router.websocket("/video_in")`
- 输入：WebSocket binary message（`bytes`），内容为 **JPEG 单帧**（通常是摄像头抓图编码后发送）
- 输出：
  - 通过 `StreamHub.broadcast_bytes()` 广播给所有连接到本后端的 WebSocket 客户端
  - 以及（可选）转发到 ZMQ relay（见 2.2）

### 1.2 WebSocket：`/video_codec`（编码流 -> 后端解码成 JPEG 帧）

- 路由：`blankend/streaming/routes.py` 的 `@router.websocket("/video_codec")`
- 输入：WebSocket binary message（`bytes`），内容为 H264/H265 等编码流（由 query 参数 `codec` 指定）
- 后端动作：
  - 启动 ffmpeg 子进程，将输入编码流解码并输出 MJPEG（image2pipe）
  - 从 stdout 中切分出 JPEG 帧（`FFD8...FFD9`），然后广播/转发
- 输出：同 `/video_in`

### 1.3 ZMQ Relay（DEALER<->ROUTER，多路分发）

实现：`blankend/streaming/routes.py` 中 `class ZmqRelay`，默认连接 `blankend/config.json` 的：

- `stream.zmq_endpoint`（默认 `tcp://localhost:8888`）

消息格式（站在 ROUTER 侧看）：

- blankend(DEALER) 发送：`[routing_id, client_id, data]`
- ROUTER 向 blankend 发回：`[routing_id, client_id, data]`
  - 其中 `routing_id` 是 blankend 的 `identity`（`BLANKEND_ZMQ_IDENTITY` / `stream.zmq_identity`）
  - `client_id` 用于在“总线/路由器”侧区分来源与目标（如 `SRC`、`WEB_FRONTEND`）

仓库里有自测脚本验证该协议：`blankend/scripts/zmq_relay_selftest.py`。

---

## 2. 处理流程（数据流转路径）

### 2.1 主线程/事件循环：StreamHub

- `StreamHub` 位于 `blankend/streaming/hub.py`
- 负责维护已连接 WebSocket 客户端集合，并对所有客户端广播：
  - `broadcast_bytes(data: bytes)`
  - `broadcast_text(text: str)`

### 2.2 `/video_in` 的流程（收到 JPEG 帧后）

位置：`blankend/streaming/routes.py` 的 `stream(ws)`

当收到 WebSocket 的 `bytes`：

1. **（可选）送入 ZMQ**：如果 relay 启用，`relay.enqueue(default_client_id, data)`
2. 分支 A：**VideoProcessingService 开启**
   1. `vp_service.submit(time.time(), data)` 将帧丢入本地处理队列
   2. 如果队列满导致 `submit` 返回 False，则直接 `hub.broadcast_bytes(data)`（旁路降级）
   3. VideoProcessingService 的 encode worker 会把帧回调到 `hub.broadcast_bytes`
3. 分支 B：**VideoProcessingService 关闭**
   1. （可选）TurboJPEG 重编码（需 `stream.reencode_enabled=true` 且环境变量 `BLANKEND_HW_TRANSCODE=1` 且安装 TurboJPEG）
   2. `hub.broadcast_bytes(data)`

备注：这里的 “VideoProcessingService” 本质上是一个“异步队列 + worker 线程”的广播管线，目前只做（可选）重编码/节流，不做 AI 推理。

### 2.3 `/video_codec` 的流程（编码流 -> JPEG 帧）

位置：`blankend/streaming/routes.py` 的 `stream_codec(ws)`

1. 启动 `FfmpegHwDecoder`（`blankend/video_processing/hwdecode.py`）
   - prefer_gpu=1 时优先 `-hwaccel cuda -c:v h264_cuvid/hevc_cuvid`，失败回退 CPU
2. 启动后台线程 `run_decode_session()` 从 decoder 输出队列不断取 JPEG 帧
3. 对每个 JPEG 帧：
   - （可选）送入 ZMQ：`relay.enqueue(default_client_id, frame)`
   - 广播到 WebSocket：`hub.broadcast_bytes(frame)`（通过 `asyncio.run_coroutine_threadsafe` 回主 loop）

### 2.4 ZMQ 收到帧后的流程（总线 -> blankend）

位置：`blankend/streaming/routes.py` 的 `ZmqRelay._recv_loop()`

收到 `[client_id, data]`（控制通道除外）后，先走“处理优先级”：

1. **优先投喂 AI / 实时管线（frame_handler）**
   - 绑定逻辑在 `get_router()`：`ai_service.enabled() ? ai_service.submit(...) : rt_service.submit(...)`
   - 只要 submit 返回 True，就认为“已接管”，`_recv_loop` 会 `continue`，不再做回转与广播
2. **如果没有被 AI/实时管线接管（submit 返回 False 或服务未启用）**
   - `self.enqueue(client_id, data)`：把帧回发给总线（目标为原 client_id）
   - 若 `forward_recv_enabled`，还会额外 `enqueue(forward_recv_client_id, data)`（例如 `WEB_FRONTEND`）
   - 同时广播给本地 WebSocket：`hub.broadcast_bytes(data)`

因此从整体上看，ZMQ 接入点既可承担“把帧广播给前端”，也可以承担“把帧交给本机 AI/实时处理并将结果再发回总线”的角色。

### 2.5 AI / 实时管线处理后的回传路径

#### RealtimePipelineService（实时抠图/融合/录制）

- 服务：`blankend/realtime_pipeline/service.py`
- 处理线程 `_worker_loop()` 产出 `out_jpg` 后：
  - `sender(client_id, out)`（在 `get_router()` 中 attach 为 `relay.enqueue`）
  - （可选）`broadcaster(out)`（attach 为 `hub.broadcast_bytes` 的线程安全封装）

#### AiCoreService（人脸检测/替换/抠图/合成）

- 服务：`blankend/ai_core/service.py`
- 数据流：
  - `submit()` -> `_in_q` -> `_dispatch_loop`（分发到设备 worker）-> `_DeviceWorker` -> `_output_q` -> `_send_loop`
  - `_send_loop` 最终调用 attach 的 `sender(client_id, jpeg)` / `broadcaster(jpeg)`

总结：AI/实时管线的“输出帧”通常会通过 `relay.enqueue(client_id, out)` 回到 ZMQ 总线，再由总线转发给 `WEB_FRONTEND` 或其它 client_id（同时也可能本地 broadcast）。

---

## 3. 线程/进程模型（关键问题：启动了多少线程）

这里统计的是 **代码里显式创建的 `threading.Thread`**。不包含：

- uvicorn 自己的线程池/内部线程
- numpy/onnxruntime/insightface/ffmpeg 等库在 C/CUDA 层可能启动的线程

### 3.1 进程内启动后的“常驻线程”（默认配置）

假设使用 `uvicorn blankend.main:app` 的单进程启动方式（仓库脚本 `blankend/scripts/run_simple.sh`），并且 `stream.relay_enabled=true` 且安装 pyzmq。

1. **ZmqRelay（常驻）**
   - `_send_loop`：1
   - `_recv_loop`：1
   - `heartbeat_enabled=true` 时 `_heartbeat_loop`：1
   - `link_request_enabled=true` 时 `_link_request_loop`：1
   - 合计：**2～4 个线程**（默认 `config.json` 是 4）

2. **VideoProcessingService（常驻）**
   - `vp-monitor`：1
   - `decode_workers`：N（默认 1）
   - `encode_workers`：M（默认 1）
   - 合计：**1 + N + M**（默认 3）
   - 说明：该服务在 router startup 时 `start()`，即使 `enabled=false` 也会起 worker 线程，只是 `submit()` 会返回 False（旁路）。

3. **RealtimePipelineService（常驻）**
   - worker 数：`BLANKEND_REALTIME_PIPELINE_WORKERS`（默认 1）
   - 合计：**W**（默认 1）
   - 说明：服务在 `get_realtime_pipeline_service()` 时立即 `start()`，即使 `enabled=false` 也会有线程常驻（线程内部 sleep 轮询）。

4. **AiCoreService（常驻）**
   - `_DeviceWorker`：固定 2 个（`ai-core-gpu0`、`ai-core-gpu1`）：2
   - `ai-core-dispatch`：1
   - `ai-core-send`：1
   - 合计：**4 个线程**
   - 说明：服务在 `get_ai_core_service()` 时立即 `start()`，即使 `enabled=false` 也会常驻线程（内部队列轮询）。

**按默认值粗略合计（不含 uvicorn 主线程）：**

- ZmqRelay 4 + VideoProcessing 3 + Realtime 1 + AiCore 4 = **12 个线程**

### 3.2 每个 `/video_codec` 连接新增的线程/进程

每个 WebSocket `/video_codec` 连接会额外创建：

- `run_decode_session`：1 个线程
- `FfmpegHwDecoder` 的 stdout/stderr 读取：2 个线程
- ffmpeg 子进程：1 个进程

合计：**每连接 +3 线程 +1 子进程**

`/video_in` 连接不会额外创建线程（仅 asyncio 协程在事件循环中运行）。

---

## 4. 相关配置开关（影响数据路径）

### 4.1 `blankend/config.json`（stream 段）

文件：`blankend/config.json`

关键项：

- `relay_enabled`：是否启用 ZMQ relay
- `zmq_endpoint`：ROUTER 端点地址（默认 `tcp://localhost:8888`）
- `send_queue_size`：发送队列大小（ZMQ）
- `forward_recv_enabled` / `forward_recv_client_id` / `forward_recv_only`：收到 ZMQ 帧后的转发策略
- `heartbeat_enabled` / `heartbeat_interval_ms`
- `link_request_enabled` / `link_request_*`
- `reencode_enabled` / `jpeg_quality`：WebSocket 侧重编码策略（需 TurboJPEG）

### 4.2 主要环境变量

- ZMQ：
  - `BLANKEND_RELAY_ENABLED`：覆盖 `relay_enabled`
  - `BLANKEND_ZMQ_ENDPOINT`：覆盖 `zmq_endpoint`
  - `BLANKEND_ZMQ_IDENTITY`：覆盖 `zmq_identity`
- TurboJPEG 重编码：
  - `BLANKEND_HW_TRANSCODE=1`：启用 `turbojpeg` 初始化（用于 `/video_in` 的可选重编码）
- VideoProcessingService：
  - `BLANKEND_VIDEO_PROCESSING_ENABLED`：覆盖 pipeline enabled
- RealtimePipelineService：
  - `BLANKEND_REALTIME_PIPELINE_ENABLED`
  - `BLANKEND_REALTIME_PIPELINE_WORKERS`
- AiCoreService（部分）：
  - `FACE_DETECT_INTERVAL`：人脸检测间隔（已用于避免每帧检测）
  - `SIMPLE_FORWARD=1`：跳过 AI 推理，直接转发输入 JPEG（用于联通测试）
  - `INSIGHTFACE_APP` / `RVM_MODEL` / `DEPTH_MODEL` / `SWAP_MODEL` 等

---

## 5. 观测点（日志与状态接口）

### 5.1 常见日志

- `/video_in` 接收吞吐：`blankend.streaming` logger 中 `stream recv frames/bytes`
- 广播吞吐：`StreamHub.broadcast_bytes` 的 `stream broadcast frames/bytes clients`
- ZMQ：
  - 发送统计：`relay send frames/bytes dropped targets`
  - 接收统计：`relay recv frames/bytes`
- ffmpeg 解码 fps：`blankend.video_processing.hwdecode` 的 `ffmpeg decoded fps`

### 5.2 状态/配置 API（HTTP）

- VideoProcessing：
  - `GET /api/v1/video_processing/pipeline/status`
  - `POST /api/v1/video_processing/pipeline/config`
  - 以及 GPU/codec 探测：`/gpu/status`、`/codec/status`
- RealtimePipeline：
  - `GET /api/v1/realtime_pipeline/status`
  - `POST /api/v1/realtime_pipeline/config`
  - 录制：`/record/start`、`/record/stop`
- AiCore：
  - `GET /api/v1/ai/status`
  - `POST /api/v1/ai/process`

---

## 6. 代码检查发现的风险点（会影响下一步工作）

1. **AiCore 的人脸缓存变量未初始化会导致运行时异常（已修复）**
   - 影响：启用 AI 推理后，进入 `_swap_face` 分支会触发 AttributeError，导致设备 worker 置为 unhealthy。
   - 已在 `blankend/ai_core/service.py` 初始化 `_detect_interval` 与 `_last_face`。

2. **ZmqRelay 的 `echo_back` 语义容易误解**
   - 当前 `_recv_loop()` 不论 `echo_back` 是否开启都会 `enqueue(client_id, data)` 一次；
   - 若 `echo_back=true`，还会再次 `enqueue(client_id, data)`，存在重复发送的可能。
   - 若后续需要严格控制“是否回发给来源 client_id”，建议把“回发”与“echo_back”逻辑拆清楚。

3. **RealtimePipeline/AiCore 线程常驻，即使 enabled=false**
   - 目前是设计选择（便于热启用），但会导致“线程数”与“资源占用”在 disabled 状态下仍不可忽略；
   - 若下一步要做资源按需启停，需要把 `start()` 与 enabled 状态绑定，或引入显式 stop/join。

4. **`/video_in` 同时走本地 pipeline 与 ZMQ 转发，可能造成双路径回显**
   - 如果总线侧把帧又发回 `WEB_FRONTEND`，同时本地 `hub.broadcast` 也广播给前端，前端可能看到“重复帧源”。
   - 下一步要明确：前端订阅的是“直连 WS 广播”，还是“总线回流后的帧”，避免双播。

---

## 7. 推荐的下一步对齐（建议）

- 明确“单一真源”：
  - 方案 A：前端只收本机 WS 广播；ZMQ 仅用于 worker 间转发/远端处理
  - 方案 B：前端只收 `WEB_FRONTEND` 的 ZMQ 回流；WS 仅做上行输入
- 明确 client_id 语义与路由策略：
  - 谁是 `SRC`、谁是 `WEB_FRONTEND`、谁是 `AI_WORKER`，以及“回发/转发/广播”的预期
- 为不同处理模式加上明确的 frame 元数据（至少区分：原始帧/处理后帧/来自哪里）：
  - 可用 WebSocket text 控制信令，或把 JSON header 与 JPEG bytes 分帧发送

