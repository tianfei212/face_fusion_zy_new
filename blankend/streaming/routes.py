from fastapi import APIRouter, WebSocket
from fastapi.websockets import WebSocketDisconnect
import asyncio
import json
import logging
import os
from pathlib import Path
import queue
import threading
import time
from .hub import StreamHub
from ..video_processing.service import get_video_processing_service
from ..video_processing.hwdecode import FfmpegHwDecoder, HwDecodeConfig, run_decode_session
from ..realtime_pipeline.service import get_realtime_pipeline_service
from ..ai_core.manager import get_ai_manager

try:
    import zmq
except Exception:
    zmq = None

try:
    from turbojpeg import TurboJPEG
except Exception:
    TurboJPEG = None

jpeg = None
if TurboJPEG and os.getenv("BLANKEND_HW_TRANSCODE") == "1":
    try:
        jpeg = TurboJPEG()
    except Exception:
        jpeg = None

logger = logging.getLogger("blankend.streaming")

_relay_ref = None

def _load_config() -> dict:
    path = Path(__file__).resolve().parent.parent / "config.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

class ZmqRelay:
    def __init__(
        self,
        endpoint: str,
        hub: StreamHub,
        send_queue_size: int,
        identity: str,
        echo_back: bool,
        control_client_id: str,
        link_request_enabled: bool,
        link_request_payload: bytes,
        link_request_timeout_ms: int,
        link_request_retry_interval_ms: int,
        link_request_max_retries: int,
        heartbeat_enabled: bool,
        heartbeat_interval_ms: int,
        heartbeat_payload: bytes | None,
        forward_recv_enabled: bool,
        forward_recv_client_id: str | None,
        forward_recv_only: bool,
        default_client_id: str,
        frame_handler=None,
    ) -> None:
        self.endpoint = endpoint
        self.hub = hub
        self.identity = identity
        self.echo_back = echo_back
        self.default_client_id = default_client_id
        self.control_client_id = control_client_id
        self.link_request_enabled = link_request_enabled
        self.link_request_payload = link_request_payload
        self.link_request_timeout_ms = link_request_timeout_ms
        self.link_request_retry_interval_ms = link_request_retry_interval_ms
        self.link_request_max_retries = link_request_max_retries
        self.heartbeat_enabled = heartbeat_enabled
        self.heartbeat_interval_ms = heartbeat_interval_ms
        self.heartbeat_payload = heartbeat_payload
        self.forward_recv_enabled = forward_recv_enabled
        self.forward_recv_client_id = forward_recv_client_id
        self.forward_recv_only = forward_recv_only
        self.frame_handler = frame_handler
        self._loop = None
        self._started = False
        self._stop = threading.Event()
        self._linked = threading.Event()
        self._send_queue: queue.Queue[tuple[bytes, bytes]] = queue.Queue(maxsize=send_queue_size)
        self._recv_frames = 0
        self._recv_bytes = 0
        self._send_frames = 0
        self._send_bytes = 0
        self._send_dropped = 0
        self._last_log = time.monotonic()
        self._last_send_log = time.monotonic()
        self._ctx = zmq.Context.instance() if zmq else None
        self._send_socket = None
        self._recv_socket = None
        self._control_client_id_bytes = self.control_client_id.encode()
        self._default_client_id_bytes = self.default_client_id.encode() if self.default_client_id else None
        if self.forward_recv_client_id:
            parts = [p.strip() for p in str(self.forward_recv_client_id).split(",")]
            parts = [p for p in parts if p]
            self._forward_recv_client_id_bytes_list = [p.encode() for p in parts]
        else:
            self._forward_recv_client_id_bytes_list = []
        self._send_by_target: dict[bytes, tuple[int, int]] = {}

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def start(self) -> None:
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            logging.basicConfig(level=logging.INFO)
        logger.setLevel(logging.INFO)
        if self._started:
            return
        self._started = True
        if not zmq or not self._ctx:
            logger.error("zmq is not available, relay disabled")
            return
        self._send_socket = self._ctx.socket(zmq.DEALER)
        self._recv_socket = self._ctx.socket(zmq.DEALER)
        try:
            self._send_socket.setsockopt(zmq.IDENTITY, f"{self.identity}-send".encode())
            self._recv_socket.setsockopt(zmq.IDENTITY, self.identity.encode())
        except Exception:
            pass
        self._send_socket.connect(self.endpoint)
        self._recv_socket.connect(self.endpoint)
        logger.info(
            "zmq relay connected endpoint=%s identity=%s control_client_id=%s heartbeat=%s heartbeat_interval_ms=%s",
            self.endpoint,
            self.identity,
            self.control_client_id,
            self.heartbeat_enabled,
            self.heartbeat_interval_ms,
        )
        threading.Thread(target=self._send_loop, daemon=True).start()
        threading.Thread(target=self._recv_loop, daemon=True).start()
        if self.heartbeat_enabled:
            self.enqueue(self._control_client_id_bytes, self._build_heartbeat_payload())
            logger.info("zmq relay heartbeat sent endpoint=%s", self.endpoint)
            threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        if self.link_request_enabled:
            threading.Thread(target=self._link_request_loop, daemon=True).start()

    def enqueue(self, client_id: str | bytes | None, data: bytes) -> None:
        if not self._started or not self._send_socket:
            return
        if not client_id:
            if self._default_client_id_bytes is None:
                return
            client_id = self._default_client_id_bytes
        if isinstance(client_id, str):
            client_id = client_id.encode()
        try:
            self._send_queue.put_nowait((client_id, data))
        except queue.Full:
            self._send_dropped += 1
            try:
                self._send_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._send_queue.put_nowait((client_id, data))
            except queue.Full:
                self._send_dropped += 1
                pass

    def _send_loop(self) -> None:
        if not self._send_socket:
            return
        while not self._stop.is_set():
            try:
                client_id, data = self._send_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._send_socket.send_multipart([client_id, data])
                self._send_frames += 1
                self._send_bytes += len(data)
                prev = self._send_by_target.get(client_id)
                if prev is None:
                    self._send_by_target[client_id] = (1, len(data))
                else:
                    self._send_by_target[client_id] = (prev[0] + 1, prev[1] + len(data))
                now = time.monotonic()
                if now - self._last_send_log >= 1:
                    def _fmt_target(cid: bytes, v: tuple[int, int]) -> str:
                        try:
                            cid_s = cid.decode("utf-8", errors="replace")
                        except Exception:
                            cid_s = repr(cid)
                        if len(cid_s) > 32:
                            cid_s = cid_s[:32] + "…"
                        return f"{cid_s}:{v[0]}/{v[1]}"

                    top = sorted(self._send_by_target.items(), key=lambda kv: kv[1][1], reverse=True)[:5]
                    top_s = ",".join(_fmt_target(k, v) for k, v in top)
                    logger.info(
                        "relay send frames=%s bytes=%s dropped=%s targets=%s",
                        self._send_frames,
                        self._send_bytes,
                        self._send_dropped,
                        top_s,
                    )
                    self._send_frames = 0
                    self._send_bytes = 0
                    self._send_dropped = 0
                    self._send_by_target = {}
                    self._last_send_log = now
                if client_id == self._control_client_id_bytes and data == self.link_request_payload and not self._linked.is_set():
                    self._linked.set()
                    logger.info("zmq relay channel linked endpoint=%s", self.endpoint)
            except Exception:
                time.sleep(0.1)

    def _recv_loop(self) -> None:
        if not self._recv_socket:
            return
        while not self._stop.is_set():
            if self._loop is None:
                time.sleep(0.05)
                continue
            try:
                msg = self._recv_socket.recv_multipart()
            except Exception:
                time.sleep(0.05)
                continue
            if len(msg) < 2:
                continue
            client_id = msg[0]
            data = msg[1]
            if client_id == self._control_client_id_bytes:
                if not self._linked.is_set():
                    self._linked.set()
                    logger.info("zmq relay channel linked endpoint=%s", self.endpoint)
                continue
            try:
                if self.frame_handler is not None:
                    # Log relay handler call for debug
                    # if self._recv_frames % 60 == 0:
                    #     logger.info("relay_frame_handler_call client_id=%s size=%s", client_id.hex()[:8], len(data))
                    if self.frame_handler(client_id, data):
                        continue
            except Exception:
                pass
            self._recv_frames += 1
            self._recv_bytes += len(data)
            now = time.monotonic()
            if now - self._last_log >= 1:
                logger.info("relay recv frames=%s bytes=%s", self._recv_frames, self._recv_bytes)
                self._recv_frames = 0
                self._recv_bytes = 0
                self._last_log = now
            self.enqueue(client_id, data)
            if self.forward_recv_enabled and self._forward_recv_client_id_bytes_list:
                for target_id in self._forward_recv_client_id_bytes_list:
                    if target_id != client_id:
                        self.enqueue(target_id, data)
            try:
                asyncio.run_coroutine_threadsafe(self.hub.broadcast_bytes(data), self._loop)
            except Exception:
                pass
            if self.forward_recv_only:
                continue
            if self.echo_back:
                self.enqueue(client_id, data)

    def _link_request_loop(self) -> None:
        if not self._started or self._send_socket is None:
            return
        retries_left = self.link_request_max_retries
        timeout_s = max(0, self.link_request_timeout_ms) / 1000.0
        interval_s = max(0, self.link_request_retry_interval_ms) / 1000.0
        while not self._stop.is_set() and not self._linked.is_set() and retries_left > 0:
            self.enqueue(self._control_client_id_bytes, self.link_request_payload)
            logger.info("zmq relay link request sent endpoint=%s", self.endpoint)
            if self._linked.wait(timeout=timeout_s):
                return
            retries_left -= 1
            if retries_left > 0 and interval_s > 0:
                time.sleep(interval_s)
        if not self._linked.is_set():
            logger.warning("zmq relay link request timeout endpoint=%s", self.endpoint)

    def _build_heartbeat_payload(self) -> bytes:
        if self.heartbeat_payload is not None:
            return self.heartbeat_payload
        return json.dumps(
            {"type": "heartbeat", "identity": self.identity, "ts": time.time()},
            ensure_ascii=False,
        ).encode("utf-8")

    def _heartbeat_loop(self) -> None:
        if not self._started or self._send_socket is None:
            return
        interval_s = max(0, self.heartbeat_interval_ms) / 1000.0
        while not self._stop.is_set():
            self.enqueue(self._control_client_id_bytes, self._build_heartbeat_payload())
            if interval_s <= 0:
                time.sleep(0.2)
            else:
                time.sleep(interval_s)

def get_router() -> APIRouter:
    global _relay_ref
    router = APIRouter()
    hub = StreamHub()
    vp_service = get_video_processing_service()
    rt_service = get_realtime_pipeline_service()
    ai_manager = get_ai_manager()
    config = _load_config()
    stream_config = config.get("stream", {}) if isinstance(config, dict) else {}
    relay_enabled = bool(stream_config.get("relay_enabled", True))
    relay_enabled_env = os.getenv("BLANKEND_RELAY_ENABLED")
    if relay_enabled_env is not None:
        relay_enabled = relay_enabled_env.strip() not in {"0", "false", "False", "no", "NO", ""}
    zmq_endpoint = os.getenv("BLANKEND_ZMQ_ENDPOINT") or stream_config.get("zmq_endpoint") or "tcp://121.199.29.249:8100"
    try:
        send_queue_size = int(stream_config.get("send_queue_size", 4))
    except Exception:
        send_queue_size = 4
    if send_queue_size < 1:
        send_queue_size = 1
    try:
        jpeg_quality = int(stream_config.get("jpeg_quality", 85))
    except Exception:
        jpeg_quality = 85
    if jpeg_quality < 1 or jpeg_quality > 100:
        jpeg_quality = 85
    reencode_enabled = bool(stream_config.get("reencode_enabled", False))
    zmq_identity = os.getenv("BLANKEND_ZMQ_IDENTITY") or stream_config.get("zmq_identity") or "AI_WORKER"
    # Important: Do not use AI_WORKER as default client_id for video source, otherwise it loops back
    default_client_id = str(stream_config.get("default_client_id") or "WEB_FRONTEND")
    
    echo_back = bool(stream_config.get("echo_back", True))
    control_client_id = str(stream_config.get("control_client_id") or "__blankend_control__")
    link_request_enabled = bool(stream_config.get("link_request_enabled", True))
    try:
        link_request_timeout_ms = int(stream_config.get("link_request_timeout_ms", 1000))
    except Exception:
        link_request_timeout_ms = 1000
    try:
        link_request_retry_interval_ms = int(stream_config.get("link_request_retry_interval_ms", 1000))
    except Exception:
        link_request_retry_interval_ms = 1000
    try:
        link_request_max_retries = int(stream_config.get("link_request_max_retries", 5))
    except Exception:
        link_request_max_retries = 5
    heartbeat_enabled = bool(stream_config.get("heartbeat_enabled", True))
    try:
        heartbeat_interval_ms = int(stream_config.get("heartbeat_interval_ms", 1000))
    except Exception:
        heartbeat_interval_ms = 1000
    heartbeat_payload = None
    if "heartbeat_payload" in stream_config:
        payload_cfg = stream_config.get("heartbeat_payload")
        if isinstance(payload_cfg, str):
            heartbeat_payload = payload_cfg.encode()
        else:
            try:
                heartbeat_payload = json.dumps(payload_cfg, ensure_ascii=False).encode("utf-8")
            except Exception:
                heartbeat_payload = b'{"type":"heartbeat"}'
    forward_recv_enabled = bool(stream_config.get("forward_recv_enabled", False))
    forward_recv_client_id = stream_config.get("forward_recv_client_id")
    if forward_recv_client_id is not None:
        forward_recv_client_id = str(forward_recv_client_id)
    forward_recv_only = bool(stream_config.get("forward_recv_only", True))
    if "link_request_payload" in stream_config:
        payload_cfg = stream_config.get("link_request_payload")
        if isinstance(payload_cfg, str):
            link_request_payload = payload_cfg.encode()
        else:
            try:
                link_request_payload = json.dumps(payload_cfg, ensure_ascii=False).encode("utf-8")
            except Exception:
                link_request_payload = b'{"type":"link_request"}'
    else:
        link_request_payload = json.dumps(
            {"type": "link_request", "identity": zmq_identity, "ts": time.time()},
            ensure_ascii=False,
        ).encode("utf-8")
    relay = (
        ZmqRelay(
            zmq_endpoint,
            hub,
            send_queue_size,
            zmq_identity,
            echo_back,
            control_client_id,
            link_request_enabled,
            link_request_payload,
            link_request_timeout_ms,
            link_request_retry_interval_ms,
            link_request_max_retries,
            heartbeat_enabled,
            heartbeat_interval_ms,
            heartbeat_payload,
            forward_recv_enabled,
            forward_recv_client_id,
            forward_recv_only,
            default_client_id,
            frame_handler=(
                lambda cid, data: (
                    ai_manager.submit_frame(cid, data) if ai_manager.enabled() else rt_service.submit(cid, data)
                )
            ),
        )
        if relay_enabled
        else None
    )
    _relay_ref = relay

    @router.on_event("startup")
    async def start_relay_on_startup() -> None:
        loop = asyncio.get_running_loop()
        vp_service.attach(loop, hub.broadcast_bytes)
        vp_service.start()
        rt_service.attach(
            sender=(lambda cid, data: relay.enqueue(cid, data)) if relay is not None else (lambda _cid, _data: None),
            broadcaster=(lambda data: asyncio.run_coroutine_threadsafe(hub.broadcast_bytes(data), loop)),
        )
        ai_manager.attach(
            sender=(lambda cid, data: relay.enqueue(cid, data)) if relay is not None else (lambda _cid, _data: None),
            broadcaster=(lambda data: asyncio.run_coroutine_threadsafe(hub.broadcast_bytes(data), loop)),
        )
        if relay is None:
            return
        if relay._loop is None:
            relay.set_loop(loop)
        relay.start()

    @router.websocket("/video_in")
    async def stream(ws: WebSocket):
        await hub.connect(ws)
        if relay is not None:
            if relay._loop is None:
                relay.set_loop(asyncio.get_running_loop())
            relay.start()
        recv_frames = 0
        recv_bytes = 0
        last_log = time.monotonic()
        try:
            while True:
                try:
                    msg = await ws.receive()
                except RuntimeError:
                    # 'Cannot call "receive" once a disconnect message has been received'
                    break
                
                data = msg.get("bytes")
                text = msg.get("text")
                if data is not None:
                    # Direct submission to AI Manager (Bypass ZMQ loopback requirement)
                    if ai_manager.enabled():
                        if ai_manager.submit_frame(default_client_id.encode(), data):
                            continue

                    if vp_service.enabled():
                        recv_frames += 1
                        recv_bytes += len(data)
                        now = time.monotonic()
                        if now - last_log >= 1:
                            logger.info("stream recv frames=%s bytes=%s", recv_frames, recv_bytes)
                            recv_frames = 0
                            recv_bytes = 0
                            last_log = now
                        if relay is not None:
                            relay.enqueue(default_client_id, data)
                        accepted = vp_service.submit(time.time(), data)
                        if not accepted:
                            await hub.broadcast_bytes(data)
                        continue
                    if reencode_enabled and jpeg:
                        try:
                            decoded = jpeg.decode(data)
                            data = jpeg.encode(decoded, quality=jpeg_quality)
                        except Exception:
                            pass

                    recv_frames += 1
                    recv_bytes += len(data)
                    now = time.monotonic()
                    if now - last_log >= 1:
                        logger.info("stream recv frames=%s bytes=%s", recv_frames, recv_bytes)
                        recv_frames = 0
                        recv_bytes = 0
                        last_log = now
                    if relay is not None:
                        relay.enqueue(default_client_id, data)
                    await hub.broadcast_bytes(data)
                elif text is not None:
                    logger.info("stream recv text size=%s", len(text))
                    await hub.broadcast_text(text)
        except WebSocketDisconnect:
            hub.disconnect(ws)

    @router.websocket("/video_codec")
    async def stream_codec(ws: WebSocket):
        await hub.connect(ws)
        if relay is not None:
            if relay._loop is None:
                relay.set_loop(asyncio.get_running_loop())
            relay.start()
        codec = (ws.query_params.get("codec") or "h264").lower()
        prefer_gpu = (ws.query_params.get("prefer_gpu") or "1").strip() not in {"0", "false", "False", "no", "NO", ""}
        try:
            quality = int(ws.query_params.get("mjpeg_quality") or "5")
        except Exception:
            quality = 5
        if quality < 2 or quality > 31:
            quality = 5

        decoder = FfmpegHwDecoder(HwDecodeConfig(codec=codec, prefer_gpu=prefer_gpu, mjpeg_quality=quality))
        stop_evt = threading.Event()

        try:
            decoder.start()
        except Exception as e:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False))
            hub.disconnect(ws)
            return

        loop = asyncio.get_running_loop()

        def _broadcast_to_loop(frame: bytes) -> None:
            if ai_manager.enabled():
                if ai_manager.submit_frame(default_client_id.encode(), frame):
                    return
            if relay is not None:
                relay.enqueue(default_client_id, frame)
            try:
                asyncio.run_coroutine_threadsafe(hub.broadcast_bytes(frame), loop)
            except Exception:
                pass

        t = threading.Thread(target=run_decode_session, args=(decoder, stop_evt, _broadcast_to_loop), daemon=True)
        t.start()

        try:
            await ws.send_text(json.dumps({"type": "started", "codec": codec, "mode": decoder.status().get("mode")}, ensure_ascii=False))
            while True:
                msg = await ws.receive()
                data = msg.get("bytes")
                text = msg.get("text")
                if data is not None:
                    decoder.feed(data)
                elif text is not None:
                    try:
                        payload = json.loads(text)
                    except Exception:
                        payload = None
                    if isinstance(payload, dict) and payload.get("type") == "reset":
                        stop_evt.set()
                        try:
                            decoder.stop()
                        except Exception:
                            pass
                        stop_evt = threading.Event()
                        decoder = FfmpegHwDecoder(
                            HwDecodeConfig(codec=codec, prefer_gpu=prefer_gpu, mjpeg_quality=quality)
                        )
                        decoder.start()
                        t = threading.Thread(
                            target=run_decode_session, args=(decoder, stop_evt, _broadcast_to_loop), daemon=True
                        )
                        t.start()
                        await ws.send_text(
                            json.dumps({"type": "restarted", "mode": decoder.status().get("mode")}, ensure_ascii=False)
                        )
        except WebSocketDisconnect:
            pass
        finally:
            stop_evt.set()
            try:
                decoder.stop()
            except Exception:
                pass
            hub.disconnect(ws)

    return router


def start_relay(loop: asyncio.AbstractEventLoop) -> None:
    if _relay_ref is None:
        return
    if _relay_ref._loop is None:
        _relay_ref.set_loop(loop)
    _relay_ref.start()
