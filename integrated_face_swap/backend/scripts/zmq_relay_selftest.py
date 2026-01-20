import asyncio
import json
import socket
import sys
import threading
import time
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    import zmq
except Exception:
    zmq = None

from blankend.streaming.hub import StreamHub
from blankend.streaming.routes import ZmqRelay


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def main() -> int:
    if zmq is None:
        print("zmq is not available; selftest skipped")
        return 0

    port = _pick_free_port()
    endpoint = f"tcp://127.0.0.1:{port}"
    ctx = zmq.Context.instance()
    router = ctx.socket(zmq.ROUTER)
    router.bind(endpoint)

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=_run_loop, args=(loop,), daemon=True)
    loop_thread.start()

    identity = "SELFTEST_WORKER"
    control_client_id = "__blankend_control__"
    default_client_id = identity
    payload = json.dumps({"type": "link_request", "identity": identity, "ts": time.time()}, ensure_ascii=False).encode(
        "utf-8"
    )

    hub = StreamHub()
    relay = ZmqRelay(
        endpoint=endpoint,
        hub=hub,
        send_queue_size=8,
        identity=identity,
        echo_back=False,
        control_client_id=control_client_id,
        link_request_enabled=True,
        link_request_payload=payload,
        link_request_timeout_ms=200,
        link_request_retry_interval_ms=50,
        link_request_max_retries=10,
        heartbeat_enabled=True,
        heartbeat_interval_ms=1000,
        heartbeat_payload=None,
        forward_recv_enabled=True,
        forward_recv_client_id="WEB_FRONTEND",
        forward_recv_only=True,
        default_client_id=default_client_id,
    )
    relay.set_loop(loop)
    relay.start()

    router.RCVTIMEO = 2000
    msg = router.recv_multipart()
    if len(msg) < 3:
        raise RuntimeError(f"unexpected handshake frame count: {len(msg)}")

    routing_id, client_id, data = msg[0], msg[1], msg[2]
    if client_id != control_client_id.encode():
        raise RuntimeError(f"unexpected control client_id: {client_id!r}")
    if b"heartbeat" not in data and b"link_request" not in data:
        raise RuntimeError(f"unexpected control payload: {data!r}")

    deadline = time.monotonic() + 2.0
    while not relay._linked.is_set() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not relay._linked.is_set():
        raise RuntimeError("relay did not mark linked after link request send")

    relay.enqueue(default_client_id, b"hello")
    router.RCVTIMEO = 200
    deadline2 = time.monotonic() + 2.0
    while time.monotonic() < deadline2:
        msg2 = router.recv_multipart()
        if len(msg2) < 3:
            continue
        _, client_id2, data2 = msg2[0], msg2[1], msg2[2]
        if client_id2 == default_client_id.encode() and data2 == b"hello":
            break
    else:
        raise RuntimeError("did not receive expected data frame within timeout")

    recv_routing_id = identity.encode()
    router.send_multipart([recv_routing_id, b"SRC", b"frame"])
    deadline3 = time.monotonic() + 2.0
    got_src = False
    got_frontend = False
    while time.monotonic() < deadline3:
        msg3 = router.recv_multipart()
        if len(msg3) < 3:
            continue
        send_routing_id, client_id3, data3 = msg3[0], msg3[1], msg3[2]
        if send_routing_id != routing_id or data3 != b"frame":
            continue
        if client_id3 == b"SRC":
            got_src = True
        if client_id3 == b"WEB_FRONTEND":
            got_frontend = True
        if got_src and got_frontend:
            break
    else:
        raise RuntimeError(f"did not receive forwarded frame to SRC+WEB_FRONTEND (src={got_src} frontend={got_frontend})")

    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
