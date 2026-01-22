from __future__ import annotations

import os
import queue
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from blankend.ai_core.manager import AiActiveStandbyManager


def main() -> int:
    os.environ["SIMPLE_FORWARD"] = "1"

    out_q: queue.Queue[tuple[bytes, bytes]] = queue.Queue()

    def sender(cid: bytes, data: bytes) -> None:
        out_q.put((cid, data))

    mgr = AiActiveStandbyManager()
    mgr.attach(sender=sender, broadcaster=None)
    mgr.start()
    mgr.configure(enabled=True)

    mgr.activate(0)
    mgr.load_dfm("dummy.onnx")
    mgr.activate(1)

    cid = b"__smoke__"
    payload = b"not_a_real_jpeg_but_ok_in_simple_forward"
    ok = mgr.submit_frame(cid, payload, time.time())
    if not ok:
        return 2

    try:
        got_cid, got = out_q.get(timeout=5.0)
    except Exception:
        return 3
    if got_cid != cid or got != payload:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
