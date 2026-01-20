from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blankend.video_processing.service import get_video_processing_service


async def _noop_broadcast(_: bytes) -> None:
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="blankend/assets/human_pic/001.png", help="Input file to use as bytes payload")
    ap.add_argument("--frames", type=int, default=300, help="Number of frames to submit")
    ap.add_argument("--mode", choices=["pipeline", "direct"], default="pipeline")
    ap.add_argument("--enable", action="store_true", help="Enable pipeline")
    ap.add_argument("--queue", type=int, default=8)
    ap.add_argument("--decode", type=int, default=1)
    ap.add_argument("--encode", type=int, default=1)
    args = ap.parse_args()

    p = Path(args.file)
    if not p.exists():
        raise SystemExit(f"file not found: {p}")
    payload = p.read_bytes()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    t0 = time.perf_counter()
    accepted = 0
    if args.mode == "direct":
        for _ in range(int(args.frames)):
            _ = payload
            accepted += 1
        st = {
            "enabled": False,
            "dropped": 0,
            "in_queue_len": 0,
            "mid_queue_len": 0,
            "decode_workers": 0,
            "encode_workers": 0,
        }
    else:
        svc = get_video_processing_service()
        svc.attach(loop, _noop_broadcast)
        svc.start()
        svc.configure(enabled=bool(args.enable), queue_size=args.queue, decode_workers=args.decode, encode_workers=args.encode)
        for _ in range(int(args.frames)):
            if svc.submit(time.time(), payload):
                accepted += 1
    t1 = time.perf_counter()

    time.sleep(0.5)
    if args.mode != "direct":
        st = svc.status()
    elapsed = max(1e-9, t1 - t0)
    fps = accepted / elapsed
    print(
        f"mode={args.mode} pipeline_enabled={st['enabled']} accepted={accepted} dropped={st['dropped']} submit_fps={fps:.2f} "
        f"in_q={st['in_queue_len']} mid_q={st['mid_queue_len']} decode={st['decode_workers']} encode={st['encode_workers']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
