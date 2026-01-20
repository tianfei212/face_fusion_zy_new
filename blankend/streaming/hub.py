from typing import Set
import logging
import time
from fastapi import WebSocket

logger = logging.getLogger("blankend.streaming")

class StreamHub:
    def __init__(self) -> None:
        self.clients: Set[WebSocket] = set()
        self._sent_frames = 0
        self._sent_bytes = 0
        self._last_sent_log = time.monotonic()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)
        logger.info("stream client connected total=%s", len(self.clients))

    def disconnect(self, ws: WebSocket) -> None:
        self.clients.discard(ws)
        logger.info("stream client disconnected total=%s", len(self.clients))

    async def broadcast_bytes(self, data: bytes) -> None:
        dead = []
        for c in list(self.clients):
            try:
                await c.send_bytes(data)
                self._sent_frames += 1
                self._sent_bytes += len(data)
            except Exception:
                dead.append(c)
        for d in dead:
            self.disconnect(d)
        now = time.monotonic()
        if now - self._last_sent_log >= 1 and self._sent_frames > 0:
            logger.info(
                "stream broadcast frames=%s bytes=%s clients=%s",
                self._sent_frames,
                self._sent_bytes,
                len(self.clients),
            )
            self._sent_frames = 0
            self._sent_bytes = 0
            self._last_sent_log = now

    async def broadcast_text(self, text: str) -> None:
        dead = []
        for c in list(self.clients):
            try:
                await c.send_text(text)
            except Exception:
                dead.append(c)
        for d in dead:
            self.disconnect(d)
