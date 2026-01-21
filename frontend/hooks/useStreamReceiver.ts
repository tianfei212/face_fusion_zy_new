import { useEffect, useRef, useState } from 'react';
import type { RefObject } from 'react';

const toWsUrl = (httpUrl: string) => {
  try {
    if (httpUrl.startsWith('ws://') || httpUrl.startsWith('wss://')) {
      return httpUrl;
    }
    // Handle relative paths
    if (httpUrl.startsWith('/')) {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const path = httpUrl === '/' ? '/video_in' : httpUrl;
      return `${protocol}//${window.location.host}${path}`;
    }
    const u = new URL(httpUrl);
    u.protocol = u.protocol === 'https:' ? 'wss:' : 'ws:';
    if ((u.pathname === '/' || u.pathname.length === 0) && (u.hostname === 'localhost' || u.hostname === '127.0.0.1')) {
      u.pathname = '/video_in';
    }
    return u.toString();
  } catch {
    return 'ws://localhost:5100/video_in';
  }
};

export const useStreamReceiver = (
  backendBaseUrl: string,
  canvasRef: RefObject<HTMLCanvasElement>
) => {
  const [status, setStatus] = useState('未连接');
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!backendBaseUrl) {
      setStatus('未连接');
      return;
    }
    const wsUrl = toWsUrl(backendBaseUrl);
    const ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;
    setStatus('连接中');

    ws.onopen = () => setStatus('已连接');
    ws.onclose = () => setStatus('未连接');
    ws.onerror = () => setStatus('错误');
    ws.onmessage = async (ev) => {
      const buf = ev.data as ArrayBuffer;
      if (!canvasRef.current || !buf) return;
      const blob = new Blob([buf], { type: 'image/jpeg' });
      try {
        const bmp = await createImageBitmap(blob);
        const c = canvasRef.current;
        c.width = bmp.width; c.height = bmp.height;
        const ctx = c.getContext('2d');
        if (ctx) ctx.drawImage(bmp, 0, 0);
      } catch {}
    };

    return () => { try { ws.close(); } catch {} };
  }, [backendBaseUrl, canvasRef]);

  return { status };
};
