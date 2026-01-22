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
  const bcRef = useRef<BroadcastChannel | null>(null);
  const lastBroadcastTimeRef = useRef(0); // For throttling

  useEffect(() => {
    // 1. Setup Broadcast Channel
    if ('BroadcastChannel' in window) {
      try {
        bcRef.current = new BroadcastChannel('stream_sync_channel');
      } catch {
        bcRef.current = null;
      }
    }

    // 2. Setup WebSocket
    if (!backendBaseUrl) {
      setStatus('未连接');
      // Even without WS, we keep BC open to clean up properly
      return () => { try { bcRef.current?.close(); } catch {} };
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
      if (!buf) return;

      // A. Broadcast (Throttled to ~30 FPS to save CPU)
      const now = Date.now();
      if (now - lastBroadcastTimeRef.current > 33) {
        if (bcRef.current) {
          // Send raw buffer, let Viewer wrap it in Blob if needed
          // Wrapping { type: 'frame', payload: buf } to match Viewer expectation
          bcRef.current.postMessage({ type: 'frame', payload: buf });
          lastBroadcastTimeRef.current = now;
        }
      }

      // B. Local Draw
      if (!canvasRef.current) return;
      
      // Use Blob only for createImageBitmap
      const blob = new Blob([buf], { type: 'image/jpeg' });
      try {
        const bmp = await createImageBitmap(blob);
        const c = canvasRef.current;
        if (c.width !== bmp.width) c.width = bmp.width;
        if (c.height !== bmp.height) c.height = bmp.height;
        
        const ctx = c.getContext('2d', { alpha: false }); // Opt: alpha false
        if (ctx) ctx.drawImage(bmp, 0, 0);
        bmp.close(); // Opt: Close bitmap
      } catch {}
    };

    return () => {
      try { ws.close(); } catch {}
      try { bcRef.current?.close(); } catch {}
      bcRef.current = null;
    };
  }, [backendBaseUrl, canvasRef]);

  return { status };
};