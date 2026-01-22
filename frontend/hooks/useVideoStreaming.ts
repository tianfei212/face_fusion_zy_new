import { useEffect, useRef, useState } from 'react';
import type { RefObject } from 'react';
import { LogLevel } from '../types';

const toWsUrl = (httpUrl: string) => {
  if (httpUrl.startsWith('/')) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}${httpUrl}`;
  }
  try {
    if (httpUrl.startsWith('ws://') || httpUrl.startsWith('wss://')) {
      return httpUrl;
    }
    const u = new URL(httpUrl);
    u.protocol = u.protocol === 'https:' ? 'wss:' : 'ws:';
    if ((u.pathname === '/' || u.pathname.length === 0) && (u.hostname === 'localhost' || u.hostname === '127.0.0.1')) {
      u.pathname = '/video_in';
    }
    return u.toString();
  } catch {
    console.warn('Invalid URL for WebSocket:', httpUrl);
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}/video_in`;
  }
};

export const useVideoStreaming = (
  cameraStream: MediaStream | null,
  isStreaming: boolean,
  sendBaseUrl: string,
  receiveBaseUrl: string,
  log: (level: LogLevel, message: string) => void,
  remoteCanvasRef?: RefObject<HTMLCanvasElement>
) => {
  const [status, setStatus] = useState('未连接');
  const wsRef = useRef<WebSocket | null>(null);
  const captureVideoRef = useRef<HTMLVideoElement | null>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const drawTimerRef = useRef<number | null>(null);
  const statsTimerRef = useRef<number | null>(null);
  const frameCallbackHandleRef = useRef<number | null>(null);
  const streamActiveRef = useRef(false);
  const encoderRef = useRef<any>(null);
  const sentCountRef = useRef(0);
  const recvCountRef = useRef(0);
  const bcRef = useRef<BroadcastChannel | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const workerUrlRef = useRef<string | null>(null);
  const lastBroadcastTimeRef = useRef(0); // Throttle control

  useEffect(() => {
    const stopAll = () => {
      if (workerRef.current) {
        try { workerRef.current.terminate(); } catch {}
        workerRef.current = null;
      }
      if (workerUrlRef.current) {
        try { URL.revokeObjectURL(workerUrlRef.current); } catch {}
        workerUrlRef.current = null;
      }
      if (bcRef.current) {
        try { bcRef.current.close(); } catch {}
        bcRef.current = null;
      }
      if (drawTimerRef.current) {
        window.clearInterval(drawTimerRef.current);
        drawTimerRef.current = null;
      }
      if (statsTimerRef.current) {
        window.clearInterval(statsTimerRef.current);
        statsTimerRef.current = null;
      }
      if (frameCallbackHandleRef.current !== null && captureVideoRef.current?.cancelVideoFrameCallback) {
        captureVideoRef.current.cancelVideoFrameCallback(frameCallbackHandleRef.current);
        frameCallbackHandleRef.current = null;
      }
      streamActiveRef.current = false;
      sentCountRef.current = 0;
      recvCountRef.current = 0;
      if (wsRef.current) {
        try { wsRef.current.close(); } catch {}
        wsRef.current = null;
      }
      setStatus('未连接');
    };

    if (!isStreaming) {
      stopAll();
      return;
    }

    if (!cameraStream) {
      log('warn', '开启传输失败：摄像头未启动');
      setStatus('摄像头未启动');
      return;
    }

    const wsUrl = toWsUrl(receiveBaseUrl || sendBaseUrl);
    const ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;
    setStatus('连接中');
    log('info', `正在连接视频通道: ${wsUrl}`);
    let ready = false;
    const updateStatus = () => {
      if (ready) setStatus('已连接');
      else setStatus('连接中');
    };

    // Initialize Broadcast Channel
    if ('BroadcastChannel' in window) {
      try {
        bcRef.current = new BroadcastChannel('stream_sync_channel');
      } catch {
        bcRef.current = null;
      }
    }

    // --- Optimized Broadcast Function ---
    const broadcastMessage = (type: string, buf: ArrayBuffer) => {
      const bc = bcRef.current;
      if (!bc) return;
      
      // Throttle: Max 30 FPS broadcast to avoid choking main thread
      const now = Date.now();
      if (now - lastBroadcastTimeRef.current < 33) return; 

      try {
        bc.postMessage({ type, payload: buf });
        lastBroadcastTimeRef.current = now;
      } catch {}
    };

    const handleVideoFrame = async (ev: MessageEvent) => {
      const buf = ev.data as ArrayBuffer;
      if (!buf) return;
      broadcastMessage('frame', buf);

      if (!remoteCanvasRef?.current) return;
      recvCountRef.current += 1;
      
      try {
        // Optimized Draw Logic (ImageDecoder or ImageBitmap)
        const Decoder = (window as any).ImageDecoder;
        if (Decoder) {
          const decoder = new Decoder({ data: buf, type: 'image/jpeg' });
          const { image } = await decoder.decode();
          const c = remoteCanvasRef.current;
          const w = (image as any).displayWidth ?? image.width;
          const h = (image as any).displayHeight ?? image.height;
          if (c.width !== w) c.width = w;
          if (c.height !== h) c.height = h;
          
          const ctx = c.getContext('2d', { alpha: false });
          if (ctx) ctx.drawImage(image, 0, 0);
          
          image.close();
          decoder.close();
          return;
        }

        // Fallback
        const blob = new Blob([buf], { type: 'image/jpeg' });
        const bmp = await createImageBitmap(blob);
        const c = remoteCanvasRef.current;
        if (c.width !== bmp.width) c.width = bmp.width;
        if (c.height !== bmp.height) c.height = bmp.height;
        
        const ctx = c.getContext('2d', { alpha: false });
        if (ctx) ctx.drawImage(bmp, 0, 0);
        
        bmp.close();
      } catch {}
    };

    ws.onopen = () => {
      ready = true;
      updateStatus();
      log('info', '视频通道已连接，开始推流（JPEG帧）');
      
      // ... Stats timer logic (kept same) ...
      if (!statsTimerRef.current) {
        statsTimerRef.current = window.setInterval(() => {
          const sent = sentCountRef.current;
          const recv = recvCountRef.current;
          if (sent > 0 || recv > 0) {
            log('info', `视频流统计：发送${sent}帧/秒 接收${recv}帧/秒`);
          }
          sentCountRef.current = 0;
          recvCountRef.current = 0;
        }, 1000);
      }

      streamActiveRef.current = true;
      if (!captureVideoRef.current) captureVideoRef.current = document.createElement('video');
      const v = captureVideoRef.current;
      v.srcObject = cameraStream;
      v.muted = true;
      v.play().catch(() => {});
      if (!offscreenCanvasRef.current) offscreenCanvasRef.current = document.createElement('canvas');
      const oc = offscreenCanvasRef.current;
      const targetInterval = 33;

      const canUseEncoder = typeof (window as any).ImageEncoder !== 'undefined' && typeof (window as any).VideoFrame !== 'undefined';
      
      const sendFrame = async () => {
        const w = v.videoWidth; const h = v.videoHeight;
        if (w === 0 || h === 0) return;
        
        oc.width = w; oc.height = h;
        const ctx = oc.getContext('2d', { alpha: false });
        if (!ctx) return;
        ctx.drawImage(v, 0, 0, w, h);

        // --- ENCODER PATH ---
        if (canUseEncoder) {
          try {
            if (!encoderRef.current) {
              encoderRef.current = new (window as any).ImageEncoder({ type: 'image/jpeg' });
            }
            const frame = new (window as any).VideoFrame(oc);
            const result = await encoderRef.current.encode(frame, { quality: 0.7 });
            frame.close();
            const data = result?.data;
            if (data) {
              const buf = data instanceof ArrayBuffer ? data : data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
              
              broadcastMessage('raw_frame', buf);
              
              if (wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send(buf);
                sentCountRef.current += 1;
              }
            }
          } catch {}
        } 
        // --- BLOB PATH (Fallback) ---
        else {
          await new Promise<void>(resolve => oc.toBlob(async (blob) => {
            if (!blob) return resolve();
            try {
              const buf = await blob.arrayBuffer();
              
              broadcastMessage('raw_frame', buf);
              
              if (wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send(buf);
                sentCountRef.current += 1;
              }
            } catch {}
            resolve();
          }, 'image/jpeg', 0.7));
        }
      };

      // ... Worker Timer Logic (Kept same) ...
      let inFlight = false;
      const tick = async () => {
        if (!streamActiveRef.current) return;
        if (inFlight) return;
        inFlight = true;
        try { await sendFrame(); } finally { inFlight = false; }
      };
      
      try {
        const workerCode = `
          self.onmessage = function(e) {
            if (!e || !e.data || e.data.type !== 'start') return;
            var interval = e.data.interval || 33;
            setInterval(function() { self.postMessage(0); }, interval);
          };
        `;
        const workerUrl = URL.createObjectURL(new Blob([workerCode], { type: 'application/javascript' }));
        workerUrlRef.current = workerUrl;
        workerRef.current = new Worker(workerUrl);
        workerRef.current.onmessage = () => { void tick(); };
        workerRef.current.postMessage({ type: 'start', interval: targetInterval });
      } catch {
        drawTimerRef.current = window.setInterval(() => { void tick(); }, targetInterval);
      }
    };

    ws.onmessage = handleVideoFrame;

    // ... Cleanup and error handlers (Kept same) ...
    ws.onclose = () => {
      ready = false;
      log('warn', '视频通道连接已关闭');
      updateStatus();
    };
    ws.onerror = () => {
      ready = false;
      log('error', '视频通道连接错误');
      updateStatus();
    };

    return stopAll;
  }, [cameraStream, isStreaming, sendBaseUrl, receiveBaseUrl, log, remoteCanvasRef]);

  return { streamingStatus: status };
};
