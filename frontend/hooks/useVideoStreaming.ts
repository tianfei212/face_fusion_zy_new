import { useEffect, useRef, useState } from 'react';
import type { RefObject } from 'react';
import { LogLevel } from '../types';

const toWsUrl = (httpUrl: string) => {
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
    return 'ws://localhost:5100/video_in';
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
  const sendWsRef = useRef<WebSocket | null>(null);
  const recvWsRef = useRef<WebSocket | null>(null);
  const captureVideoRef = useRef<HTMLVideoElement | null>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const drawTimerRef = useRef<number | null>(null);
  const statsTimerRef = useRef<number | null>(null);
  const frameCallbackHandleRef = useRef<number | null>(null);
  const streamActiveRef = useRef(false);
  const encoderRef = useRef<any>(null);
  const sentCountRef = useRef(0);
  const recvCountRef = useRef(0);

  useEffect(() => {
    const stopAll = () => {
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
      if (sendWsRef.current) {
        try { sendWsRef.current.close(); } catch {}
        sendWsRef.current = null;
      }
      if (recvWsRef.current) {
        try { recvWsRef.current.close(); } catch {}
        recvWsRef.current = null;
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

    const sendWsUrl = toWsUrl(sendBaseUrl);
    const recvWsUrl = toWsUrl(receiveBaseUrl);
    const sendWs = new WebSocket(sendWsUrl);
    const recvWs = new WebSocket(recvWsUrl);
    sendWs.binaryType = 'arraybuffer';
    recvWs.binaryType = 'arraybuffer';
    sendWsRef.current = sendWs;
    recvWsRef.current = recvWs;
    setStatus('连接中');
    log('info', `正在连接视频发送通道: ${sendWsUrl}`);
    log('info', `正在连接视频接收通道: ${recvWsUrl}`);
    let sendReady = false;
    let recvReady = false;
    const updateStatus = () => {
      if (sendReady && recvReady) setStatus('已连接');
      else setStatus('连接中');
    };

    const handleVideoFrame = async (ev: MessageEvent) => {
      const buf = ev.data as ArrayBuffer;
      if (!remoteCanvasRef?.current || !buf) return;
      recvCountRef.current += 1;
      try {
        const Decoder = (window as any).ImageDecoder;
        if (Decoder) {
          const decoder = new Decoder({ data: buf, type: 'image/jpeg' });
          const { image } = await decoder.decode();
          const c = remoteCanvasRef.current;
          const w = (image as any).displayWidth ?? image.width;
          const h = (image as any).displayHeight ?? image.height;
          if (c.width !== w || c.height !== h) {
            c.width = w;
            c.height = h;
          }
          const ctx = c.getContext('2d');
          if (ctx) {
            ctx.drawImage(image, 0, 0);
          }
          image.close();
          decoder.close();
          return;
        }
        const blob = new Blob([buf], { type: 'image/jpeg' });
        const bmp = await createImageBitmap(blob);
        const c = remoteCanvasRef.current;
        if (c.width !== bmp.width || c.height !== bmp.height) {
          c.width = bmp.width;
          c.height = bmp.height;
        }
        const ctx = c.getContext('2d');
        if (ctx) {
          ctx.drawImage(bmp, 0, 0);
        }
        bmp.close();
      } catch {}
    };

    sendWs.onopen = () => {
      sendReady = true;
      updateStatus();
      log('info', '视频发送通道已连接，开始推流（JPEG帧）');
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
      let lastSentAt = 0;
      const canUseEncoder = typeof (window as any).ImageEncoder !== 'undefined' && typeof (window as any).VideoFrame !== 'undefined';
      const sendFrame = async () => {
        if (!sendWsRef.current || sendWsRef.current.readyState !== WebSocket.OPEN) return;
        const w = v.videoWidth; const h = v.videoHeight;
        if (w === 0 || h === 0) return;
        oc.width = w; oc.height = h;
        const ctx = oc.getContext('2d');
        if (!ctx) return;
        ctx.drawImage(v, 0, 0, w, h);
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
              sendWsRef.current?.send(buf);
              sentCountRef.current += 1;
            }
          } catch {}
        } else {
          await new Promise<void>(resolve => oc.toBlob(async (blob) => {
            if (!blob) return resolve();
            try {
              const buf = await blob.arrayBuffer();
              sendWsRef.current?.send(buf);
              sentCountRef.current += 1;
            } catch {}
            resolve();
          }, 'image/jpeg', 0.7));
        }
      };
      if (v.requestVideoFrameCallback) {
        const onFrame = async () => {
          if (!streamActiveRef.current) return;
          const now = performance.now();
          if (now - lastSentAt >= targetInterval) {
            await sendFrame();
            lastSentAt = now;
          }
          frameCallbackHandleRef.current = v.requestVideoFrameCallback(onFrame);
        };
        frameCallbackHandleRef.current = v.requestVideoFrameCallback(onFrame);
      } else {
        drawTimerRef.current = window.setInterval(sendFrame, targetInterval);
      }
    };

    sendWs.onmessage = handleVideoFrame;
    recvWs.onmessage = handleVideoFrame;

    sendWs.onclose = () => {
      sendReady = false;
      log('warn', '视频发送通道连接已关闭');
      stopAll();
    };
    sendWs.onerror = () => {
      log('error', '视频发送通道连接错误');
      stopAll();
    };
    recvWs.onopen = () => {
      recvReady = true;
      updateStatus();
      log('info', '视频接收通道已连接');
    };
    recvWs.onclose = () => {
      recvReady = false;
      log('warn', '视频接收通道连接已关闭');
      stopAll();
    };
    recvWs.onerror = () => {
      log('error', '视频接收通道连接错误');
      stopAll();
    };

    return stopAll;
  }, [cameraStream, isStreaming, sendBaseUrl, receiveBaseUrl, log, remoteCanvasRef]);

  return { streamingStatus: status };
};
