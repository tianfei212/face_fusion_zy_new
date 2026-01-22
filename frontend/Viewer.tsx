import React, { useEffect, useRef, useState } from 'react';
import { X } from 'lucide-react';

export const Viewer: React.FC = () => {
  const remoteCanvasRef = useRef<HTMLCanvasElement>(null);
  const showBackendStreamRef = useRef(true);
  
  // UI 状态（仅用于低频更新显示，对应您截图中的信息）
  const [status, setStatus] = useState('WAITING FOR SIGNAL...');
  const [debugInfo, setDebugInfo] = useState({
    channel: 'initializing',
    totalFrames: 0,
    lastPayload: 'n/a',
    lastFrameTime: 0,
  });

  // 性能计数器（使用 Ref，变化时不会触发 React 重新渲染，保证流畅度）
  const statsRef = useRef({
    total: 0,
    frameCount: 0,
    lastTime: Date.now(),
  });

  useEffect(() => {
    const readSync = () => {
      try {
        const raw = localStorage.getItem('cmai_viewport_sync');
        if (!raw) return;
        const parsed = JSON.parse(raw);
        showBackendStreamRef.current = parsed?.showBackendStream !== false;
      } catch {}
    };
    readSync();
    const onStorage = (e: StorageEvent) => {
      if (e.key === 'cmai_viewport_sync') readSync();
    };
    window.addEventListener('storage', onStorage);

    // 检查浏览器支持
    if (!('BroadcastChannel' in window)) {
      setDebugInfo(prev => ({ ...prev, channel: 'unsupported' }));
      window.removeEventListener('storage', onStorage);
      return;
    }

    // 1. 建立连接
    const bc = new BroadcastChannel('stream_sync_channel');
    setDebugInfo(prev => ({ ...prev, channel: 'listening' }));
    setStatus('LISTENING');

    // 2. 监听消息
    bc.onmessage = async (event) => {
      // 基础校验：必须匹配 { type: 'frame', payload: ... } 格式
      const expectedType = showBackendStreamRef.current ? 'frame' : 'raw_frame';
      if (!event?.data || event.data.type !== expectedType) return;
      
      const payload = event.data.payload;
      
      // 解析 Blob
      let blob: Blob | null = null;
      if (payload instanceof Blob) {
        blob = payload;
      } else if (payload instanceof ArrayBuffer) {
        blob = new Blob([payload], { type: 'image/jpeg' });
      }
      if (!blob) return;

      // 3. 绘图 (这是最高优先级任务，直接操作 DOM)
      try {
        const bitmap = await createImageBitmap(blob);
        const canvas = remoteCanvasRef.current;
        if (canvas) {
          // 只有尺寸变化时才赋值，减少闪烁
          if (canvas.width !== bitmap.width) canvas.width = bitmap.width;
          if (canvas.height !== bitmap.height) canvas.height = bitmap.height;
          
          const ctx = canvas.getContext('2d', { alpha: false }); // alpha: false 稍微提升性能
          ctx?.drawImage(bitmap, 0, 0);
          bitmap.close(); // 必须释放内存
        }
      } catch (e) {
        console.error('Frame render error', e);
      }

      // 4. 计数 (静默更新，不触发渲染)
      statsRef.current.total += 1;
      statsRef.current.frameCount += 1;
      const now = Date.now();

      // 5. 节流更新 UI (每 1000ms 更新一次 React 状态)
      // 这就是您截图中 "UPDATE: 1S INTERVAL" 的来源
      if (now - statsRef.current.lastTime > 1000) {
        const fps = statsRef.current.frameCount;
        
        // 批量更新所有状态，只触发一次重绘
        setStatus(`SYNCING: ${fps} FPS`);
        setDebugInfo(prev => ({
          ...prev,
          totalFrames: statsRef.current.total,
          lastFrameTime: now,
          lastPayload: payload?.constructor?.name || 'Blob'
        }));

        // 重置计数器
        statsRef.current.frameCount = 0;
        statsRef.current.lastTime = now;
      }
    };

    return () => {
      bc.close();
      window.removeEventListener('storage', onStorage);
    };
  }, []);

  const handleClose = () => {
    window.close();
  };

  return (
    <div className="fixed inset-0 bg-black z-[9999] flex flex-col overflow-hidden select-none">
      <div className="flex-1 relative w-full h-full overflow-hidden bg-black">
        
        {/* Remote Stream Canvas */}
        <div className="absolute inset-0 z-10 flex items-center justify-center">
          <canvas ref={remoteCanvasRef} className="w-full h-full" />
        </div>

        {/* Status Badge (对应您的截图样式) */}
        <div className="absolute top-10 left-10 z-50 pointer-events-none">
          <div className="px-3 py-2 rounded-xl glass-hud border border-white/10 text-[10px] font-black uppercase tracking-widest text-white/70 backdrop-blur-md bg-black/30">
            {status}
          </div>
          <div className="mt-2 px-3 py-2 rounded-xl glass-hud border border-white/10 text-[10px] font-black uppercase tracking-widest text-white/70 backdrop-blur-md bg-black/30">
            <div>channel: {debugInfo.channel}</div>
            <div>frames: {debugInfo.totalFrames}</div>
            <div>last payload: {debugInfo.lastPayload}</div>
            <div>update: 1s interval</div>
          </div>
        </div>
        
        {/* Close Button */}
        <div className="absolute top-10 right-10 z-50">
          <button 
            onClick={handleClose}
            className="w-14 h-14 rounded-full glass-card flex items-center justify-center text-white/40 hover:text-white transition-all hover:scale-110 active:scale-90 border border-white/10 shadow-2xl group bg-black/20"
          >
            <X size={24} className="group-hover:rotate-90 transition-transform duration-300" />
          </button>
        </div>
      </div>

      <style>{`
        .glass-hud {
            background: rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(25px);
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
        }
      `}</style>
    </div>
  );
};
