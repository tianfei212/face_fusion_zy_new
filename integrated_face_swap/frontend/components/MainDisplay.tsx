
import React, { useMemo, useState, useRef, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { 
  Minus, 
  Move, 
  Video, 
  VideoOff, 
  Network, 
  Layers,
  Settings,
  ExternalLink,
  Scan,
  ShieldCheck,
  Zap,
  Camera
} from 'lucide-react';
import { AppState, LogEntry, OperationMode, SystemConfig } from '../types';
import { ConfigOverlay } from './ConfigOverlay';

interface MainDisplayProps {
  config: SystemConfig;
  state: AppState;
  videoRef: React.RefObject<HTMLVideoElement>;
  cameraStream: MediaStream | null;
  remoteCanvasRef: React.RefObject<HTMLCanvasElement>;
  resolveAssetUrl: (url: string) => string;
  onToggleCamera: () => void;
  onToggleMatting: () => void;
  onTestServer: () => void;
  onToggleConfig: () => void;
  videoDevices?: MediaDeviceInfo[];
  selectedDeviceId?: string | null;
  onDeviceChange?: (deviceId: string) => void;
  isStreaming: boolean;
  streamingStatus: string;
  logs: LogEntry[];
}

export const MainDisplay: React.FC<MainDisplayProps> = ({ 
  config, 
  state, 
  videoRef, 
  cameraStream,
  remoteCanvasRef,
  resolveAssetUrl,
  onToggleCamera, 
  onToggleMatting, 
  onTestServer,
  onToggleConfig,
  videoDevices = [],
  selectedDeviceId = null,
  onDeviceChange = (_deviceId: string) => {},
  isStreaming,
  streamingStatus,
  logs
}) => {
  const [isMinimized, setIsMinimized] = useState(window.innerWidth < 768);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragStartPos = useRef({ x: 0, y: 0 });
  const [logoError, setLogoError] = useState(false);
  const [showCameraMenu, setShowCameraMenu] = useState(false);
  const cameraButtonRef = useRef<HTMLButtonElement>(null);
  const [menuPos, setMenuPos] = useState({ bottom: 0, left: 0 });

  useEffect(() => {
    if (showCameraMenu && cameraButtonRef.current) {
      const rect = cameraButtonRef.current.getBoundingClientRect();
      setMenuPos({
        bottom: window.innerHeight - rect.top + 10,
        left: rect.left + rect.width / 2
      });
    }
  }, [showCameraMenu]);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (showCameraMenu && cameraButtonRef.current && !cameraButtonRef.current.contains(e.target as Node)) {
        setShowCameraMenu(false);
      }
    };
    window.addEventListener('mousedown', handleClickOutside);
    return () => window.removeEventListener('mousedown', handleClickOutside);
  }, [showCameraMenu]);

  const isFaceSwap = state.mode === OperationMode.FACE_SWAP;
  const showCameraLayer = state.isCameraOn;

  const selectedPortrait = useMemo(() => {
    const list = isFaceSwap ? state.portraits : state.models;
    return list.find(p => p.id === state.selectedPortraitId);
  }, [state.selectedPortraitId, state.portraits, state.models, isFaceSwap]);

  const selectedScene = useMemo(() => 
    state.scenes.find(bg => bg.id === state.selectedBackgroundId), 
    [state.selectedBackgroundId, state.scenes]
  );

  const showSceneLayer = selectedScene && selectedScene.id !== 'bg-blank';
  const isMobile = window.innerWidth < 768;

  const rawVideoLocalRef = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    if (rawVideoLocalRef.current && cameraStream) {
      rawVideoLocalRef.current.srcObject = cameraStream;
    }
  }, [cameraStream]);

  const handleOpenInNewPage = () => {
    const currentUrl = new URL(window.location.href);
    currentUrl.searchParams.set('view', 'true');
    window.open(currentUrl.toString(), '_blank');
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (isMinimized || isMobile) return;
    setIsDragging(true);
    dragStartPos.current = {
      x: e.clientX - position.x,
      y: e.clientY - position.y
    };
  };

  useEffect(() => {
    if (config?.system?.standbyImageUrl) {
        setLogoError(false); // Reset error state when config changes
    }
  }, [config?.system?.standbyImageUrl]);

  useEffect(() => {
    const handleGlobalMouseMove = (e: MouseEvent) => {
      if (isDragging && !isMobile) {
        setPosition({
          x: e.clientX - dragStartPos.current.x,
          y: e.clientY - dragStartPos.current.y
        });
      }
    };
    const handleGlobalMouseUp = () => setIsDragging(false);

    if (isDragging) {
      window.addEventListener('mousemove', handleGlobalMouseMove);
      window.addEventListener('mouseup', handleGlobalMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleGlobalMouseMove);
      window.removeEventListener('mouseup', handleGlobalMouseUp);
    };
  }, [isDragging, isMobile]);

  useEffect(() => {
    if (videoRef.current && cameraStream) {
      videoRef.current.srcObject = cameraStream;
    }
  }, [cameraStream, state.isCameraOn]);

  const isSecure = window.location.protocol === 'https:';
  const isOnline = state.serverStatus === 'success';

  return (
    <main className="flex-1 relative p-4 md:p-10 flex flex-col bg-transparent overflow-hidden">
      {/* 视口主容器 */}
      <div className="flex-1 relative rounded-[32px] md:rounded-[48px] overflow-hidden glass-card shadow-2xl border border-white/20 flex flex-col transition-all duration-700 mb-4 md:mb-6 bg-transparent">
        
        {/* 底层场景图同步 */}
        {showSceneLayer && (
          <img 
            src={resolveAssetUrl(selectedScene?.url)} 
            className="absolute inset-0 w-full h-full object-cover blur-[15px] opacity-40 transition-opacity duration-1000 z-0 scale-105" 
            alt="Scene Layer" 
          />
        )}

        {/* 顶部功能按键 */}
        <div className="absolute top-6 right-6 md:top-10 md:right-10 z-30 flex space-x-4">
          <button 
            onClick={handleOpenInNewPage}
            className="w-10 h-10 md:w-12 md:h-12 rounded-full glass-panel flex items-center justify-center text-white/60 hover:text-white transition-all hover:scale-110 border border-white/10 shadow-2xl"
          >
            <ExternalLink size={20} />
          </button>
        </div>

        {/* 核心内容区 */}
        <div className="absolute inset-0 z-10 flex items-center justify-center">
          {state.isProcessing ? (
             <div className="relative w-full h-full flex items-center justify-center overflow-hidden">
                <img 
                  src={resolveAssetUrl(selectedPortrait?.url || '')} 
                  className="w-full h-full object-cover transition-all duration-700 brightness-110 scale-100" 
                  alt="AI Synthesis Result" 
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-black/20" />
                <div className="absolute top-0 left-0 w-full h-[3px] bg-blue-500 shadow-[0_0_40px_rgba(0,122,255,1)] animate-scan z-20" />
                <div className="absolute bottom-10 left-10 md:bottom-14 md:left-14 text-left">
                    <div className="text-blue-400 text-[9px] font-black uppercase tracking-[0.4em] mb-1">Target Engine Locked</div>
                    <div className="text-white text-xl md:text-3xl font-bold tracking-tight drop-shadow-lg">{selectedPortrait?.name}</div>
                </div>
             </div>
          ) : (
            <div className="w-full h-full relative overflow-hidden flex items-center justify-center bg-transparent">
               {showCameraLayer ? (
                 state.isAutoMatting ? (
                   <canvas className="absolute inset-0 w-full h-full object-cover" ref={remoteCanvasRef} />
                 ) : (
                   <video 
                     ref={videoRef}
                     autoPlay
                     playsInline
                     muted
                     className="absolute inset-0 w-full h-full object-cover" />
                 )
               ) : config.system.standbyImageUrl ? (
                 <img 
                  key={config.system.standbyImageUrl}
                  src={resolveAssetUrl(config.system.standbyImageUrl)} 
                  alt="Standby Screen" 
                  className="w-full h-full object-cover opacity-80"
                  onError={(e) => {
                    console.error('Standby image load failed:', config.system.standbyImageUrl);
                    e.currentTarget.style.display = 'none';
                  }}
                 />
               ) : config.system.placeholderUrl && !logoError ? (
                 <img 
                  src={resolveAssetUrl(config.system.placeholderUrl)} 
                  alt="Placeholder Logo" 
                  onError={() => setLogoError(true)}
                  className="w-1/3 md:w-1/4 max-h-[50%] object-contain opacity-60 hover:opacity-100 transition-opacity duration-1000 scale-100" 
                 />
               ) : (
                 <div className="text-white/10 text-[10px] md:text-[14px] font-black uppercase tracking-[1em] animate-pulse">CMAI STANDBY ENGINE</div>
               )}
            </div>
          )}
        </div>

        {/* 左上角 Live 状态 */}
        <div className="absolute top-6 left-6 md:top-10 md:left-10 z-30">
           <div className="flex items-center space-x-3 glass-panel px-4 py-2.5 rounded-2xl border border-white/10 shadow-2xl">
              <div className={`w-2 h-2 rounded-full ${state.isProcessing ? 'bg-red-500 animate-pulse shadow-[0_0_10px_rgba(239,68,68,0.8)]' : 'bg-green-500 shadow-[0_0_10px_rgba(34,197,94,0.8)]'}`} />
              <span className="text-[10px] font-black text-white/80 uppercase tracking-[0.2em]">Live System</span>
           </div>
        </div>

        <div className="absolute top-20 left-6 md:top-24 md:left-10 z-30">
          <div className="glass-panel px-4 py-3 rounded-2xl border border-white/10 shadow-2xl min-w-[220px]">
            <div className="text-[9px] font-black text-white/50 uppercase tracking-[0.2em] mb-2">传输状态</div>
            <div className="text-[10px] font-bold text-white/80">状态: {streamingStatus}</div>
            <div className="text-[10px] font-bold text-white/80">
              当前模式: <span className={isStreaming ? 'text-green-400' : 'text-yellow-400'}>{isStreaming ? '推流到后端' : '本地预览'}</span>
            </div>
          </div>
        </div>

        <div className="absolute top-20 right-6 md:top-24 md:right-10 z-30">
          <div className="glass-panel px-4 py-3 rounded-2xl border border-white/10 shadow-2xl w-[260px] max-h-[220px] overflow-hidden">
            <div className="text-[9px] font-black text-white/50 uppercase tracking-[0.2em] mb-2">运行日志</div>
            <div className="space-y-1 overflow-y-auto max-h-[180px] pr-1">
              {logs.length === 0 ? (
                <div className="text-[9px] font-bold text-white/40">暂无日志</div>
              ) : (
                logs.slice(-12).map(entry => (
                  <div key={entry.id} className="flex items-center gap-2 text-[9px] font-mono text-white/70">
                    <span className="text-white/40">{entry.time}</span>
                    <span className={entry.level === 'error' ? 'text-red-400' : entry.level === 'warn' ? 'text-yellow-400' : 'text-blue-300'}>
                      {entry.level.toUpperCase()}
                    </span>
                    <span className="truncate">{entry.message}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* 源监视窗口 */}
        <div 
          style={{ 
            transform: isMobile ? 'none' : `translate(${position.x}px, ${position.y}px)`,
            bottom: isMinimized ? '-500px' : (isMobile ? '20px' : '40px'),
            right: isMinimized ? '20px' : (isMobile ? '20px' : '40px'),
            opacity: isMinimized ? 0 : 1,
            pointerEvents: isMinimized ? 'none' : 'auto'
          }}
          className={`absolute w-[280px] md:w-[420px] aspect-video glass-panel rounded-3xl overflow-hidden border border-white/20 shadow-[0_30px_90px_rgba(0,0,0,0.9)] transition-all duration-500 ease-out flex flex-col z-[40] ${isDragging ? 'scale-[1.03] ring-2 ring-blue-500/30' : ''}`}
        >
          <div onMouseDown={handleMouseDown} className={`h-11 flex items-center justify-between px-5 ${isMobile ? '' : 'cursor-move'} bg-white/5`}>
             <div className="flex items-center space-x-2">
                {!isMobile && <Move size={14} className="text-white/20" />}
                <span className="text-[9px] font-black text-white/40 uppercase tracking-[0.2em]">Raw Media Feed</span>
             </div>
             <button onClick={(e) => { e.stopPropagation(); setIsMinimized(true); }} className="p-1.5 rounded-full hover:bg-white/10 text-white/40">
                <Minus size={18} />
             </button>
          </div>
          <div className="flex-1 relative bg-black">
            {state.isCameraOn ? (
              <video 
                ref={rawVideoLocalRef} 
                autoPlay 
                playsInline 
                muted 
                className="w-full h-full object-cover" 
              />
            ) : config.system.cameraOfflineUrl ? (
              <img 
                key={config.system.cameraOfflineUrl}
                src={resolveAssetUrl(config.system.cameraOfflineUrl)} 
                alt="Camera Offline" 
                className="w-full h-full object-cover"
                onError={(e) => {
                  console.error('Camera offline image load failed:', config.system.cameraOfflineUrl);
                  e.currentTarget.style.display = 'none';
                }}
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-white/5"><VideoOff size={32} /></div>
            )}
            {state.isAutoMatting && (
              <div className="absolute top-3 left-3 px-2 py-1 bg-blue-600/80 rounded-lg text-[8px] font-black text-white uppercase tracking-widest flex items-center space-x-2">
                <Scan size={10} /><span>Streaming Active</span>
              </div>
            )}
          </div>
        </div>

        <ConfigOverlay config={config} isOpen={state.isConfigOpen} onClose={onToggleConfig} />
      </div>

      {/* 底部信息条 */}
      <footer className="flex items-center justify-between h-14 md:h-16 px-4 md:px-6 overflow-x-auto scrollbar-hide select-none">
        <div className="flex items-center space-x-10 shrink-0">
           <div className="flex items-center space-x-2.5">
              <span className="text-[9px] font-black text-white/30 uppercase tracking-widest">特征点</span>
              <span className="text-[9px] font-black text-green-400">运行中</span>
           </div>
           <div className="flex items-center space-x-2.5">
              <span className="text-[9px] font-black text-white/30 uppercase tracking-widest">稳定性</span>
              <span className="text-[10px] font-mono font-bold text-blue-400">99.42%</span>
           </div>
           <div className="flex items-center space-x-2.5">
              <span className="text-[9px] font-black text-white/30 uppercase tracking-widest">模型拟合</span>
              <span className="text-[10px] font-mono font-bold text-blue-400">98.15%</span>
           </div>
        </div>

        <div className="flex items-center space-x-4 px-6">
           {videoDevices.length > 1 && (
             <div className="relative">
               {showCameraMenu && createPortal(
                 <div 
                   className="fixed z-[9999] w-48 bg-black/90 border border-white/10 rounded-xl overflow-hidden backdrop-blur-xl py-1 shadow-2xl"
                   style={{ 
                     bottom: menuPos.bottom, 
                     left: menuPos.left, 
                     transform: 'translateX(-50%)' 
                   }}
                 >
                   {videoDevices.map(device => (
                     <button
                       key={device.deviceId}
                       onClick={() => {
                         onDeviceChange(device.deviceId);
                         setShowCameraMenu(false);
                       }}
                       className={`w-full text-left px-4 py-3 text-xs font-medium hover:bg-white/10 transition-colors ${selectedDeviceId === device.deviceId ? 'text-blue-400' : 'text-white/80'}`}
                     >
                       {device.label || `Camera ${device.deviceId.slice(0, 5)}...`}
                     </button>
                   ))}
                 </div>,
                 document.body
               )}
               <button 
                 ref={cameraButtonRef}
                 onClick={() => setShowCameraMenu(!showCameraMenu)}
                 className={`w-11 h-11 rounded-2xl transition-all flex items-center justify-center border ${showCameraMenu ? 'text-blue-400 bg-blue-500/10 border-blue-500/30' : 'text-white/30 bg-white/5 border-white/5'}`}
                 title="选择摄像头"
               >
                 <Camera size={18} />
               </button>
             </div>
           )}
           <button onClick={onToggleCamera} className={`w-11 h-11 rounded-2xl transition-all flex items-center justify-center border ${state.isCameraOn ? 'text-blue-400 bg-blue-500/10 border-blue-500/30 shadow-[0_0_15px_rgba(0,122,255,0.2)]' : 'text-white/30 bg-white/5 border-white/5'}`}>
              <Video size={18} />
           </button>
           <button onClick={onToggleMatting} className={`w-11 h-11 rounded-2xl transition-all flex items-center justify-center border ${state.isAutoMatting ? 'text-blue-400 bg-blue-500/10 border-blue-500/30' : 'text-white/30 bg-white/5 border-white/5'}`}>
              <Layers size={18} />
           </button>
           <button onClick={onTestServer} className={`w-11 h-11 rounded-2xl transition-all flex items-center justify-center border ${state.serverStatus === 'success' ? 'text-blue-400 bg-blue-500/10 border-blue-500/30' : 'text-white/30 bg-white/5 border-white/5'}`}>
              <Network size={18} />
           </button>
           <button onClick={onToggleConfig} className={`w-11 h-11 rounded-2xl transition-all flex items-center justify-center border ${state.isConfigOpen ? 'text-blue-400 bg-blue-500/10 border-blue-500/30' : 'text-white/30 bg-white/5 border-white/5'}`}>
              <Settings size={18} />
           </button>
        </div>

        <div className="flex items-center space-x-4 shrink-0">
          <div className="flex items-center space-x-3 px-3 py-2 bg-white/5 rounded-xl border border-white/10">
            <ShieldCheck size={12} className={isSecure ? "text-green-500" : "text-red-500"} />
            <span className={`text-[9px] font-black tracking-widest uppercase ${isSecure ? "text-white/60" : "text-red-400"}`}>安全环境</span>
          </div>
          <div className="flex items-center space-x-3 px-3 py-2 bg-white/5 rounded-xl border border-white/10">
            <Zap size={12} className={isOnline ? "text-blue-400" : "text-red-500"} />
            <span className={`text-[9px] font-black tracking-widest uppercase ${isOnline ? "text-white/60" : "text-red-400"}`}>{isOnline ? "已联机" : "脱机"}</span>
          </div>
          <div className={`text-[10px] font-mono font-bold ${isOnline ? "text-green-400" : "text-white/30"}`}>
            {isOnline && state.latency !== null ? `${state.latency}ms` : "--"}
          </div>
        </div>
      </footer>

      <style>{`
        @keyframes scan { 
          0% { transform: translateY(-100%); opacity: 0; } 
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { transform: translateY(100vh); opacity: 0; } 
        }
        .animate-scan { animation: scan 6s cubic-bezier(0.4, 0, 0.2, 1) infinite; height: 100%; }
        .scrollbar-hide::-webkit-scrollbar { display: none; }
      `}</style>
    </main>
  );
};
