
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Sidebar } from './components/Sidebar';
import { MainDisplay } from './components/MainDisplay';
import { Viewer } from './Viewer';
import { AppState, LogEntry, LogLevel, OperationMode, SystemConfig, ThumbnailItem } from './types';
import { Menu } from 'lucide-react';
import { resolveAssetUrl } from './utils/onnxUtils';
import { useVideoStreaming } from './hooks/useVideoStreaming';

const App: React.FC = () => {
  const queryParams = new URLSearchParams(window.location.search);
  const isViewerMode = queryParams.get('view') === 'true';

  const [config, setConfig] = useState<SystemConfig | null>(null);
  const [sidebarWidth, setSidebarWidth] = useState(340);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(window.innerWidth < 1024);
  const [isResizing, setIsResizing] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  
  const [state, setState] = useState<AppState>({
    mode: OperationMode.FACE_SWAP,
    similarity: 75,
    sharpness: 50,
    isProcessing: false,
    selectedPortraitId: null,
    selectedBackgroundId: null,
    portraits: [],
    models: [],
    scenes: [],
    isCameraOn: false,
    isAutoMatting: false,
    serverStatus: 'idle',
    latency: null,
    isConfigOpen: false
  });

  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const remoteCanvasRef = useRef<HTMLCanvasElement>(null);
  const isCameraTogglingRef = useRef(false);
  const cameraStreamRef = useRef<MediaStream | null>(null);
  const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  
  // 前端不再负责抠像，改为推流
  const serverStatusRef = useRef(state.serverStatus);
  const isCameraOnRef = useRef(state.isCameraOn);
  const lastFaceNameRef = useRef<string | null>(null);
  const lastSceneNameRef = useRef<string | null>(null);
  const lastDfmNameRef = useRef<string | null>(null);

  useEffect(() => {
    cameraStreamRef.current = cameraStream;
  }, [cameraStream]);

  useEffect(() => {
    isCameraOnRef.current = state.isCameraOn;
  }, [state.isCameraOn]);

  useEffect(() => {
    serverStatusRef.current = state.serverStatus;
  }, [state.serverStatus]);

  if (isViewerMode) {
    return <Viewer />;
  }

  const log = useCallback((level: LogLevel, message: string) => {
    const time = new Date().toLocaleTimeString('zh-CN', { hour12: false });
    const entry: LogEntry = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      time,
      level,
      message
    };
    setLogs(prev => [...prev.slice(-199), entry]);
    if (level === 'error') {
      console.error(message);
    } else if (level === 'warn') {
      console.warn(message);
    } else {
      console.log(message);
    }
  }, []);

  const ensureWsPath = (url: string) => {
    if (!url) return '';
    const cleanUrl = url.replace(/\/$/, '');
    return cleanUrl.endsWith('/ws') ? cleanUrl : `${cleanUrl}/ws`;
  };

  const ensureVideoInPath = (url: string) => {
    if (!url) return '';
    const cleanUrl = url.replace(/\/$/, '');
    return cleanUrl.endsWith('/video_in') ? cleanUrl : `${cleanUrl}/video_in`;
  };

  const { streamingStatus } = useVideoStreaming(
    cameraStream,
    !isViewerMode && state.isCameraOn,
    ensureVideoInPath(config?.stream_send_url || '/video_in'),
    ensureVideoInPath(config?.stream_recv_url || '/video_in'),
    log,
    remoteCanvasRef
  );

  const getAvailableDevices = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = devices.filter(device => device.kind === 'videoinput');
      setVideoDevices(videoInputs);
      log('info', `检测到摄像头数量: ${videoInputs.length}`);
    } catch (err) {
      console.error("Error enumerating devices:", err);
      log('error', '摄像头列表获取失败');
    }
  }, [log]);

  useEffect(() => {
    if (isViewerMode) return;
    navigator.mediaDevices.addEventListener('devicechange', getAvailableDevices);
    getAvailableDevices();
    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', getAvailableDevices);
    };
  }, [getAvailableDevices, isViewerMode]);

  useEffect(() => {
    if (!isViewerMode && config) {
      const activeList = state.mode === OperationMode.FACE_SWAP ? state.portraits : state.models;
      const selectedItem = activeList.find(p => p.id === state.selectedPortraitId);
      const selectedScene = state.scenes.find(bg => bg.id === state.selectedBackgroundId);
      
      const syncData = {
        isProcessing: state.isProcessing,
        showBackendStream: state.isAutoMatting,
        portraitUrl: selectedItem?.url || '',
        backgroundUrl: resolveAssetUrl(selectedScene?.url || ''),
        logoUrl: resolveAssetUrl(config.system.logoUrl || ''),
        mode: state.mode
      };
      localStorage.setItem('cmai_viewport_sync', JSON.stringify(syncData));
    }
  }, [state.isProcessing, state.selectedPortraitId, state.selectedBackgroundId, state.portraits, state.models, state.mode, config, isViewerMode]);

  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      if (mobile) setIsSidebarCollapsed(true);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const getFullUrl = (baseUrl: string, path: string) => {
    const base = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
    const cleanPath = path.replace(/^public\//, '').replace(/^\//, '');
    return `${base}${cleanPath}`;
  };

  const setupCamera = useCallback(async (deviceId?: string) => {
    if (isCameraTogglingRef.current) return;
    isCameraTogglingRef.current = true;
    log('info', deviceId ? `切换摄像头: ${deviceId}` : '初始化摄像头');
    
    try {
      if (cameraStreamRef.current) {
        cameraStreamRef.current.getTracks().forEach(track => track.stop());
      }

      const constraints: MediaStreamConstraints = {
        video: { 
          width: 1280, 
          height: 720,
          deviceId: deviceId ? { exact: deviceId } : undefined
        }, 
        audio: false 
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      setCameraStream(stream);
      setState(prev => ({ ...prev, isCameraOn: true, isAutoMatting: true }));
      log('info', '摄像头已启动');

      if (deviceId) {
        setSelectedDeviceId(deviceId);
      } else {
        const videoTrack = stream.getVideoTracks()[0];
        if (videoTrack) {
          const settings = videoTrack.getSettings();
          if (settings.deviceId) {
            setSelectedDeviceId(settings.deviceId);
          }
        }
      }
      
      getAvailableDevices();
    } catch (err) {
      console.error("无法访问摄像头:", err);
      setState(prev => ({ ...prev, isCameraOn: false }));
      log('error', '摄像头启动失败');
    } finally {
      isCameraTogglingRef.current = false;
    }
  }, [getAvailableDevices, log]);

  const handleDeviceChange = (deviceId: string) => {
    setSelectedDeviceId(deviceId);
    setupCamera(deviceId);
    log('info', `用户选择摄像头: ${deviceId}`);
  };

  const handleToggleCamera = () => {
    if (isCameraTogglingRef.current) return;

    if (state.isCameraOn) {
      if (state.isProcessing) {
        void sendAiProcess({ enable: false, mode: state.mode });
      }
      if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
      }
      setCameraStream(null);
      setState(prev => ({ ...prev, isCameraOn: false, isAutoMatting: false, isProcessing: false }));
      log('info', '摄像头已关闭');
    } else {
      setupCamera(selectedDeviceId || undefined);
      log('info', '摄像头开启请求发送');
    }
  };

  const handleToggleMatting = () => {
    setState(prev => {
      const next = !prev.isAutoMatting;
      log('info', next ? '后端画面显示已开启' : '后端画面显示已关闭');
      return { ...prev, isAutoMatting: next };
    });
  };

  const checkServerStatus = useCallback(async () => {
    if (!config) return;
    
    const start = performance.now();
    try {
      await fetch(config.blank_url, { method: 'HEAD', mode: 'no-cors' });
      const end = performance.now();
      const latency = Math.round(end - start);
      const nextStatus = 'success' as const;
      if (serverStatusRef.current !== nextStatus) {
        log('info', `服务状态已连接，延迟 ${latency}ms`);
      }
      setState(prev => ({ 
        ...prev, 
        serverStatus: nextStatus,
        latency: latency
      }));
    } catch (err) {
      const nextStatus = 'error' as const;
      if (serverStatusRef.current !== nextStatus) {
        log('warn', '服务状态检查失败');
      }
      setState(prev => ({ 
        ...prev, 
        serverStatus: nextStatus,
        latency: null 
      }));
    }
  }, [config, log]);

  useEffect(() => {
    if (isViewerMode) return;
    if (!config) return;
    checkServerStatus();
    const interval = setInterval(checkServerStatus, 5000);
    return () => clearInterval(interval);
  }, [config, checkServerStatus, log, isViewerMode]);

  const handleTestServer = () => {
    setState(prev => ({ ...prev, serverStatus: 'testing' }));
    log('info', '触发服务状态检测');
    checkServerStatus();
  };

  const startResizing = useCallback(() => !isMobile && setIsResizing(true), [isMobile]);
  const stopResizing = useCallback(() => setIsResizing(false), []);
  const resize = useCallback((e: MouseEvent) => {
    if (isResizing && !isMobile) {
      const newWidth = e.clientX;
      if (newWidth > 280 && newWidth < 500) setSidebarWidth(newWidth);
    }
  }, [isResizing, isMobile]);

  useEffect(() => {
    window.addEventListener('mousemove', resize);
    window.addEventListener('mouseup', stopResizing);
    return () => {
      window.removeEventListener('mousemove', resize);
      window.removeEventListener('mouseup', stopResizing);
    };
  }, [resize, stopResizing]);

  const fetchAssetList = async (url: string, prefix: string, currentConfig: SystemConfig): Promise<ThumbnailItem[]> => {
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Failed to fetch from ${url}`);
      const data = await response.json();
      if (Array.isArray(data)) {
        return data.map((item, i) => ({
          id: `${prefix}-${i}-${Date.now()}`,
          name: typeof item === 'string' ? item.split('/').pop()?.split('.')[0] || `${prefix} ${i + 1}` : item.name,
          url: typeof item === 'string' ? (item.startsWith('http') ? item : getFullUrl(currentConfig.blank_url, item)) : item.url
        }));
      }
      return [];
    } catch (err) { return []; }
  };

  const loadAssetsFromGlob = (modules: Record<string, any>, prefix: string): ThumbnailItem[] => {
    return Object.keys(modules).map((path, index) => {
      const url = modules[path].default;
      const name = path.split('/').pop()?.split('.')[0] || `${prefix} ${index + 1}`;
      return {
        id: `${prefix}-${index}-${name}`,
        name: name,
        url: url
      };
    });
  };

  const refreshPortraits = async (currentConfig: SystemConfig) => {
    let list: ThumbnailItem[] = [];
    if (currentConfig.Is_from_blank_get_config) {
      list = await fetchAssetList(getFullUrl(currentConfig.blank_url, currentConfig.directories_blank.portraitWorkList), 'portrait', currentConfig);
    } else {
      const portraitModules = import.meta.glob('./src/assets/human_pic/*.{png,jpg,jpeg,webp}', { eager: true });
      list = loadAssetsFromGlob(portraitModules, 'portrait');
    }
    
    setState(prev => ({ ...prev, portraits: list }));
    if (state.mode === OperationMode.FACE_SWAP && list.length > 0 && !state.selectedPortraitId) {
      setState(prev => ({ ...prev, selectedPortraitId: list[0].id }));
    }
  };

  const refreshModels = async (currentConfig: SystemConfig) => {
    let list: ThumbnailItem[] = [];
    if (currentConfig.Is_from_blank_get_config) {
      list = await fetchAssetList(getFullUrl(currentConfig.blank_url, currentConfig.directories_blank.portraitDfmList), 'model', currentConfig);
    } else {
      const modelModules = import.meta.glob('./src/assets/DFM/*.{png,jpg,jpeg,webp}', { eager: true });
      list = loadAssetsFromGlob(modelModules, 'model');
    }

    setState(prev => ({ ...prev, models: list }));
    if (state.mode === OperationMode.MODEL_SWAP && list.length > 0 && !state.selectedPortraitId) {
      setState(prev => ({ ...prev, selectedPortraitId: list[0].id }));
    }
  };

  const refreshScenes = async (currentConfig: SystemConfig) => {
    let list: ThumbnailItem[] = [];
    if (currentConfig.Is_from_blank_get_config) {
      list = await fetchAssetList(getFullUrl(currentConfig.blank_url, currentConfig.directories_blank.sceneList), 'scene', currentConfig);
    } else {
      const sceneModules = import.meta.glob('./src/assets/m_blank/*.{png,jpg,jpeg,webp}', { eager: true });
      list = loadAssetsFromGlob(sceneModules, 'scene');
    }

    const finalScenes = [...list];
    setState(prev => ({ ...prev, scenes: finalScenes }));
  };

  useEffect(() => {
    const initApp = async () => {
      if (isViewerMode) return;
      try {
        const configRes = await fetch(`./systemConfig.json?t=${Date.now()}`);
        const data: SystemConfig = await configRes.json();
        setConfig(data);
        log('info', '系统配置加载完成');
        await refreshPortraits(data);
        log('info', '人像资源加载完成');
        await refreshModels(data);
        log('info', '模型资源加载完成');
        await refreshScenes(data);
        log('info', '场景资源加载完成');
      } catch (err) { console.error("Initialization failed:", err); }
    };
    initApp();
  }, [setupCamera, log, isViewerMode]);

  const handleModeChange = (mode: OperationMode) => {
    setState(prev => {
      const activeList = mode === OperationMode.FACE_SWAP ? prev.portraits : prev.models;
      return { 
        ...prev, 
        mode,
        selectedPortraitId: activeList.length > 0 ? activeList[0].id : null
      };
    });
  };

  const buildApiUrl = useCallback((path: string) => {
    const base = (config?.blank_url || '').replace(/\/$/, '');
    const cleanPath = path.startsWith('/') ? path : `/${path}`;
    return `${base}${cleanPath}`;
  }, [config?.blank_url]);

  const sendAiProcess = useCallback(async (payload: any) => {
    if (!config?.blank_url) return null;
    try {
      const res = await fetch(buildApiUrl('/api/v1/ai/process'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e: any) {
      log('error', `AI process 请求失败: ${e?.message || e}`);
      return null;
    }
  }, [buildApiUrl, config?.blank_url, log]);

  const sendAiCommand = useCallback(async (payload: any) => {
    if (!config?.blank_url) return null;
    try {
      const res = await fetch(buildApiUrl('/api/v1/ai/command'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e: any) {
      log('error', `AI command 请求失败: ${e?.message || e}`);
      return null;
    }
  }, [buildApiUrl, config?.blank_url, log]);

  const waitForDfmReady = useCallback(async (expectedName: string, standbyWorkerId: number, timeoutMs: number = 60000) => {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      try {
        const res = await fetch(buildApiUrl('/api/v1/ai/status'));
        if (res.ok) {
          const data = await res.json();
          const workers = data?.status?.workers;
          const st = workers?.[standbyWorkerId];
          const loaded = st?.loaded_dfm;
          if (loaded && String(loaded) === String(expectedName)) {
            return true;
          }
        }
      } catch (e) {}
      await new Promise(r => setTimeout(r, 200));
    }
    return false;
  }, [buildApiUrl]);

  useEffect(() => {
    if (isViewerMode) return;
    if (!state.isProcessing) {
      lastFaceNameRef.current = null;
      lastSceneNameRef.current = null;
      lastDfmNameRef.current = null;
      return;
    }

    const activeList = state.mode === OperationMode.FACE_SWAP ? state.portraits : state.models;
    const selectedItem = activeList.find(p => p.id === state.selectedPortraitId) || null;
    const selectedScene = state.scenes.find(bg => bg.id === state.selectedBackgroundId) || null;
    const faceName = selectedItem?.name ? String(selectedItem.name) : null;
    const sceneName = selectedScene?.name ? String(selectedScene.name) : null;

    if (sceneName && sceneName !== lastSceneNameRef.current) {
      lastSceneNameRef.current = sceneName;
      void sendAiCommand({ type: 'SET_BG', payload: sceneName });
    }

    if (!faceName) return;

    if (state.mode === OperationMode.FACE_SWAP) {
      if (faceName !== lastFaceNameRef.current) {
        lastFaceNameRef.current = faceName;
        void sendAiCommand({ type: 'SET_FACE', payload: faceName });
      }
      return;
    }

    if (faceName === lastDfmNameRef.current) return;
    lastDfmNameRef.current = faceName;
    void (async () => {
      const loadRes = await sendAiCommand({ type: 'LOAD_DFM', payload: faceName });
      const standbyId = loadRes?.standby_worker_id;
      if (typeof standbyId === 'number') {
        const ready = await waitForDfmReady(faceName, standbyId, 90000);
        if (ready) {
          await sendAiCommand({ type: 'ACTIVATE', worker_id: standbyId });
        }
      }
    })();
  }, [
    isViewerMode,
    sendAiCommand,
    state.isProcessing,
    state.mode,
    state.models,
    state.portraits,
    state.scenes,
    state.selectedBackgroundId,
    state.selectedPortraitId,
    waitForDfmReady
  ]);

  const handleToggleProcessing = useCallback(async () => {
    if (!config?.blank_url) {
      log('warn', '未加载后端地址配置，无法发送指令');
      return;
    }

    const next = !state.isProcessing;
    if (!next) {
      await sendAiProcess({ enable: false, mode: state.mode });
      setState(prev => ({ ...prev, isProcessing: false }));
      log('info', '已发送停止合成指令');
      return;
    }

    const activeList = state.mode === OperationMode.FACE_SWAP ? state.portraits : state.models;
    const selectedItem = activeList.find(p => p.id === state.selectedPortraitId) || null;
    const selectedScene = state.scenes.find(bg => bg.id === state.selectedBackgroundId) || null;

    if (!selectedItem) {
      log('warn', state.mode === OperationMode.FACE_SWAP ? '未选择目标人像' : '未选择 DFM 模型');
      return;
    }
    if (!selectedScene) {
      log('warn', '未选择场景图片');
      return;
    }

    const modeParam = state.mode === OperationMode.FACE_SWAP ? 'FACE' : 'DFM';
    const param1 = modeParam;
    const param2 = String(selectedItem.name || '');
    const param3 = String(selectedScene.name || '');

    log('info', `发送切换指令: [${param1}, ${param2}, ${param3}]`);

    if (state.mode === OperationMode.FACE_SWAP) {
      await sendAiCommand({ type: 'UNLOAD_MODEL' });
      await sendAiProcess({
        enable: true,
        mode: modeParam,
        portrait_id: param2,
        scene_id: param3,
        similarity: state.similarity,
        sharpness: state.sharpness
      });
      setState(prev => ({ ...prev, isProcessing: true }));
      return;
    }

    await sendAiProcess({
      enable: true,
      mode: modeParam,
      portrait_id: null,
      scene_id: param3,
      similarity: state.similarity,
      sharpness: state.sharpness
    });

    const loadRes = await sendAiCommand({ type: 'LOAD_DFM', payload: param2 });
    const standbyId = loadRes?.standby_worker_id;
    if (typeof standbyId === 'number') {
      const ready = await waitForDfmReady(param2, standbyId, 90000);
      if (ready) {
        await sendAiCommand({ type: 'ACTIVATE', worker_id: standbyId });
        setState(prev => ({ ...prev, isProcessing: true }));
        return;
      }
      log('warn', 'DFM 预加载超时，未执行主备切换');
      setState(prev => ({ ...prev, isProcessing: true }));
      return;
    }

    log('warn', '未获得 standby worker id，跳过主备切换');
    setState(prev => ({ ...prev, isProcessing: true }));
  }, [config?.blank_url, log, sendAiCommand, sendAiProcess, state.isProcessing, state.mode, state.models, state.portraits, state.scenes, state.selectedBackgroundId, state.selectedPortraitId, state.sharpness, state.similarity, waitForDfmReady]);

  if (isViewerMode) return <Viewer />;

  if (!config) return <div className="h-screen w-full flex items-center justify-center text-white/50 font-black uppercase tracking-[0.5em] bg-[#050505]">Initializing CMAI Engine...</div>;

  const bgUrl = resolveAssetUrl(config.system.backgroundUrl);

  return (
    <div className={`relative flex h-screen w-full overflow-hidden transition-all duration-500 ${isResizing ? 'cursor-col-resize select-none' : ''}`}>
      {bgUrl && (
        <div 
          className="fixed inset-0 z-[-2] pointer-events-none transition-opacity duration-1000 bg-black"
          style={{
            backgroundImage: `url('${bgUrl}')`,
            backgroundSize: 'cover',
            backgroundPosition: 'center',
            backgroundRepeat: 'no-repeat'
          }}
        />
      )}
      <div className="fixed inset-0 z-[-1] pointer-events-none bg-black/10 backdrop-blur-[2px]" />
      
      <div 
        style={{ width: isSidebarCollapsed ? '0px' : (isMobile ? '100%' : `${sidebarWidth}px`) }} 
        className={`h-full overflow-hidden transition-all duration-500 ease-in-out relative shrink-0 z-[200] ${isMobile ? 'fixed inset-0 bg-black/80 backdrop-blur-xl' : ''}`}
      >
        <Sidebar 
          config={config} state={state} width={isMobile ? window.innerWidth * 0.85 : sidebarWidth}
          resolveAssetUrl={resolveAssetUrl}
          onModeChange={handleModeChange}
          onSliderChange={(k, v) => setState(prev => ({ ...prev, [k]: v }))}
          onToggleProcessing={() => { void handleToggleProcessing(); }}
          onSwitch={() => {
             const activeList = state.mode === OperationMode.FACE_SWAP ? state.portraits : state.models;
             const idx = activeList.findIndex(p => p.id === state.selectedPortraitId);
             if (idx === -1) return;
             const nextIdx = (idx + 1) % activeList.length;
             setState(prev => ({ ...prev, selectedPortraitId: activeList[nextIdx].id }));
          }}
          onSelectPortrait={(id) => setState(prev => ({ ...prev, selectedPortraitId: id }))}
          onSelectBackground={(id) => setState(prev => ({ ...prev, selectedBackgroundId: id }))}
          onToggleCollapse={() => setIsSidebarCollapsed(true)}
          onRefreshPortraits={() => {
            refreshPortraits(config);
            refreshModels(config);
          }}
          onRefreshScenes={() => refreshScenes(config)}
        />
        {isMobile && !isSidebarCollapsed && (
          <div className="absolute inset-0 bg-transparent" onClick={() => setIsSidebarCollapsed(true)} />
        )}
      </div>

      {isSidebarCollapsed && (
        <button 
          onClick={() => setIsSidebarCollapsed(false)} 
          className={`absolute ${isMobile ? 'top-6 left-6' : 'left-0 top-1/2 -translate-y-1/2'} ${isMobile ? 'w-10 h-10 rounded-xl glass-card' : 'w-3 h-24 rounded-r-2xl glass-panel border-l-0'} z-[100] flex items-center justify-center transition-all duration-500 group border border-white/5 shadow-2xl hover:bg-blue-500/20`}
        >
          {isMobile ? (
            <Menu className="text-white/60" size={20} />
          ) : (
            <div className="w-0.5 h-10 rounded-full bg-white/10 group-hover:bg-blue-400 transition-colors" />
          )}
        </button>
      )}

      {!isSidebarCollapsed && !isMobile && (
        <div onMouseDown={startResizing} className={`w-1 h-full relative z-[100] cursor-col-resize group transition-colors duration-300 ${isResizing ? 'bg-blue-500/30' : 'hover:bg-blue-500/10'}`}>
          <div className={`absolute inset-y-0 left-1/2 -translate-x-1/2 w-[1px] bg-white/5 group-hover:bg-blue-400/50 transition-colors ${isResizing ? 'bg-blue-400/50' : ''}`} />
        </div>
      )}

      <MainDisplay 
        config={config} state={state} videoRef={videoRef}
        cameraStream={cameraStream}
        remoteCanvasRef={remoteCanvasRef}
        resolveAssetUrl={resolveAssetUrl}
        onToggleCamera={handleToggleCamera}
        onToggleMatting={handleToggleMatting}
        onTestServer={handleTestServer}
        onToggleConfig={() => setState(prev => ({ ...prev, isConfigOpen: !prev.isConfigOpen }))}
        videoDevices={videoDevices}
        selectedDeviceId={selectedDeviceId}
        onDeviceChange={handleDeviceChange}
        isStreaming={state.isCameraOn}
        streamingStatus={streamingStatus}
        logs={logs}
      />
    </div>
  );
};

export default App;
