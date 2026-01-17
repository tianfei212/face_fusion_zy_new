
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Sidebar } from './components/Sidebar';
import { MainDisplay } from './components/MainDisplay';
import { Viewer } from './Viewer';
import { AppState, OperationMode, SystemConfig, ThumbnailItem } from './types';
import { ChevronRight, Menu } from 'lucide-react';

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
    isCameraOn: true,
    isAutoMatting: false,
    serverStatus: 'idle',
    isConfigOpen: false
  });

  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const portraitNames = ["张伟", "王芳", "李强", "刘敏", "陈静", "杨波", "赵磊", "黄欢", "周宇", "吴倩", "徐明", "孙悦", "朱军", "马莉", "高峰", "郭兰", "林森", "何琴", "罗勇", "梁红"];
  const modelNames = ["超写实01", "影视级02", "赛博仿真03", "古风拟真04", "商业模特05", "潮流博主06", "职场精英07", "运动达人08", "科技未来09", "梦幻少女10", "成熟绅士11", "活力少年12", "优雅名媛13", "硬朗战士14", "虚拟偶像15", "禅意大师16", "街头艺人17", "医疗助手18", "教育导师19", "艺术总监20"];
  const sceneStyles = ["紫禁城", "苏州园林", "赛博重庆", "外滩夜景", "敦煌石窟", "泰山日出", "杭州西湖", "张家界", "香格里拉", "现代影棚", "科幻实验室", "极简客厅", "欧式礼堂", "宇宙空间站", "赛车场", "歌剧院", "荒野草原", "热带雨林", "冰雪大世界", "霓虹街道"];

  const resolveAssetUrl = (url: string) => {
    if (!url) return '';
    if (url.startsWith('http') || url.startsWith('data:') || url.startsWith('blob:')) return url;
    const cleanUrl = url.replace(/^public\//, '').replace(/^\//, '');
    return `./${cleanUrl}`;
  };

  useEffect(() => {
    if (!isViewerMode && config) {
      const activeList = state.mode === OperationMode.FACE_SWAP ? state.portraits : state.models;
      const selectedItem = activeList.find(p => p.id === state.selectedPortraitId);
      const selectedScene = state.scenes.find(bg => bg.id === state.selectedBackgroundId);
      
      const syncData = {
        isProcessing: state.isProcessing,
        portraitUrl: selectedItem?.url || '',
        backgroundUrl: resolveAssetUrl(selectedScene?.url || ''),
        logoUrl: resolveAssetUrl(config.system.logoUrl || ''),
        mode: state.mode
      };
      localStorage.setItem('cmai_viewport_sync', JSON.stringify(syncData));
    }
  }, [state.isProcessing, state.selectedPortraitId, state.selectedBackgroundId, state.portraits, state.models, state.mode, config, isViewerMode]);

  if (isViewerMode) {
    return <Viewer />;
  }

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

  const setupCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 1280, height: 720 }, 
        audio: false 
      });
      setCameraStream(stream);
      // if (videoRef.current) {
      //   videoRef.current.srcObject = stream;
      // }
      setState(prev => ({ ...prev, isCameraOn: true }));
    } catch (err) {
      console.error("无法访问摄像头:", err);
      setState(prev => ({ ...prev, isCameraOn: false }));
    }
  }, []);

  const handleToggleCamera = () => {
    if (state.isCameraOn) {
      if (cameraStream) cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null);
      setState(prev => ({ ...prev, isCameraOn: false }));
    } else {
      setupCamera();
    }
  };

  const handleTestServer = async () => {
    if (!config) return;
    setState(prev => ({ ...prev, serverStatus: 'testing' }));
    try {
      await fetch(config.blank_url, { method: 'HEAD', mode: 'no-cors' });
      setState(prev => ({ ...prev, serverStatus: 'success' }));
      setTimeout(() => setState(prev => ({ ...prev, serverStatus: 'idle' })), 3000);
    } catch (err) {
      setState(prev => ({ ...prev, serverStatus: 'error' }));
      setTimeout(() => setState(prev => ({ ...prev, serverStatus: 'idle' })), 3000);
    }
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
      try {
        const configRes = await fetch(`./systemConfig.json?t=${Date.now()}`);
        const data: SystemConfig = await configRes.json();
        setConfig(data);
        await refreshPortraits(data);
        await refreshModels(data);
        await refreshScenes(data);
        setupCamera();
      } catch (err) { console.error("Initialization failed:", err); }
    };
    initApp();
  }, [setupCamera]);

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
          onToggleProcessing={() => setState(prev => ({ ...prev, isProcessing: !prev.isProcessing }))}
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
        resolveAssetUrl={resolveAssetUrl}
        onToggleCamera={handleToggleCamera}
        onToggleMatting={() => setState(prev => ({ ...prev, isAutoMatting: !prev.isAutoMatting }))}
        onTestServer={handleTestServer}
        onToggleConfig={() => setState(prev => ({ ...prev, isConfigOpen: !prev.isConfigOpen }))}
      />
    </div>
  );
};

export default App;
