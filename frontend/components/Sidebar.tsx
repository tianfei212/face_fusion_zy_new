
import React, { useState } from 'react';
import { 
  UserCircle2, 
  ScanFace, 
  Settings2, 
  Play, 
  Pause, 
  RefreshCw,
  LayoutGrid,
  ChevronLeft,
  ChevronDown,
  ChevronUp,
  RotateCw,
  X,
  Image as ImageIcon
} from 'lucide-react';
import { AppState, OperationMode, SystemConfig, ThumbnailItem } from '../types';

interface SidebarProps {
  config: SystemConfig;
  state: AppState;
  width: number;
  resolveAssetUrl: (url: string) => string;
  onModeChange: (mode: OperationMode) => void;
  onSliderChange: (key: 'similarity' | 'sharpness', value: number) => void;
  onToggleProcessing: () => void;
  onSwitch: () => void;
  onSelectPortrait: (id: string) => void;
  onSelectBackground: (id: string) => void;
  onToggleCollapse: () => void;
  onRefreshPortraits: () => void;
  onRefreshScenes: () => void;
}

const ImageWithFallback: React.FC<{ url: string, name: string, className?: string }> = ({ url, name, className }) => {
  const [error, setError] = useState(false);
  const firstChar = name.charAt(0);
  
  if (error || !url) {
    return (
      <div className={`${className} bg-gradient-to-br from-[#1a1c1e] to-[#0f1113] flex flex-col items-center justify-center border border-white/5 group-hover:border-blue-500/30 transition-all`}>
        <div className="w-9 h-9 rounded-full bg-blue-500/10 flex items-center justify-center mb-1 shadow-inner">
          <span className="text-blue-400 text-xs font-black">{firstChar}</span>
        </div>
        <span className="text-[9px] font-bold text-white/30 truncate w-[80%] text-center uppercase tracking-tighter">{name}</span>
      </div>
    );
  }

  return (
    <img 
      src={url} 
      alt={name} 
      className={`${className} transition-transform duration-700 group-hover:scale-110`} 
      onError={() => setError(true)} 
    />
  );
};

export const Sidebar: React.FC<SidebarProps> = ({ 
  config,
  state, 
  width,
  resolveAssetUrl,
  onModeChange, 
  onSliderChange, 
  onToggleProcessing,
  onSwitch,
  onSelectPortrait,
  onSelectBackground,
  onToggleCollapse,
  onRefreshPortraits,
  onRefreshScenes
}) => {
  const [isRefreshingP, setIsRefreshingP] = useState(false);
  const [isRefreshingS, setIsRefreshingS] = useState(false);
  const [logoLoadError, setLogoLoadError] = useState(false);
  
  const [isPortraitExpanded, setIsPortraitExpanded] = useState(true);
  const [isSceneExpanded, setIsSceneExpanded] = useState(true);

  const handleRefreshPortraits = (e: React.MouseEvent) => {
    setIsRefreshingP(true);
    onRefreshPortraits();
    setTimeout(() => setIsRefreshingP(false), 800);
  };

  const handleRefreshScenes = () => {
    setIsRefreshingS(true);
    onRefreshScenes();
    setTimeout(() => setIsRefreshingS(false), 800);
  };

  const isMobile = window.innerWidth < 768;
  const isFaceSwap = state.mode === OperationMode.FACE_SWAP;
  const activeList = isFaceSwap ? state.portraits : state.models;

  return (
    <aside 
      style={{ width: isMobile ? `${width}px` : `${width}px` }}
      className={`h-screen glass-panel flex flex-col z-[200] shadow-2xl overflow-hidden shrink-0 border-r border-white/5 pointer-events-auto ${isMobile ? 'absolute left-0 top-0' : ''}`}
    >
      {/* Header - Fixed */}
      <div className="p-6 md:p-8 flex items-center justify-between shrink-0">
        <div className="flex items-center space-x-3">
          {config.system.logoUrl && !logoLoadError ? (
            <img 
              src={resolveAssetUrl(config.system.logoUrl)} 
              alt="Logo" 
              onError={() => setLogoLoadError(true)}
              className="w-9 h-9 md:w-10 md:h-10 object-contain rounded-lg shadow-lg border border-white/10" 
            />
          ) : (
            <div className="w-9 h-9 md:w-10 md:h-10 apple-gradient rounded-xl flex items-center justify-center text-white font-bold text-lg shadow-lg">AI</div>
          )}
          <div className="truncate">
            <h1 className="text-base md:text-lg font-black tracking-tight text-white/90 truncate">{config.system.name}</h1>
            <p className="text-[7px] md:text-[8px] uppercase tracking-[0.3em] text-white/40 font-bold truncate">{config.system.subName}</p>
          </div>
        </div>
        <button onClick={onToggleCollapse} className="p-2 rounded-full hover:bg-white/5 text-white/20 hover:text-white transition-all">
          {isMobile ? <X size={20} /> : <ChevronLeft size={16} />}
        </button>
      </div>

      {/* Main Content Area - Layout Optimized to avoid side scroll */}
      <div className="flex-1 px-6 md:px-8 flex flex-col overflow-hidden gap-6">
        
        {/* Mode Switcher */}
        <section className="shrink-0">
          <div className="grid grid-cols-2 gap-2 bg-black/40 p-1 rounded-2xl border border-white/5">
            <button 
              onClick={() => onModeChange(OperationMode.FACE_SWAP)}
              className={`flex items-center justify-center space-x-2 py-2.5 rounded-xl transition-all ${state.mode === OperationMode.FACE_SWAP ? 'bg-white/10 text-white shadow-lg ring-1 ring-white/10' : 'text-white/30 hover:text-white/50'}`}
            >
              <ScanFace size={14} />
              <span className="font-bold text-[10px] md:text-[11px] uppercase tracking-wider">面容替换</span>
            </button>
            <button 
              onClick={() => onModeChange(OperationMode.MODEL_SWAP)}
              className={`flex items-center justify-center space-x-2 py-2.5 rounded-xl transition-all ${state.mode === OperationMode.MODEL_SWAP ? 'bg-white/10 text-white shadow-lg ring-1 ring-white/10' : 'text-white/30 hover:text-white/50'}`}
            >
              <UserCircle2 size={14} />
              <span className="font-bold text-[10px] md:text-[11px] uppercase tracking-wider">模型替换</span>
            </button>
          </div>
        </section>

        {/* Sliders */}
        <section className="space-y-4 shrink-0 px-1">
          <div className="space-y-2.5">
            <div className="flex justify-between text-[9px] font-black uppercase tracking-widest text-white/40">
              <span>相似度调节</span>
              <span className="text-blue-400">{state.similarity}%</span>
            </div>
            <input type="range" min="0" max="100" value={state.similarity} onChange={(e) => onSliderChange('similarity', parseInt(e.target.value))} className="w-full h-1.5" />
          </div>
          <div className="space-y-2.5">
            <div className="flex justify-between text-[9px] font-black uppercase tracking-widest text-white/40">
              <span>锐化调节</span>
              <span className="text-blue-400">{state.sharpness}%</span>
            </div>
            <input type="range" min="0" max="100" value={state.sharpness} onChange={(e) => onSliderChange('sharpness', parseInt(e.target.value))} className="w-full h-1.5" />
          </div>
        </section>

        {/* Action Buttons */}
        <section className="flex space-x-3 shrink-0">
          <button 
            onClick={onToggleProcessing}
            className={`flex-1 h-12 rounded-xl flex items-center justify-center space-x-2 transition-all duration-500 font-black text-xs uppercase tracking-widest ${state.isProcessing ? 'bg-red-500/80 text-white shadow-lg shadow-red-500/20' : 'apple-gradient text-white shadow-xl shadow-blue-500/20 active:scale-[0.98]'}`}
          >
            {state.isProcessing ? <><Pause size={16} fill="currentColor" /> <span>停止合成</span></> : <><Play size={16} fill="currentColor" /> <span>开始合成</span></>}
          </button>
          <button onClick={onSwitch} className="w-12 h-12 glass-card rounded-xl flex items-center justify-center text-white/40 hover:text-blue-400 transition-all active:scale-95 border border-white/10">
            <RefreshCw size={18} />
          </button>
        </section>

        {/* Asset Lists - Flex-1 to fill space */}
        <div className="flex-1 flex flex-col min-h-0 gap-4 pb-4">
          
          {/* Portrait Tile - Collapsible */}
          <section className={`flex flex-col min-h-0 transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)] ${isPortraitExpanded ? 'flex-[1.2]' : 'flex-none'}`}>
             <div className="bg-black/20 rounded-2xl border border-white/5 flex flex-col h-full overflow-hidden">
                {/* Tile Header */}
                <div 
                  className="flex items-center justify-between p-3.5 bg-white/5 cursor-pointer hover:bg-white/10 transition-colors select-none group"
                  onClick={() => setIsPortraitExpanded(!isPortraitExpanded)}
                >
                   <div className="flex items-center space-x-2">
                      <LayoutGrid size={12} className="text-blue-400" />
                      <span className="text-[10px] font-black text-white/60 uppercase tracking-[0.15em]">{isFaceSwap ? '目标人像' : 'DFM模型'}</span>
                   </div>
                   <div className="flex items-center space-x-2">
                      <button onClick={handleRefreshPortraits} className={`text-white/20 hover:text-blue-400 transition-all p-1 hover:bg-white/10 rounded-lg ${isRefreshingP ? 'animate-spin' : ''}`}>
                         <RotateCw size={12} />
                      </button>
                      {isPortraitExpanded ? <ChevronUp size={14} className="text-white/20" /> : <ChevronDown size={14} className="text-white/20" />}
                   </div>
                </div>

                {/* Tile Content */}
                {isPortraitExpanded && (
                  <div className="flex-1 overflow-y-auto scrollbar-custom p-2.5 min-h-0">
                      <div className="grid grid-cols-3 gap-2.5">
                        {activeList.map((p) => (
                          <button 
                            key={p.id}
                            onClick={() => onSelectPortrait(p.id)}
                            className={`group relative aspect-square rounded-lg overflow-hidden transition-all duration-500 ${state.selectedPortraitId === p.id ? 'ring-2 ring-blue-500 shadow-[0_0_15px_rgba(0,122,255,0.3)] z-10 scale-[1.02]' : 'opacity-50 hover:opacity-100 border border-white/5'}`}
                          >
                            <ImageWithFallback url={resolveAssetUrl(p.url)} name={p.name} className="w-full h-full object-cover" />
                            <div className={`absolute inset-0 bg-blue-500/10 transition-opacity ${state.selectedPortraitId === p.id ? 'opacity-100' : 'opacity-0'}`} />
                            <div className="absolute bottom-0 inset-x-0 p-1 bg-black/60 backdrop-blur-md opacity-0 group-hover:opacity-100 transition-opacity">
                              <p className="text-[7px] text-white/80 text-center truncate font-bold uppercase tracking-tighter">{p.name}</p>
                            </div>
                          </button>
                        ))}
                      </div>
                  </div>
                )}
             </div>
          </section>

          {/* Scene Tile - Collapsible */}
          <section className={`flex flex-col min-h-0 transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)] ${isSceneExpanded ? 'flex-1' : 'flex-none'}`}>
             <div className="bg-black/20 rounded-2xl border border-white/5 flex flex-col h-full overflow-hidden">
                {/* Tile Header */}
                <div 
                  className="flex items-center justify-between p-3.5 bg-white/5 cursor-pointer hover:bg-white/10 transition-colors select-none group"
                  onClick={() => setIsSceneExpanded(!isSceneExpanded)}
                >
                   <div className="flex items-center space-x-2">
                      <ImageIcon size={12} className="text-purple-400" />
                      <span className="text-[10px] font-black text-white/60 uppercase tracking-[0.15em]">场景替换</span>
                   </div>
                   <div className="flex items-center space-x-2">
                      <button onClick={handleRefreshScenes} className={`text-white/20 hover:text-blue-400 transition-all p-1 hover:bg-white/10 rounded-lg ${isRefreshingS ? 'animate-spin' : ''}`}>
                         <RotateCw size={12} />
                      </button>
                      {isSceneExpanded ? <ChevronUp size={14} className="text-white/20" /> : <ChevronDown size={14} className="text-white/20" />}
                   </div>
                </div>

                {/* Tile Content */}
                {isSceneExpanded && (
                  <div className="flex-1 overflow-y-auto scrollbar-custom p-2.5 min-h-0">
                      <div className="grid grid-cols-2 gap-2.5">
                        {state.scenes.map((bg) => (
                          <button 
                            key={bg.id}
                            onClick={() => onSelectBackground(bg.id)}
                            className={`group relative aspect-video rounded-lg overflow-hidden transition-all duration-500 ${state.selectedBackgroundId === bg.id ? 'ring-2 ring-blue-500 shadow-[0_0_15px_rgba(0,122,255,0.3)] z-10 scale-[1.02]' : 'opacity-50 hover:opacity-100 border border-white/5'}`}
                          >
                            <ImageWithFallback url={resolveAssetUrl(bg.url)} name={bg.name} className="w-full h-full object-cover" />
                            <div className={`absolute inset-0 bg-blue-500/5 transition-opacity ${state.selectedBackgroundId === bg.id ? 'opacity-100' : 'opacity-0'}`} />
                            <div className="absolute bottom-0 inset-x-0 p-1 bg-black/60 backdrop-blur-md opacity-0 group-hover:opacity-100 transition-opacity">
                              <p className="text-[7px] text-white/80 text-center truncate font-bold uppercase tracking-tighter">{bg.name}</p>
                            </div>
                          </button>
                        ))}
                      </div>
                  </div>
                )}
             </div>
          </section>
        </div>
      </div>

      {/* Footer - Fixed */}
      <div className="px-8 py-6 border-t border-white/5 flex items-center justify-center shrink-0">
        <p className="text-[7px] md:text-[8px] text-white/20 font-black uppercase tracking-[0.4em]">{config.system.footerText}</p>
      </div>

      <style>{`
        .scrollbar-custom::-webkit-scrollbar { width: 3px; }
        .scrollbar-custom::-webkit-scrollbar-track { background: transparent; }
        .scrollbar-custom::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.05); border-radius: 10px; }
        .scrollbar-custom::-webkit-scrollbar-thumb:hover { background: rgba(0, 122, 255, 0.3); }
        input[type='range']::-webkit-slider-thumb {
            height: 14px;
            width: 14px;
            margin-top: -6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.8);
        }
      `}</style>
    </aside>
  );
};
