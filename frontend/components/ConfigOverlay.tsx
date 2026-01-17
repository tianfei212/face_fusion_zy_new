
import React, { useState } from 'react';
import { 
  X, 
  Save, 
  Monitor, 
  FolderOpen, 
  Globe, 
  Database,
  Radio,
  Eye,
  EyeOff
} from 'lucide-react';
import { SystemConfig } from '../types';

interface ConfigOverlayProps {
  config: SystemConfig;
  isOpen: boolean;
  onClose: () => void;
}

export const ConfigOverlay: React.FC<ConfigOverlayProps> = ({ config, isOpen, onClose }) => {
  if (!isOpen) return null;

  const [localConfig, setLocalConfig] = useState<SystemConfig>(config);

  const handleInputChange = (section: keyof SystemConfig | 'directories' | 'directories_blank' | 'obsConfig', key: string, value: any) => {
    if (section === 'Is_from_blank_get_config' || section === 'blank_url') {
      setLocalConfig(prev => ({
        ...prev,
        [section]: value
      }));
    } else {
      setLocalConfig(prev => ({
        ...prev,
        [section]: {
          ...prev[section as any],
          [key]: value
        }
      }));
    }
  };

  const [activeTab, setActiveTab] = useState('general');
  const [showSecret, setShowSecret] = useState(false);
  const [obsEnabled, setObsEnabled] = useState(config.obsConfig?.enabled || false);

  const navItems = [
    { id: 'general', icon: Monitor, label: '通用设置' },
    { id: 'dir', icon: FolderOpen, label: '本地目录' },
    { id: 'network', icon: Globe, label: '后端网络' },
    { id: 'streaming', icon: Radio, label: '流媒体配置' }
  ];

  const handleSave = async () => {
    // 模拟保存：这里实际上应该调用后端接口
    console.log('Saving config:', localConfig);
    alert('配置保存功能暂未对接后端接口');
  };

  return (
    <div className="absolute inset-0 z-[1000] glass-panel backdrop-blur-[60px] animate-fade-in flex flex-col p-6 md:p-20 select-none overflow-hidden">
      <div className="flex items-center justify-between mb-8 md:mb-16">
        <div>
          <h2 className="text-2xl md:text-4xl font-bold text-white mb-1 md:mb-2 tracking-tight">系统独立配置</h2>
          <p className="text-white/40 text-[8px] md:text-xs font-black uppercase tracking-[0.4em]">CMAI Engine Settings Panel</p>
        </div>
        <button onClick={onClose} className="w-10 h-10 md:w-14 md:h-14 rounded-full glass-card flex items-center justify-center text-white/60 hover:text-white transition-all active:scale-90 border border-white/10"><X size={28} /></button>
      </div>

      <div className="flex-1 flex flex-col md:flex-row space-y-6 md:space-y-0 md:space-x-12 overflow-hidden">
        <div className="flex md:flex-col space-x-2 md:space-x-0 md:space-y-4 overflow-x-auto md:overflow-x-visible pb-2 md:pb-0 scrollbar-hide md:w-64 shrink-0">
          {navItems.map((item) => (
            <button 
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`flex items-center space-x-3 md:space-x-4 px-4 md:px-6 py-3 md:py-4 rounded-xl md:rounded-2xl transition-all font-bold text-xs md:text-sm whitespace-nowrap ${activeTab === item.id ? 'bg-blue-500 text-white shadow-lg' : 'text-white/40 hover:bg-white/5 hover:text-white/80'}`}
            >
              <item.icon size={18} /><span>{item.label}</span>
            </button>
          ))}
        </div>

        <div className="flex-1 bg-black/20 rounded-[32px] md:rounded-[40px] border border-white/5 p-6 md:p-12 overflow-y-auto scrollbar-hide">
          {activeTab === 'general' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 md:gap-10 animate-slide-up">
              <div className="space-y-6 md:space-y-8">
                <div className="space-y-3">
                  <label className="text-[9px] md:text-[10px] font-black text-white/30 uppercase tracking-widest block">系统名称</label>
                  <input 
                    type="text" 
                    value={localConfig.system.name} 
                    onChange={(e) => handleInputChange('system', 'name', e.target.value)}
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:border-blue-500 outline-none" 
                  />
                </div>
                <div className="space-y-3">
                  <label className="text-[9px] md:text-[10px] font-black text-white/30 uppercase tracking-widest block">页脚文案</label>
                  <input 
                    type="text" 
                    value={localConfig.system.footerText} 
                    onChange={(e) => handleInputChange('system', 'footerText', e.target.value)}
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:border-blue-500 outline-none" 
                  />
                </div>
                <div className="space-y-3">
                  <label className="text-[9px] md:text-[10px] font-black text-white/30 uppercase tracking-widest block">待机图片 URL</label>
                  <input 
                    type="text" 
                    value={localConfig.system.standbyImageUrl || ''} 
                    onChange={(e) => handleInputChange('system', 'standbyImageUrl', e.target.value)}
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:border-blue-500 outline-none" 
                  />
                </div>
              </div>
              <div className="space-y-6 md:space-y-8">
                <div className="space-y-3">
                  <label className="text-[9px] md:text-[10px] font-black text-white/30 uppercase tracking-widest block">副标题</label>
                  <input 
                    type="text" 
                    value={localConfig.system.subName} 
                    onChange={(e) => handleInputChange('system', 'subName', e.target.value)}
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:border-blue-500 outline-none" 
                  />
                </div>
                <div className="space-y-3">
                  <label className="text-[9px] md:text-[10px] font-black text-white/30 uppercase tracking-widest block">摄像头离线图片 URL</label>
                  <input 
                    type="text" 
                    value={localConfig.system.cameraOfflineUrl || ''} 
                    onChange={(e) => handleInputChange('system', 'cameraOfflineUrl', e.target.value)}
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:border-blue-500 outline-none" 
                  />
                </div>
                <div className="flex items-center justify-between p-6 bg-white/5 rounded-2xl border border-white/5">
                  <div>
                    <div className="text-white font-bold text-sm">远程加载</div>
                    <div className="text-white/30 text-[10px]">Is_from_blank_get_config</div>
                  </div>
                  <div 
                    onClick={() => setLocalConfig(prev => ({ ...prev, Is_from_blank_get_config: !prev.Is_from_blank_get_config }))}
                    className={`w-12 h-6 rounded-full relative cursor-pointer transition-all ${localConfig.Is_from_blank_get_config ? 'bg-blue-500' : 'bg-white/10'}`}
                  >
                    <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-all ${localConfig.Is_from_blank_get_config ? 'right-1' : 'left-1'}`} />
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'dir' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 md:gap-10 animate-slide-up">
              <div className="space-y-6 md:space-y-8">
                <div className="space-y-3">
                  <label className="text-[9px] md:text-[10px] font-black text-white/30 uppercase tracking-widest block">目标人像目录</label>
                  <input 
                    type="text" 
                    value={localConfig.directories.portraitWorkDir} 
                    onChange={(e) => handleInputChange('directories', 'portraitWorkDir', e.target.value)}
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:border-blue-500 outline-none" 
                  />
                </div>
                <div className="space-y-3">
                  <label className="text-[9px] md:text-[10px] font-black text-white/30 uppercase tracking-widest block">场景背景目录</label>
                  <input 
                    type="text" 
                    value={localConfig.directories.sceneDir} 
                    onChange={(e) => handleInputChange('directories', 'sceneDir', e.target.value)}
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:border-blue-500 outline-none" 
                  />
                </div>
              </div>
              <div className="space-y-6 md:space-y-8">
                <div className="space-y-3">
                  <label className="text-[9px] md:text-[10px] font-black text-white/30 uppercase tracking-widest block">DFM 模型目录</label>
                  <input 
                    type="text" 
                    value={localConfig.directories.portraitDfmDir} 
                    onChange={(e) => handleInputChange('directories', 'portraitDfmDir', e.target.value)}
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:border-blue-500 outline-none" 
                  />
                </div>
              </div>
            </div>
          )}

          {activeTab === 'network' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 md:gap-10 animate-slide-up">
              <div className="space-y-6 md:space-y-8">
                <div className="space-y-3">
                  <label className="text-[9px] md:text-[10px] font-black text-white/30 uppercase tracking-widest block">后端服务地址</label>
                  <input 
                    type="text" 
                    value={localConfig.blank_url} 
                    onChange={(e) => handleInputChange('blank_url', '', e.target.value)}
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:border-blue-500 outline-none" 
                  />
                </div>
                <div className="space-y-3">
                  <label className="text-[9px] md:text-[10px] font-black text-white/30 uppercase tracking-widest block">人像列表 API</label>
                  <input 
                    type="text" 
                    value={localConfig.directories_blank.portraitWorkList} 
                    onChange={(e) => handleInputChange('directories_blank', 'portraitWorkList', e.target.value)}
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:border-blue-500 outline-none" 
                  />
                </div>
              </div>
              <div className="space-y-6 md:space-y-8">
                <div className="space-y-3">
                  <label className="text-[9px] md:text-[10px] font-black text-white/30 uppercase tracking-widest block">DFM 模型 API</label>
                  <input 
                    type="text" 
                    value={localConfig.directories_blank.portraitDfmList} 
                    onChange={(e) => handleInputChange('directories_blank', 'portraitDfmList', e.target.value)}
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:border-blue-500 outline-none" 
                  />
                </div>
                <div className="space-y-3">
                  <label className="text-[9px] md:text-[10px] font-black text-white/30 uppercase tracking-widest block">场景列表 API</label>
                  <input 
                    type="text" 
                    value={localConfig.directories_blank.sceneList} 
                    onChange={(e) => handleInputChange('directories_blank', 'sceneList', e.target.value)}
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:border-blue-500 outline-none" 
                  />
                </div>
              </div>
            </div>
          )}

          {activeTab === 'streaming' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 md:gap-10 animate-slide-up">
               <div className="space-y-6 md:space-y-8">
                  <div className="flex items-center justify-between p-6 bg-white/5 rounded-2xl border border-white/5">
                    <div>
                      <div className="text-white font-bold text-sm">启用推流</div>
                      <div className="text-white/30 text-[10px]">Enable OBS Streaming</div>
                    </div>
                    <div 
                      onClick={() => {
                        const newVal = !localConfig.obsConfig.enabled;
                        setObsEnabled(newVal);
                        handleInputChange('obsConfig', 'enabled', newVal);
                      }}
                      className={`w-12 h-6 rounded-full relative cursor-pointer transition-all ${localConfig.obsConfig.enabled ? 'bg-blue-500' : 'bg-white/10'}`}
                    >
                      <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-all ${localConfig.obsConfig.enabled ? 'right-1' : 'left-1'}`} />
                    </div>
                  </div>
                  <div className="space-y-3">
                    <label className="text-[9px] md:text-[10px] font-black text-white/30 uppercase tracking-widest block">推流地址</label>
                    <input 
                      type="text" 
                      value={localConfig.obsConfig.streamUrl} 
                      onChange={(e) => handleInputChange('obsConfig', 'streamUrl', e.target.value)}
                      className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:border-blue-500 outline-none" 
                    />
                  </div>
               </div>
               <div className="space-y-6 md:space-y-8">
                  <div className="space-y-3">
                    <label className="text-[9px] md:text-[10px] font-black text-white/30 uppercase tracking-widest block">用户名</label>
                    <input 
                      type="text" 
                      value={localConfig.obsConfig.username} 
                      onChange={(e) => handleInputChange('obsConfig', 'username', e.target.value)}
                      className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:border-blue-500 outline-none" 
                    />
                  </div>
                  <div className="space-y-3">
                    <label className="text-[9px] md:text-[10px] font-black text-white/30 uppercase tracking-widest block">密钥</label>
                    <div className="relative">
                      <input 
                        type={showSecret ? "text" : "password"}
                        value={localConfig.obsConfig.secret} 
                        onChange={(e) => handleInputChange('obsConfig', 'secret', e.target.value)}
                        className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:border-blue-500 outline-none pr-10" 
                      />
                      <button onClick={() => setShowSecret(!showSecret)} className="absolute right-3 top-3.5 text-white/30 hover:text-white">
                        {showSecret ? <EyeOff size={16} /> : <Eye size={16} />}
                      </button>
                    </div>
                  </div>
               </div>
            </div>
          )}
          <div className="mt-16 flex justify-end">
            <button onClick={handleSave} className="px-10 py-4 apple-gradient text-white rounded-2xl font-bold text-sm shadow-xl active:scale-95 transition-all flex items-center space-x-3"><Save size={18} /><span>保存配置</span></button>
          </div>
        </div>
      </div>
      <style>{`
        @keyframes fade-in { from { opacity: 0; transform: scale(0.98); } to { opacity: 1; transform: scale(1); } }
        @keyframes slide-up { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .animate-fade-in { animation: fade-in 0.3s ease-out forwards; }
        .animate-slide-up { animation: slide-up 0.4s ease-out forwards; }
      `}</style>
    </div>
  );
};
