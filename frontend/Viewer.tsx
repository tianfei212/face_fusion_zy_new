
import React, { useEffect, useRef, useState } from 'react';
import { X } from 'lucide-react';
import { useStreamReceiver } from './hooks/useStreamReceiver';

export const Viewer: React.FC = () => {
  const [configUrl, setConfigUrl] = useState('');
  const remoteCanvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const init = async () => {
      try {
        const res = await fetch('./systemConfig.json');
        const cfg = await res.json();
        setConfigUrl(cfg.stream_recv_url || cfg.stream_url || cfg.blank_url || '');
      } catch {}
    };
    init();
  }, []);

  const { status } = useStreamReceiver(configUrl, remoteCanvasRef);

  const handleClose = () => {
    window.close();
  };

  return (
    <div className="fixed inset-0 bg-black z-[9999] flex flex-col overflow-hidden select-none">
      <div className="flex-1 relative w-full h-full overflow-hidden bg-black">
        
        {/* Remote Stream Canvas */}
        <div className="absolute inset-0 z-10">
          <canvas ref={remoteCanvasRef} className="w-full h-full object-cover" />
        </div>

        {/* Status Badge */}
        <div className="absolute top-10 left-10 z-50">
          <div className="px-3 py-2 rounded-xl glass-hud border border-white/10 text-[10px] font-black uppercase tracking-widest text-white/70">
            {status}
          </div>
        </div>

        {/* Top Right Close Button */}
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
        @keyframes scan { 
          0% { transform: translateY(-10vh); opacity: 0; } 
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { transform: translateY(110vh); opacity: 0; } 
        }
        .animate-scan { animation: scan 7s cubic-bezier(0.4, 0, 0.2, 1) infinite; }
        .animate-fade-in { animation: fade-in 1.2s ease-out forwards; }
        @keyframes fade-in { from { opacity: 0; } to { opacity: 1; } }
        
        .glass-hud {
            background: rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(25px) saturate(150%);
            -webkit-backdrop-filter: blur(25px) saturate(150%);
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
        }
      `}</style>
    </div>
  );
};
