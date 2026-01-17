
import React, { useEffect, useState } from 'react';
import { X } from 'lucide-react';

export const Viewer: React.FC = () => {
  const [viewState, setViewState] = useState({
    isProcessing: false,
    portraitUrl: '',
    backgroundUrl: '',
    logoUrl: ''
  });

  useEffect(() => {
    // Initial sync
    const saved = localStorage.getItem('cmai_viewport_sync');
    if (saved) setViewState(JSON.parse(saved));

    // Listen for real-time updates from main window
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'cmai_viewport_sync' && e.newValue) {
        setViewState(JSON.parse(e.newValue));
      }
    };

    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  const handleClose = () => {
    window.close();
  };

  return (
    <div className="fixed inset-0 bg-black z-[9999] flex flex-col overflow-hidden select-none">
      <div className="flex-1 relative w-full h-full overflow-hidden bg-black">
        
        {/* Sync Background */}
        {viewState.backgroundUrl && (
          <img 
            src={viewState.backgroundUrl} 
            className="absolute inset-0 w-full h-full object-cover blur-[20px] opacity-40 transition-opacity duration-1000 z-0" 
            alt="Scene Layer" 
          />
        )}

        {/* Sync Main Content */}
        <div className="absolute inset-0 z-10">
          {viewState.isProcessing ? (
            <div className="relative w-full h-full flex items-center justify-center overflow-hidden animate-fade-in">
                <img 
                  src={viewState.portraitUrl} 
                  className="w-full h-full object-cover brightness-110" 
                  alt="AI Synthesis Result" 
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/50 via-transparent to-black/10" />
                <div className="absolute top-0 left-0 w-full h-[4px] bg-blue-500 shadow-[0_0_50px_rgba(0,122,255,1)] animate-scan z-20" />
            </div>
          ) : (
            <div className="w-full h-full relative overflow-hidden flex items-center justify-center bg-[#050505]">
               <div className="px-10 py-5 rounded-3xl glass-card border border-white/5 bg-white/5 backdrop-blur-3xl">
                  <span className="text-white/20 text-base font-black uppercase tracking-[0.8em] animate-pulse">Wait Signal...</span>
               </div>
            </div>
          )}
        </div>

        {/* HUD Overlay - Logo in Bottom Right Corner */}
        {viewState.logoUrl && (
          <div className="absolute bottom-10 right-10 z-50 animate-fade-in">
             <div className="flex flex-col items-end group">
                <div className="p-4 rounded-2xl glass-hud border border-white/10 shadow-2xl transition-all duration-500 hover:scale-110 hover:border-white/30">
                   <img 
                    src={viewState.logoUrl} 
                    alt="System Logo" 
                    className="h-10 md:h-14 w-auto object-contain brightness-125 opacity-90 grayscale-[0.2]" 
                   />
                </div>
                <div className="mt-3 text-[8px] font-black text-white/20 uppercase tracking-[0.4em] px-2 drop-shadow-md">
                   China Movie AI Institute
                </div>
             </div>
          </div>
        )}

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
