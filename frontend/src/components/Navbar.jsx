import React from 'react';
import { 
  ScanFace, 
  Cpu, 
  Layers, 
  Settings, 
  Activity,
  CheckCircle2,
  AlertTriangle
} from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, health, loadingHealth }) {
  const isHealthy = health?.status === 'ok';
  const device = health?.device?.toUpperCase() || 'CPU';
  const isCuda = health?.cuda_available;

  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-40 shadow-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Title */}
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center text-white shadow-xs">
              <ScanFace className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-lg text-slate-900 tracking-tight">Face Quality Lab</span>
                <span className="text-xs bg-slate-100 text-slate-600 px-2 py-0.5 rounded-full font-medium border border-slate-200">
                  eDifFIQA-S
                </span>
              </div>
              <p className="text-xs text-slate-500">UniFace SCRFD & RetinaFace Assessment</p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex space-x-1 bg-slate-100 p-1 rounded-xl border border-slate-200/80">
            <button
              onClick={() => setActiveTab('single')}
              className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'single'
                  ? 'bg-white text-blue-600 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
              }`}
            >
              <ScanFace className="w-4 h-4" />
              <span>Single Image</span>
            </button>
            <button
              onClick={() => setActiveTab('batch')}
              className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'batch'
                  ? 'bg-white text-blue-600 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
              }`}
            >
              <Layers className="w-4 h-4" />
              <span>Batch Evaluation</span>
            </button>
            <button
              onClick={() => setActiveTab('settings')}
              className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'settings'
                  ? 'bg-white text-blue-600 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
              }`}
            >
              <Settings className="w-4 h-4" />
              <span>Settings</span>
            </button>
          </nav>

          {/* System Indicators */}
          <div className="flex items-center space-x-3">
            {/* Device indicator */}
            <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-slate-100 text-slate-700 border border-slate-200" title={`Inference Device: ${device}`}>
              <Cpu className="w-3.5 h-3.5 text-slate-500" />
              <span>{device}</span>
              {isCuda && <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />}
            </div>

            {/* Model Status Indicator */}
            <div 
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-md text-xs font-medium border ${
                loadingHealth 
                  ? 'bg-slate-50 text-slate-500 border-slate-200' 
                  : isHealthy
                    ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                    : 'bg-amber-50 text-amber-700 border-amber-200'
              }`}
              title={isHealthy ? "eDifFIQA-S Weights Loaded & Ready" : "Model or Weights Offline"}
            >
              <Activity className={`w-3.5 h-3.5 ${isHealthy ? 'text-emerald-500' : 'text-amber-500'}`} />
              <span>{loadingHealth ? 'Connecting...' : isHealthy ? 'Model Ready' : 'Degraded'}</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
