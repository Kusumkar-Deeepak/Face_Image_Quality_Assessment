import React from 'react';
import { 
  FileCheck2, 
  Smile, 
  AlertCircle, 
  Sliders, 
  HelpCircle,
  Clock
} from 'lucide-react';

export default function SessionStats({ stats, onRefresh }) {
  if (!stats) return null;

  const total = stats.total_processed || 0;
  const detected = stats.faces_detected || 0;
  const failures = stats.detection_failures || 0;
  const decisions = stats.quality_decisions_completed || 0;
  const notEvaluated = stats.not_evaluated_count || 0;
  const config = stats.current_config || {};

  return (
    <div className="bg-white rounded-xl border border-slate-200/90 p-4 shadow-xs mb-6">
      <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-100">
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
          <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
            Session Overview
          </h2>
        </div>
        <div className="flex items-center space-x-2 text-xs text-slate-400">
          <Clock className="w-3.5 h-3.5" />
          <span>Active since {new Date(stats.session_start_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        {/* Images Processed */}
        <div className="bg-slate-50/70 rounded-lg p-3 border border-slate-100">
          <div className="flex items-center justify-between text-slate-500 mb-1">
            <span className="text-xs font-medium">Images Processed</span>
            <FileCheck2 className="w-4 h-4 text-blue-500" />
          </div>
          <div className="text-xl font-bold text-slate-800">{total}</div>
          <div className="text-[11px] text-slate-400">Current session count</div>
        </div>

        {/* Faces Detected */}
        <div className="bg-slate-50/70 rounded-lg p-3 border border-slate-100">
          <div className="flex items-center justify-between text-slate-500 mb-1">
            <span className="text-xs font-medium">Faces Detected</span>
            <Smile className="w-4 h-4 text-emerald-500" />
          </div>
          <div className="text-xl font-bold text-slate-800">{detected}</div>
          <div className="text-[11px] text-emerald-600 font-medium">
            {total > 0 ? `${Math.round((detected / total) * 100)}% detection rate` : 'No runs yet'}
          </div>
        </div>

        {/* Detection Failures */}
        <div className="bg-slate-50/70 rounded-lg p-3 border border-slate-100">
          <div className="flex items-center justify-between text-slate-500 mb-1">
            <span className="text-xs font-medium">Detection Failures</span>
            <AlertCircle className="w-4 h-4 text-amber-500" />
          </div>
          <div className="text-xl font-bold text-slate-800">{failures}</div>
          <div className="text-[11px] text-slate-400">NO_FACE or Read errors</div>
        </div>

        {/* Quality Decisions */}
        <div className="bg-slate-50/70 rounded-lg p-3 border border-slate-100">
          <div className="flex items-center justify-between text-slate-500 mb-1">
            <span className="text-xs font-medium">Decisions Completed</span>
            <HelpCircle className="w-4 h-4 text-slate-500" />
          </div>
          <div className="text-xl font-bold text-slate-800">{decisions}</div>
          <div className="text-[11px] text-slate-400" title="Decisions require calibrated thresholds">
            {notEvaluated} NOT_EVALUATED
          </div>
        </div>

        {/* Current Detector Config */}
        <div className="bg-slate-50/70 rounded-lg p-3 border border-slate-100 col-span-2 sm:col-span-1">
          <div className="flex items-center justify-between text-slate-500 mb-1">
            <span className="text-xs font-medium">Detector Config</span>
            <Sliders className="w-4 h-4 text-indigo-500" />
          </div>
          <div className="text-sm font-bold text-slate-800 truncate uppercase">
            {config.primary_detector || 'SCRFD'}
            {config.fallback_detector && (
              <span className="text-xs font-normal text-slate-500 lowercase"> + {config.fallback_detector}</span>
            )}
          </div>
          <div className="text-[11px] text-slate-400">
            Size: {config.scrfd_input_size}px | Pose: {config.head_pose ? 'ON' : 'OFF'}
          </div>
        </div>
      </div>
    </div>
  );
}
