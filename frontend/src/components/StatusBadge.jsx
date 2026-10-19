import React from 'react';
import { 
  CheckCircle2, 
  HelpCircle, 
  AlertOctagon, 
  FileWarning, 
  XCircle,
  EyeOff
} from 'lucide-react';

export function ProcessingStatusBadge({ status }) {
  switch (status) {
    case 'OK':
      return (
        <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800 border border-emerald-200">
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
          <span>OK</span>
        </span>
      );
    case 'NO_FACE':
      return (
        <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-100 text-amber-800 border border-amber-200" title="No usable face detected by the active detector">
          <EyeOff className="w-3.5 h-3.5 text-amber-600" />
          <span>NO_FACE</span>
        </span>
      );
    case 'READ_ERROR':
      return (
        <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-rose-100 text-rose-800 border border-rose-200">
          <FileWarning className="w-3.5 h-3.5 text-rose-600" />
          <span>READ_ERROR</span>
        </span>
      );
    case 'ALIGNMENT_ERROR':
      return (
        <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-red-100 text-red-800 border border-red-200">
          <AlertOctagon className="w-3.5 h-3.5 text-red-600" />
          <span>ALIGNMENT_ERROR</span>
        </span>
      );
    case 'MODEL_ERROR':
      return (
        <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-red-100 text-red-800 border border-red-200">
          <XCircle className="w-3.5 h-3.5 text-red-600" />
          <span>MODEL_ERROR</span>
        </span>
      );
    default:
      return (
        <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-slate-100 text-slate-700 border border-slate-200">
          <span>{status}</span>
        </span>
      );
  }
}

export function DecisionBadge({ decision, policyOverride = false, originalDecision = null }) {
  const norm = (decision || '').toUpperCase();

  let badgeContent = null;
  switch (norm) {
    case 'PASS':
    case 'ACCEPT':
      badgeContent = (
        <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800 border border-emerald-200">
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
          <span>PASS</span>
          {policyOverride && (
            <span className="ml-1 px-1 py-0.2 bg-emerald-200 text-emerald-900 rounded text-[10px] font-bold" title={`Overridden by policy from ${originalDecision || 'REVIEW'}`}>
              OVERRIDE
            </span>
          )}
        </span>
      );
      break;

    case 'REJECT':
      badgeContent = (
        <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-rose-100 text-rose-800 border border-rose-200">
          <XCircle className="w-3.5 h-3.5 text-rose-600" />
          <span>REJECT</span>
          {policyOverride && (
            <span className="ml-1 px-1 py-0.2 bg-rose-200 text-rose-900 rounded text-[10px] font-bold" title={`Overridden by policy from ${originalDecision || 'REVIEW'}`}>
              OVERRIDE
            </span>
          )}
        </span>
      );
      break;

    case 'REVIEW':
      badgeContent = (
        <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-100 text-amber-800 border border-amber-200">
          <AlertOctagon className="w-3.5 h-3.5 text-amber-600" />
          <span>REVIEW</span>
        </span>
      );
      break;

    case 'NO_FACE':
      badgeContent = (
        <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-gray-100 text-gray-700 border border-gray-300" title="No face detected in source image">
          <EyeOff className="w-3.5 h-3.5 text-gray-500" />
          <span>NO_FACE</span>
        </span>
      );
      break;

    case 'ERROR':
      badgeContent = (
        <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-red-100 text-red-800 border border-red-200" title="Processing error occurred">
          <AlertOctagon className="w-3.5 h-3.5 text-red-600" />
          <span>ERROR</span>
        </span>
      );
      break;

    case 'NOT_EVALUATED':
    default:
      badgeContent = (
        <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-slate-100 text-slate-700 border border-slate-200" title="Decision threshold not calibrated">
          <HelpCircle className="w-3.5 h-3.5 text-slate-500" />
          <span>NOT_EVALUATED</span>
        </span>
      );
      break;
  }

  return badgeContent;
}
