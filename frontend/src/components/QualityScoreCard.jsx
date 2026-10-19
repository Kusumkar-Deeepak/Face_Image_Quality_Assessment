import React, { useState } from 'react';
import { Sparkles, Info, AlertCircle, ChevronDown, ChevronUp, CheckCircle, AlertTriangle } from 'lucide-react';
import { DecisionBadge } from './StatusBadge';

export default function QualityScoreCard({ score, decision, decisionReason, status }) {
  const [showTiers, setShowTiers] = useState(false);
  const hasScore = score !== null && score !== undefined && !isNaN(score);
  const numericScore = hasScore ? Number(score) : 0;
  const formattedScore = hasScore ? numericScore.toFixed(4) : '-';
  
  // Calculate percentage for visualization bar (clamped 0 to 1)
  const clampedScore = Math.min(Math.max(numericScore, 0), 1);
  const percentage = Math.round(clampedScore * 100);

  // Reference quality tier categorization (derived from eDifFIQA/NIST benchmarks)
  let tierLabel = 'Unrated';
  let tierColor = 'text-slate-600 bg-slate-100 border-slate-200';
  let tierDesc = '';
  if (hasScore) {
    if (numericScore >= 0.80) {
      tierLabel = 'Tier 1: High Quality';
      tierColor = 'text-emerald-700 bg-emerald-50 border-emerald-200';
      tierDesc = 'Sharp features, optimal contrast and alignment. Suitable for 1:N automated biometric identification.';
    } else if (numericScore >= 0.65) {
      tierLabel = 'Tier 2: Standard Usable';
      tierColor = 'text-blue-700 bg-blue-50 border-blue-200';
      tierDesc = 'Good fidelity with everyday lighting. Suitable for 1:1 automated verification.';
    } else if (numericScore >= 0.45) {
      tierLabel = 'Tier 3: Degraded Quality';
      tierColor = 'text-amber-700 bg-amber-50 border-amber-200';
      tierDesc = 'Noticeable compression, low resolution, or moderate noise. Borderline for automated recognition.';
    } else {
      tierLabel = 'Tier 4: Poor Quality';
      tierColor = 'text-rose-700 bg-rose-50 border-rose-200';
      tierDesc = 'Severe motion blur, defocus, extreme illumination, or occlusion. High probability of matching failure.';
    }
  }

  return (
    <div className="bg-white rounded-xl border border-slate-200/90 p-5 shadow-xs">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-blue-50 text-blue-600">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-900">
              eDifFIQA Quality Score
            </h3>
            <p className="text-xs text-slate-500">
              Model Quality Score (Not a Probability)
            </p>
          </div>
        </div>
        <div>
          <DecisionBadge decision={decision} />
        </div>
      </div>

      {hasScore ? (
        <div className="mt-4">
          {/* Main numeric score display & Quality Tier pill */}
          <div className="flex flex-wrap items-baseline justify-between gap-2 mb-2">
            <div className="flex items-baseline space-x-2">
              <span className="text-4xl font-extrabold tracking-tight text-slate-900 font-mono">
                {formattedScore}
              </span>
              <span className="text-xs text-slate-400 font-medium">
                raw model output
              </span>
            </div>
            <span className={`text-xs font-semibold px-2.5 py-1 rounded-full border ${tierColor}`}>
              {tierLabel}
            </span>
          </div>

          {/* Visual Gauge Bar */}
          <div className="w-full bg-slate-100 rounded-full h-2.5 overflow-hidden border border-slate-200/60 my-3">
            <div 
              className={`h-full rounded-full transition-all duration-500 ease-out ${
                numericScore >= 0.80 ? 'bg-emerald-500' :
                numericScore >= 0.65 ? 'bg-blue-600' :
                numericScore >= 0.45 ? 'bg-amber-500' : 'bg-rose-500'
              }`}
              style={{ width: `${percentage}%` }}
            />
          </div>
          <div className="flex justify-between text-[11px] text-slate-400 font-mono mb-3">
            <span>0.0000 (Poor)</span>
            <span>0.5000</span>
            <span>1.0000 (Optimal)</span>
          </div>

          <p className="text-xs text-slate-600 mb-3 italic">
            {tierDesc}
          </p>

          {/* Threshold calibration notice */}
          {decision === 'NOT_EVALUATED' && (
            <div className="flex items-start space-x-2.5 p-3 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-600">
              <Info className="w-4 h-4 text-blue-500 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-slate-800">
                  Score available — decision threshold not calibrated.
                </p>
                <p className="mt-0.5 text-slate-500">
                  The model produced an inference score, but no acceptance or rejection thresholds have been configured in <code className="text-slate-700 bg-slate-100 px-1 py-0.5 rounded font-mono text-[11px]">quality_thresholds.json</code>. Operational thresholds must be calibrated on target dataset error rates.
                </p>
              </div>
            </div>
          )}

          {decisionReason && decision !== 'NOT_EVALUATED' && (
            <div className="mt-2 text-xs text-slate-600">
              <span className="font-medium text-slate-700">Decision rule:</span> {decisionReason}
            </div>
          )}

          {/* Collapsible reference guide */}
          <div className="mt-3 pt-2 border-t border-slate-100">
            <button
              onClick={() => setShowTiers(!showTiers)}
              className="flex items-center justify-between w-full text-[11px] font-medium text-slate-500 hover:text-slate-800 transition-colors"
            >
              <span>Reference Quality Tiers Guide</span>
              {showTiers ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>
            {showTiers && (
              <div className="mt-2 space-y-1.5 text-[11px] bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                <div className="flex justify-between text-emerald-800">
                  <span className="font-semibold">≥ 0.80 — High Quality:</span>
                  <span>Optimal for 1:N biometric search</span>
                </div>
                <div className="flex justify-between text-blue-800">
                  <span className="font-semibold">0.65 to 0.79 — Standard Usable:</span>
                  <span>Reliable for 1:1 verification</span>
                </div>
                <div className="flex justify-between text-amber-800">
                  <span className="font-semibold">0.45 to 0.64 — Degraded:</span>
                  <span>Noticeable blur / compression</span>
                </div>
                <div className="flex justify-between text-rose-800">
                  <span className="font-semibold">&lt; 0.45 — Poor:</span>
                  <span>High risk of false rejection</span>
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="py-6 text-center">
          <div className="inline-flex p-2.5 rounded-full bg-slate-100 text-slate-400 mb-2">
            <AlertCircle className="w-6 h-6" />
          </div>
          <p className="text-sm font-medium text-slate-700">No Quality Score Available</p>
          <p className="text-xs text-slate-500 mt-1 max-w-sm mx-auto">
            {status === 'NO_FACE'
              ? 'No usable face was detected to feed the 112×112 quality backbone.'
              : status === 'READ_ERROR'
              ? 'Image could not be decoded.'
              : 'Quality score could not be produced.'}
          </p>
        </div>
      )}
    </div>
  );
}
