import React, { useState } from 'react';
import { 
  Sliders, 
  ShieldAlert, 
  HelpCircle, 
  CheckCircle2, 
  Info,
  Layers,
  Compass,
  Save,
  RotateCcw,
  Sparkles,
  AlertCircle
} from 'lucide-react';
import { updateThresholds } from '../api';

export default function SettingsView({ config, health, loading, onConfigUpdated }) {
  const [acceptMin, setAcceptMin] = useState(config?.thresholds?.accept_min ?? '');
  const [rejectMax, setRejectMax] = useState(config?.thresholds?.reject_max ?? '');
  const [maxAbsYaw, setMaxAbsYaw] = useState(config?.thresholds?.max_abs_yaw ?? '');
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState(null);
  const [error, setError] = useState(null);

  if (loading || !config) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-8 text-center text-slate-500 text-xs">
        Loading configuration from backend...
      </div>
    );
  }

  const { scrfd, retinaface, thresholds, is_calibrated } = config;

  const handleApplyPreset = (preset) => {
    setError(null);
    setMessage(null);
    if (preset === 'balanced') {
      setAcceptMin(0.65);
      setRejectMax(0.40);
      setMaxAbsYaw(30);
    } else if (preset === 'strict') {
      setAcceptMin(0.75);
      setRejectMax(0.50);
      setMaxAbsYaw(20);
    } else if (preset === 'uncalibrated') {
      setAcceptMin('');
      setRejectMax('');
      setMaxAbsYaw('');
    }
  };

  const handleSaveThresholds = async (e) => {
    e.preventDefault();
    setSaving(true);
    setError(null);
    setMessage(null);

    const aMin = acceptMin === '' ? null : parseFloat(acceptMin);
    const rMax = rejectMax === '' ? null : parseFloat(rejectMax);
    const yMax = maxAbsYaw === '' ? null : parseFloat(maxAbsYaw);

    if (aMin !== null && rMax !== null && rMax >= aMin) {
      setError('Reject Max must be strictly lower than Accept Min.');
      setSaving(false);
      return;
    }

    try {
      await updateThresholds({
        accept_min: aMin,
        reject_max: rMax,
        max_abs_pitch: null,
        max_abs_yaw: yMax,
        max_abs_roll: null,
      });

      setMessage(
        aMin === null && rMax === null
          ? 'Thresholds reset to Uncalibrated (Decisions will output NOT_EVALUATED).'
          : `Thresholds updated! Scores ≥ ${aMin} will be ACCEPT, ≤ ${rMax} will be REJECT, in-between will be REVIEW.`
      );
      if (onConfigUpdated) onConfigUpdated();
    } catch (err) {
      setError(err.message || 'Failed to update thresholds');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Header */}
      <div>
        <h2 className="text-base font-bold text-slate-900 tracking-tight">System & Decision Configuration</h2>
        <p className="text-xs text-slate-500 mt-0.5">
          Manage detector parameters and quality decision boundaries (<code className="font-mono text-slate-700 bg-slate-100 px-1 py-0.5 rounded">quality_thresholds.json</code>).
        </p>
      </div>

      {/* Thresholds Calibration Status Notice */}
      <div className={`p-4 rounded-xl border ${
        is_calibrated 
          ? 'bg-emerald-50 border-emerald-200 text-emerald-900' 
          : 'bg-amber-50/70 border-amber-200/90 text-amber-950'
      }`}>
        <div className="flex items-start space-x-3">
          {is_calibrated ? (
            <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0 mt-0.5" />
          ) : (
            <ShieldAlert className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
          )}
          <div className="space-y-1">
            <h3 className="text-sm font-semibold">
              {is_calibrated ? 'Calibrated Thresholds Active (Decisions Enabled)' : 'Thresholds Currently Uncalibrated (Default)'}
            </h3>
            <p className="text-xs text-slate-700 leading-relaxed">
              {is_calibrated ? (
                <>
                  Decisions now evaluate as <strong>ACCEPT</strong>, <strong>REJECT</strong>, or <strong>REVIEW</strong>.
                  Current rule: Score ≥ {thresholds.accept_min} → ACCEPT; Score ≤ {thresholds.reject_max} → REJECT; between → REVIEW.
                </>
              ) : (
                'All quality score decisions currently output NOT_EVALUATED because accept_min and reject_max are null. You can configure decision thresholds below using your custom values or standard biometric presets.'
              )}
            </p>
          </div>
        </div>
      </div>

      {/* Interactive Quality Decision Threshold Configuration Form */}
      <div className="bg-white rounded-xl border border-slate-200/90 p-5 shadow-xs">
        <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-100">
          <div className="flex items-center space-x-2">
            <Sliders className="w-4 h-4 text-blue-600" />
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-700">
              Configure Quality Decision Thresholds
            </h3>
          </div>
          <span className="text-xs text-slate-400">Controls ACCEPT, REJECT, and REVIEW decisions</span>
        </div>

        {/* Quick Presets Bar */}
        <div className="mb-4 bg-slate-50 p-3 rounded-lg border border-slate-100">
          <span className="text-xs font-semibold text-slate-700 block mb-2">
            Quick Calibration Presets:
          </span>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => handleApplyPreset('balanced')}
              className="px-3 py-1.5 rounded-lg text-xs font-medium bg-white text-slate-700 hover:bg-blue-50 hover:text-blue-700 border border-slate-200 shadow-2xs transition-colors flex items-center space-x-1"
            >
              <Sparkles className="w-3.5 h-3.5 text-blue-600" />
              <span>Balanced Verification (0.65 / 0.40)</span>
            </button>
            <button
              type="button"
              onClick={() => handleApplyPreset('strict')}
              className="px-3 py-1.5 rounded-lg text-xs font-medium bg-white text-slate-700 hover:bg-emerald-50 hover:text-emerald-700 border border-slate-200 shadow-2xs transition-colors flex items-center space-x-1"
            >
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              <span>Strict Biometrics (0.75 / 0.50)</span>
            </button>
            <button
              type="button"
              onClick={() => handleApplyPreset('uncalibrated')}
              className="px-3 py-1.5 rounded-lg text-xs font-medium bg-white text-slate-600 hover:bg-slate-100 border border-slate-200 shadow-2xs transition-colors flex items-center space-x-1"
            >
              <RotateCcw className="w-3.5 h-3.5 text-slate-500" />
              <span>Reset to Uncalibrated (Null)</span>
            </button>
          </div>
        </div>

        {/* Inputs Form */}
        <form onSubmit={handleSaveThresholds} className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">
                Accept Min Threshold (0.0 to 1.0)
              </label>
              <input
                type="number"
                step="0.01"
                min="0.0"
                max="1.0"
                placeholder="Leave blank for null"
                value={acceptMin}
                onChange={(e) => setAcceptMin(e.target.value)}
                className="w-full text-xs rounded-lg border border-slate-200 p-2 font-mono bg-slate-50 focus:bg-white focus:ring-1 focus:ring-blue-500"
              />
              <span className="text-[10px] text-slate-400 block mt-1">Score ≥ Accept Min → ACCEPT</span>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">
                Reject Max Threshold (0.0 to 1.0)
              </label>
              <input
                type="number"
                step="0.01"
                min="0.0"
                max="1.0"
                placeholder="Leave blank for null"
                value={rejectMax}
                onChange={(e) => setRejectMax(e.target.value)}
                className="w-full text-xs rounded-lg border border-slate-200 p-2 font-mono bg-slate-50 focus:bg-white focus:ring-1 focus:ring-blue-500"
              />
              <span className="text-[10px] text-slate-400 block mt-1">Score ≤ Reject Max → REJECT</span>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">
                Max Abs Yaw Limit (Degrees)
              </label>
              <input
                type="number"
                step="1"
                min="1"
                max="90"
                placeholder="Optional (e.g. 30)"
                value={maxAbsYaw}
                onChange={(e) => setMaxAbsYaw(e.target.value)}
                className="w-full text-xs rounded-lg border border-slate-200 p-2 font-mono bg-slate-50 focus:bg-white focus:ring-1 focus:ring-blue-500"
              />
              <span className="text-[10px] text-slate-400 block mt-1">|Yaw| &gt; Limit → REVIEW</span>
            </div>
          </div>

          {/* Feedback messages */}
          {error && (
            <div className="p-3 rounded-lg bg-rose-50 border border-rose-200 text-xs text-rose-800 flex items-center space-x-2">
              <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {message && (
            <div className="p-3 rounded-lg bg-emerald-50 border border-emerald-200 text-xs text-emerald-800 flex items-center space-x-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>{message}</span>
            </div>
          )}

          <div className="flex justify-end pt-2">
            <button
              type="submit"
              disabled={saving}
              className="flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-medium bg-blue-600 text-white hover:bg-blue-700 shadow-xs transition-colors disabled:opacity-50"
            >
              <Save className="w-3.5 h-3.5" />
              <span>{saving ? 'Saving...' : 'Save & Activate Thresholds'}</span>
            </button>
          </div>
        </form>
      </div>

      {/* Detectors Architecture Card */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* SCRFD Settings */}
        <div className="bg-white rounded-xl border border-slate-200/90 p-5 shadow-xs space-y-3">
          <div className="flex items-center space-x-2 pb-2 border-b border-slate-100">
            <Layers className="w-4 h-4 text-blue-600" />
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-800">
              UniFace SCRFD Detector
            </h4>
          </div>

          <div className="space-y-2 text-xs">
            <div className="flex justify-between py-1 border-b border-slate-100">
              <span className="text-slate-500">Confidence Threshold</span>
              <span className="font-mono font-medium text-slate-800">{scrfd.confidence_threshold}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-100">
              <span className="text-slate-500">NMS Threshold</span>
              <span className="font-mono font-medium text-slate-800">{scrfd.nms_threshold}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-100">
              <span className="text-slate-500">Default Input Size</span>
              <span className="font-mono font-medium text-slate-800">{scrfd.input_size.join(' × ')}</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-500">Model Weights</span>
              <span className="font-mono font-medium text-slate-800">{scrfd.model_name || 'SCRFD-10G (default)'}</span>
            </div>
          </div>
        </div>

        {/* RetinaFace Settings */}
        <div className="bg-white rounded-xl border border-slate-200/90 p-5 shadow-xs space-y-3">
          <div className="flex items-center space-x-2 pb-2 border-b border-slate-100">
            <Layers className="w-4 h-4 text-indigo-600" />
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-800">
              UniFace RetinaFace (Fallback)
            </h4>
          </div>

          <div className="space-y-2 text-xs">
            <div className="flex justify-between py-1 border-b border-slate-100">
              <span className="text-slate-500">Confidence Threshold</span>
              <span className="font-mono font-medium text-slate-800">{retinaface.confidence_threshold}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-100">
              <span className="text-slate-500">NMS Threshold</span>
              <span className="font-mono font-medium text-slate-800">{retinaface.nms_threshold}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-100">
              <span className="text-slate-500">Default Input Size</span>
              <span className="font-mono font-medium text-slate-800">{retinaface.input_size.join(' × ')}</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-500">Model Weights</span>
              <span className="font-mono font-medium text-slate-800">{retinaface.model_name || 'MobileNetV2 (default)'}</span>
            </div>
          </div>
        </div>

        {/* Diagnostics & Gaze Architecture */}
        <div className="md:col-span-2 bg-white rounded-xl border border-slate-200/90 p-5 shadow-xs space-y-3">
          <div className="flex items-center space-x-2 pb-2 border-b border-slate-100">
            <Compass className="w-4 h-4 text-emerald-600" />
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-800">
              Photometrics & Gaze Estimation Architecture
            </h4>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <span className="font-semibold text-slate-800 block mb-1">UniFace MobileGaze</span>
              <p className="text-[11px] text-slate-500 leading-relaxed">
                Pre-trained ResNet34 model estimating eye line-of-sight direction (pitch and yaw in degrees). Evaluated directly on face crop without altering biometric quality decisions.
              </p>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <span className="font-semibold text-slate-800 block mb-1">Photometric Diagnostics</span>
              <p className="text-[11px] text-slate-500 leading-relaxed">
                Laplacian variance (edge sharpness) and pixel intensity distribution (mean luminance, underexposure &lt; 15, overexposure &gt; 240) computed directly on detected face crop.
              </p>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <span className="font-semibold text-slate-800 block mb-1">Illumination Asymmetry</span>
              <p className="text-[11px] text-slate-500 leading-relaxed">
                Quantifies relative left-versus-right vertical half luminance differential. Diagnostic signal for lighting asymmetry without uncalibrated automatic rejection.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
