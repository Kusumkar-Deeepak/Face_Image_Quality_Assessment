import React, { useState, useRef } from 'react';
import { 
  UploadCloud, 
  Trash2, 
  RefreshCw, 
  Play, 
  Sliders, 
  Compass, 
  AlertCircle,
  FileImage,
  CheckCircle2,
  Clock,
  Layers,
  Users,
  Eye,
  Sun,
  Activity,
  Maximize2
} from 'lucide-react';
import { analyzeSingleImage } from '../api';
import QualityScoreCard from './QualityScoreCard';
import BoundingBoxViewer from './BoundingBoxViewer';
import { ProcessingStatusBadge } from './StatusBadge';
import ImageViewerModal from './ImageViewerModal';

export default function SingleImageAnalysis({ onAnalysisComplete }) {
  // Upload and file state
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [imageMeta, setImageMeta] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  // Parameter options - Defaulting to RetinaFace fallback provides robust detection for every resolution
  const [preset, setPreset] = useState('standard');
  const [detector, setDetector] = useState('scrfd');
  const [fallbackDetector, setFallbackDetector] = useState('retinaface');
  const [detectionMode, setDetectionMode] = useState('fallback');
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [scrfdInputSize, setScrfdInputSize] = useState(640);
  const [headPose, setHeadPose] = useState(true);
  const [gaze, setGaze] = useState(true);
  const [reviewPolicy, setReviewPolicy] = useState('manual_review');
  const [viewerOpen, setViewerOpen] = useState(false);

  const applyPreset = (mode) => {
    setPreset(mode);
    if (mode === 'standard') {
      setDetector('scrfd');
      setFallbackDetector('retinaface');
      setDetectionMode('fallback');
      setScrfdInputSize(640);
      setConfidenceThreshold(0.5);
    } else if (mode === 'crowd') {
      setDetector('scrfd');
      setFallbackDetector('retinaface');
      setDetectionMode('complementary');
      setScrfdInputSize(960);
      setConfidenceThreshold(0.35);
    }
  };

  // Execution state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [allResults, setAllResults] = useState([]);
  const [activeFaceIdx, setActiveFaceIdx] = useState(0);

  // Handle file selection
  const handleFile = (selectedFile) => {
    if (!selectedFile) return;

    // Validate type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/bmp'];
    if (!validTypes.includes(selectedFile.type)) {
      setError(`Unsupported file type: ${selectedFile.type}. Please upload a JPEG, PNG, or WEBP.`);
      return;
    }

    setError(null);
    setResult(null);
    setAllResults([]);
    setActiveFaceIdx(0);
    setFile(selectedFile);

    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);

    // Read image dimensions
    const img = new Image();
    img.onload = () => {
      setImageMeta({
        width: img.naturalWidth,
        height: img.naturalHeight,
        sizeKb: (selectedFile.size / 1024).toFixed(1),
        sizeMb: (selectedFile.size / (1024 * 1024)).toFixed(2),
        name: selectedFile.name,
      });
    };
    img.src = url;
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleRemoveImage = () => {
    setFile(null);
    setPreviewUrl(null);
    setImageMeta(null);
    setResult(null);
    setAllResults([]);
    setActiveFaceIdx(0);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    try {
      const response = await analyzeSingleImage({
        file,
        detector,
        fallbackDetector: fallbackDetector || null,
        detectionMode,
        confidenceThreshold: Number(confidenceThreshold),
        scrfdInputSize,
        headPose,
        gaze,
        reviewPolicy,
      });

      const resList = response.results || [];
      setAllResults(resList);
      setActiveFaceIdx(0);
      setResult(resList[0] || response.primary_result || null);

      if (onAnalysisComplete) onAnalysisComplete();
    } catch (err) {
      setError(err.message || 'Face analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const selectFace = (idx) => {
    if (allResults[idx]) {
      setActiveFaceIdx(idx);
      setResult(allResults[idx]);
    }
  };

  return (
    <div className="space-y-6">
      {/* Configuration & Parameter Bar */}
      <div className="bg-white rounded-xl border border-slate-200/90 p-4 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 mb-3 border-b border-slate-100 gap-2">
          <div className="flex items-center space-x-2">
            <Sliders className="w-4 h-4 text-blue-600" />
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-700">
              Detector & Evaluation Parameters
            </span>
          </div>
          {/* Preset Buttons */}
          <div className="flex items-center space-x-1.5 bg-slate-100 p-0.5 rounded-lg border border-slate-200">
            <button
              type="button"
              onClick={() => applyPreset('standard')}
              className={`text-[11px] px-2.5 py-1 rounded-md font-medium transition-colors ${
                preset === 'standard'
                  ? 'bg-white text-blue-700 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Standard (Portrait)
            </button>
            <button
              type="button"
              onClick={() => applyPreset('crowd')}
              className={`text-[11px] px-2.5 py-1 rounded-md font-medium transition-colors ${
                preset === 'crowd'
                  ? 'bg-blue-600 text-white shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
              title="Complementary ensemble (SCRFD + RetinaFace) at 960px with 0.35 confidence for crowds and surveillance"
            >
              👥 Crowded Scene (High Recall)
            </button>
            <button
              type="button"
              onClick={() => setPreset('custom')}
              className={`text-[11px] px-2.5 py-1 rounded-md font-medium transition-colors ${
                preset === 'custom'
                  ? 'bg-white text-blue-700 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Custom
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-7 gap-3">
          {/* Review Policy */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Review Policy
            </label>
            <select
              value={reviewPolicy}
              onChange={(e) => setReviewPolicy(e.target.value)}
              className="w-full text-xs rounded-lg border-slate-200 bg-slate-50/50 text-slate-800 p-2 focus:ring-1 focus:ring-blue-500 font-medium"
            >
              <option value="manual_review">Manual review (default)</option>
              <option value="treat_as_pass">Treat as PASS</option>
              <option value="treat_as_reject">Treat as REJECT</option>
            </select>
          </div>

          {/* Detection Mode */}
          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1">
              Detection Mode
            </label>
            <select
              value={detectionMode}
              onChange={(e) => {
                setDetectionMode(e.target.value);
                setPreset('custom');
              }}
              className="w-full text-xs rounded-lg border-slate-200 bg-slate-50/50 text-slate-800 p-2 focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="fallback">Fallback (SCRFD → Retina)</option>
              <option value="complementary">Complementary Ensemble (Merge Both)</option>
            </select>
          </div>

          {/* Primary Detector */}
          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1">
              Primary Detector
            </label>
            <select
              value={detector}
              onChange={(e) => {
                setDetector(e.target.value);
                setPreset('custom');
              }}
              className="w-full text-xs rounded-lg border-slate-200 bg-slate-50/50 text-slate-800 p-2 focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="scrfd">UniFace SCRFD</option>
              <option value="retinaface">UniFace RetinaFace</option>
            </select>
          </div>

          {/* Fallback / Complementary Partner */}
          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1">
              {detectionMode === 'complementary' ? 'Ensemble Partner' : 'Fallback Detector'}
            </label>
            <select
              value={fallbackDetector}
              onChange={(e) => {
                setFallbackDetector(e.target.value);
                setPreset('custom');
              }}
              className="w-full text-xs rounded-lg border-slate-200 bg-slate-50/50 text-slate-800 p-2 focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="retinaface">RetinaFace</option>
              <option value="">None (Primary Only)</option>
              {detector !== 'scrfd' && <option value="scrfd">SCRFD</option>}
            </select>
          </div>

          {/* SCRFD Input Resolution */}
          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1">
              Resolution Scale
            </label>
            <select
              value={scrfdInputSize}
              onChange={(e) => {
                setScrfdInputSize(Number(e.target.value));
                setPreset('custom');
              }}
              className="w-full text-xs rounded-lg border-slate-200 bg-slate-50/50 text-slate-800 p-2 focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value={640}>640×640 (Standard)</option>
              <option value={960}>960×960 (High Recall / Crowd)</option>
              <option value={1280}>1280×1280 (Ultra-high Res)</option>
              <option value={320}>320×320 (Enhanced Low-res)</option>
            </select>
          </div>

          {/* Confidence Threshold */}
          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1">
              Min Confidence
            </label>
            <select
              value={confidenceThreshold}
              onChange={(e) => {
                setConfidenceThreshold(Number(e.target.value));
                setPreset('custom');
              }}
              className="w-full text-xs rounded-lg border-slate-200 bg-slate-50/50 text-slate-800 p-2 focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value={0.5}>0.50 (Standard)</option>
              <option value={0.35}>0.35 (Crowd Balanced)</option>
              <option value={0.25}>0.25 (High Recall)</option>
            </select>
          </div>

          {/* Toggles: Head Pose & MobileGaze */}
          <div className="flex items-center space-x-2 pt-5">
            <label className="flex items-center space-x-1.5 p-1.5 rounded-lg border border-slate-200 bg-slate-50/40 cursor-pointer hover:bg-slate-50">
              <input
                type="checkbox"
                checked={headPose}
                onChange={(e) => setHeadPose(e.target.checked)}
                className="rounded border-slate-300 text-blue-600 focus:ring-blue-500 h-3.5 w-3.5"
              />
              <span className="text-[11px] text-slate-700 font-medium">Pose</span>
            </label>

            <label className="flex items-center space-x-1.5 p-1.5 rounded-lg border border-slate-200 bg-slate-50/40 cursor-pointer hover:bg-slate-50">
              <input
                type="checkbox"
                checked={gaze}
                onChange={(e) => setGaze(e.target.checked)}
                className="rounded border-slate-300 text-blue-600 focus:ring-blue-500 h-3.5 w-3.5"
              />
              <span className="text-[11px] text-slate-700 font-medium">Gaze</span>
            </label>
          </div>
        </div>
      </div>

      {/* Upload Zone & Preview Panel */}
      {!file ? (
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
            dragActive
              ? 'border-blue-500 bg-blue-50/50'
              : 'border-slate-300 hover:border-slate-400 bg-white'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".jpg,.jpeg,.png,.webp,.bmp"
            onChange={(e) => handleFile(e.target.files?.[0])}
            className="hidden"
          />
          <div className="max-w-md mx-auto flex flex-col items-center">
            <div className="p-3 bg-blue-50 text-blue-600 rounded-full mb-3">
              <UploadCloud className="w-8 h-8" />
            </div>
            <h3 className="text-sm font-semibold text-slate-800 mb-1">
              Click to browse or drag and drop image
            </h3>
            <p className="text-xs text-slate-500 mb-3">
              Supports JPEG, JPG, PNG, WEBP from any camera, phone or scanner up to 25 MB.
            </p>
            <span className="inline-flex items-center px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-100 text-slate-700 hover:bg-slate-200">
              Select Image
            </span>
          </div>
        </div>
      ) : (
        /* Image Preview & Actions Bar before analysis */
        <div className="bg-white rounded-xl border border-slate-200/90 p-4 shadow-xs">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-100">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-slate-100 rounded-lg text-slate-600">
                <FileImage className="w-5 h-5" />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-800 truncate max-w-xs sm:max-w-md">
                  {imageMeta?.name || file.name}
                </p>
                <p className="text-xs text-slate-400">
                  {imageMeta ? `${imageMeta.width} × ${imageMeta.height} px • ${imageMeta.sizeKb} KB` : 'Loading metadata...'}
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <button
                onClick={handleRemoveImage}
                disabled={loading}
                className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-600 hover:text-rose-600 hover:bg-rose-50 border border-slate-200 transition-colors"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>Remove</span>
              </button>

              <button
                onClick={handleAnalyze}
                disabled={loading}
                className="flex items-center space-x-2 px-4 py-1.5 rounded-lg text-xs font-medium bg-blue-600 text-white hover:bg-blue-700 shadow-xs transition-colors disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    <span>Analyzing Face...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5 fill-current" />
                    <span>Analyze Face</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Image Preview before analysis execution */}
          {!result && previewUrl && (
            <div className="mt-3 relative bg-slate-900/90 rounded-xl overflow-hidden flex items-center justify-center max-h-[380px] border border-slate-200">
              <img 
                src={previewUrl} 
                alt="Uploaded preview" 
                className="max-h-[360px] w-auto max-w-full object-contain mx-auto" 
              />
              <div className="absolute bottom-2.5 right-2.5 bg-slate-900/80 text-white text-[10px] px-2.5 py-1 rounded-md backdrop-blur-xs font-mono flex items-center space-x-1.5 shadow-sm">
                <div className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
                <span>Ready for analysis</span>
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="mt-3 p-3 rounded-lg bg-rose-50 border border-rose-200 flex items-start space-x-2 text-rose-800 text-xs">
              <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
              <div className="flex-1">{error}</div>
            </div>
          )}
        </div>
      )}

      {/* Multi-Face Selector Strip if image contains multiple faces */}
      {allResults.length > 1 && (
        <div className="bg-white rounded-xl border border-slate-200/90 p-3 shadow-xs flex items-center space-x-3 overflow-x-auto">
          <div className="flex items-center space-x-1.5 text-xs font-semibold text-slate-700 shrink-0">
            <Users className="w-4 h-4 text-blue-600" />
            <span>Multiple Faces Detected ({allResults.length}):</span>
          </div>
          <div className="flex items-center space-x-2">
            {allResults.map((faceItem, idx) => {
              const isSelected = idx === activeFaceIdx;
              return (
                <button
                  key={idx}
                  onClick={() => selectFace(idx)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center space-x-1.5 shrink-0 ${
                    isSelected
                      ? 'bg-blue-600 text-white shadow-xs'
                      : 'bg-slate-100 text-slate-700 hover:bg-slate-200 border border-slate-200'
                  }`}
                >
                  <span>Face #{idx + 1}</span>
                  {faceItem.quality_score !== null && (
                    <span className="font-mono text-[11px] opacity-90">
                      ({faceItem.quality_score.toFixed(3)})
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Analysis Results View */}
      {result && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Column: Image with Bounding Box & Aligned Face Crop */}
            <div className="lg:col-span-7 bg-white rounded-xl border border-slate-200/90 p-5 shadow-xs">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-700">
                  Detection Visualizer {allResults.length > 1 && `(Showing Face #${activeFaceIdx + 1} of ${allResults.length})`}
                </span>
                <div className="flex items-center space-x-2">
                  <button
                    type="button"
                    onClick={() => setViewerOpen(true)}
                    className="p-1 rounded-md text-slate-500 hover:text-blue-600 hover:bg-slate-100 transition-colors flex items-center space-x-1 text-xs"
                    title="Open Fullscreen Lightbox Zoom Viewer"
                  >
                    <Maximize2 className="w-3.5 h-3.5" />
                    <span className="text-[11px] font-medium hidden sm:inline">Zoom Viewer</span>
                  </button>
                  <ProcessingStatusBadge status={result.status} />
                  {result.fallback_used && (
                    <span className="text-[10px] bg-indigo-50 text-indigo-700 font-semibold px-2 py-0.5 rounded-full border border-indigo-200">
                      Fallback Active
                    </span>
                  )}
                </div>
              </div>

              <BoundingBoxViewer
                imageSrc={previewUrl}
                bbox={result.bbox}
                originalWidth={result.image_width || imageMeta?.width}
                originalHeight={result.image_height || imageMeta?.height}
                confidence={result.confidence}
                detector={result.detector}
                faceCropBase64={result.face_crop_base64}
                allFaces={allResults}
                activeFaceIndex={activeFaceIdx}
                onSelectFace={selectFace}
                onZoom={() => setViewerOpen(true)}
              />
            </div>

            {/* Right Column: Quality Score Card & Pose Angles */}
            <div className="lg:col-span-5 space-y-4">
              <QualityScoreCard
                score={result.quality_score}
                decision={result.decision}
                decisionReason={result.decision_reason}
                status={result.status}
              />

              {/* Head-Pose Estimation Card if enabled and ran */}
              {headPose && (
                <div className="bg-white rounded-xl border border-slate-200/90 p-5 shadow-xs">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-2">
                      <Compass className="w-4 h-4 text-slate-600" />
                      <span className="text-xs font-semibold text-slate-800">UniFace Head-Pose Angles</span>
                    </div>
                    <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                      result.pose_status === 'OK' 
                        ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                        : 'bg-slate-100 text-slate-600'
                    }`}>
                      {result.pose_status}
                    </span>
                  </div>

                  {result.pose_status === 'OK' ? (
                    <div className="grid grid-cols-3 gap-2 text-center">
                      <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                        <span className="text-[11px] font-medium text-slate-400 block">Pitch</span>
                        <span className="text-base font-bold text-slate-800 font-mono">
                          {result.pitch !== null ? `${result.pitch.toFixed(1)}°` : '-'}
                        </span>
                      </div>
                      <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                        <span className="text-[11px] font-medium text-slate-400 block">Yaw</span>
                        <span className="text-base font-bold text-slate-800 font-mono">
                          {result.yaw !== null ? `${result.yaw.toFixed(1)}°` : '-'}
                        </span>
                      </div>
                      <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                        <span className="text-[11px] font-medium text-slate-400 block">Roll</span>
                        <span className="text-base font-bold text-slate-800 font-mono">
                          {result.roll !== null ? `${result.roll.toFixed(1)}°` : '-'}
                        </span>
                      </div>
                    </div>
                  ) : (
                    <p className="text-xs text-slate-500 py-2">
                      Head-pose estimation was {result.pose_status.toLowerCase()} for this detection.
                    </p>
                  )}
                </div>
              )}

              {/* UniFace MobileGaze Estimation Card */}
              {gaze && (
                <div className="bg-white rounded-xl border border-slate-200/90 p-5 shadow-xs">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-2">
                      <Eye className="w-4 h-4 text-indigo-600" />
                      <div>
                        <span className="text-xs font-semibold text-slate-800 block">UniFace MobileGaze</span>
                        <span className="text-[10px] text-slate-400 font-normal">Eye line-of-sight direction</span>
                      </div>
                    </div>
                    <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                      result.gaze_status === 'OK' 
                        ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                        : 'bg-slate-100 text-slate-600'
                    }`}>
                      {result.gaze_status || 'DISABLED'}
                    </span>
                  </div>

                  {result.gaze_status === 'OK' ? (
                    <div>
                      <div className="grid grid-cols-2 gap-2 text-center mb-2">
                        <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                          <span className="text-[11px] font-medium text-slate-400 block">Gaze Pitch (Vertical)</span>
                          <span className="text-base font-bold text-slate-800 font-mono">
                            {result.gaze_pitch !== null && result.gaze_pitch !== undefined ? `${result.gaze_pitch.toFixed(1)}°` : '-'}
                          </span>
                        </div>
                        <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                          <span className="text-[11px] font-medium text-slate-400 block">Gaze Yaw (Horizontal)</span>
                          <span className="text-base font-bold text-slate-800 font-mono">
                            {result.gaze_yaw !== null && result.gaze_yaw !== undefined ? `${result.gaze_yaw.toFixed(1)}°` : '-'}
                          </span>
                        </div>
                      </div>
                      <p className="text-[10px] text-slate-400 leading-tight">
                        Note: Gaze direction reflects eye attention and does not alter identity quality scoring.
                      </p>
                    </div>
                  ) : (
                    <p className="text-xs text-slate-500 py-2">
                      Gaze estimation was {result.gaze_status ? result.gaze_status.toLowerCase() : 'disabled'} for this detection.
                    </p>
                  )}
                </div>
              )}

              {/* Photometric & Clarity Diagnostics Card */}
              <div className="bg-white rounded-xl border border-slate-200/90 p-5 shadow-xs">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    <Activity className="w-4 h-4 text-blue-600" />
                    <div>
                      <span className="text-xs font-semibold text-slate-800 block">Photometric & Clarity Signals</span>
                      <span className="text-[10px] text-slate-400 font-normal">Explainable crop diagnostics</span>
                    </div>
                  </div>
                  <span className="text-[10px] bg-slate-100 text-slate-600 font-medium px-2 py-0.5 rounded-full">
                    Advisory Only
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-100">
                    <span className="text-[10px] text-slate-500 font-medium block">Laplacian Sharpness</span>
                    <span className="text-sm font-bold font-mono text-slate-800">
                      {result.sharpness_laplacian_var !== null && result.sharpness_laplacian_var !== undefined 
                        ? result.sharpness_laplacian_var.toFixed(1)
                        : '-'}
                    </span>
                    <span className="text-[10px] text-slate-400 block mt-0.5">Variance (higher = crisper)</span>
                  </div>

                  <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-100">
                    <span className="text-[10px] text-slate-500 font-medium block">Mean Luminance</span>
                    <span className="text-sm font-bold font-mono text-slate-800">
                      {result.mean_luminance !== null && result.mean_luminance !== undefined 
                        ? `${result.mean_luminance.toFixed(1)} / 255`
                        : '-'}
                    </span>
                    <span className="text-[10px] text-slate-400 block mt-0.5">Crop brightness</span>
                  </div>

                  <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-100">
                    <span className="text-[10px] text-slate-500 font-medium block">Exposure Extremes</span>
                    <span className="text-xs font-bold font-mono text-slate-800">
                      {result.underexposure_ratio !== null && result.overexposure_ratio !== null
                        ? `U: ${(result.underexposure_ratio * 100).toFixed(1)}% | O: ${(result.overexposure_ratio * 100).toFixed(1)}%`
                        : '-'}
                    </span>
                    <span className="text-[10px] text-slate-400 block mt-0.5">Under &lt;15 / Over &gt;240</span>
                  </div>

                  <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-100">
                    <span className="text-[10px] text-slate-500 font-medium block">Illumination Asymmetry</span>
                    <span className="text-sm font-bold font-mono text-slate-800">
                      {result.illumination_asymmetry !== null && result.illumination_asymmetry !== undefined
                        ? result.illumination_asymmetry.toFixed(3)
                        : '-'}
                    </span>
                    <span className="text-[10px] text-slate-400 block mt-0.5">|Left - Right| / avg</span>
                  </div>
                </div>
              </div>

              {/* Processing Duration summary card */}
              <div className="bg-white rounded-xl border border-slate-200/90 p-4 shadow-xs flex items-center justify-between text-xs">
                <div className="flex items-center space-x-2 text-slate-600">
                  <Clock className="w-4 h-4 text-slate-400" />
                  <span>Total Inference Latency</span>
                </div>
                <span className="font-mono font-semibold text-slate-800">
                  {Math.round(result.processing_time_ms)} ms
                </span>
              </div>
            </div>
          </div>

          {/* Processing Details Table */}
          <div className="bg-white rounded-xl border border-slate-200/90 p-5 shadow-xs">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-700 mb-3">
              Processing & Diagnostic Details
            </h4>
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left border-collapse">
                <tbody>
                  <tr className="border-b border-slate-100">
                    <td className="py-2.5 font-medium text-slate-500 w-1/3">Processing Status</td>
                    <td className="py-2.5 text-slate-800">
                      <ProcessingStatusBadge status={result.status} />
                    </td>
                  </tr>
                  <tr className="border-b border-slate-100">
                    <td className="py-2.5 font-medium text-slate-500">Detector & Method</td>
                    <td className="py-2.5 font-mono text-slate-800">
                      {result.detector} {result.fallback_used && '(fallback invoked)'}
                    </td>
                  </tr>
                  <tr className="border-b border-slate-100">
                    <td className="py-2.5 font-medium text-slate-500">Bounding Box [x1, y1, x2, y2]</td>
                    <td className="py-2.5 font-mono text-slate-800">
                      {result.bbox ? JSON.stringify(result.bbox.map(c => Math.round(c * 10) / 10)) : 'None'}
                    </td>
                  </tr>
                  <tr className="border-b border-slate-100">
                    <td className="py-2.5 font-medium text-slate-500">Face Area Ratio</td>
                    <td className="py-2.5 font-mono text-slate-800">
                      {result.face_area_ratio !== null ? `${(result.face_area_ratio * 100).toFixed(2)}% of image area` : 'None'}
                    </td>
                  </tr>
                  {result.inter_eye_distance_px !== null && (
                    <tr className="border-b border-slate-100">
                      <td className="py-2.5 font-medium text-slate-500">Inter-Eye Distance</td>
                      <td className="py-2.5 font-mono text-slate-800">
                        {result.inter_eye_distance_px.toFixed(1)} px
                      </td>
                    </tr>
                  )}
                  {result.sharpness_laplacian_var !== null && result.sharpness_laplacian_var !== undefined && (
                    <tr className="border-b border-slate-100">
                      <td className="py-2.5 font-medium text-slate-500">Sharpness (Laplacian Var)</td>
                      <td className="py-2.5 font-mono text-slate-800">
                        {result.sharpness_laplacian_var.toFixed(2)} <span className="text-[11px] text-slate-400 font-sans">(Advisory clarity indicator)</span>
                      </td>
                    </tr>
                  )}
                  {result.mean_luminance !== null && result.mean_luminance !== undefined && (
                    <tr className="border-b border-slate-100">
                      <td className="py-2.5 font-medium text-slate-500">Exposure Statistics</td>
                      <td className="py-2.5 font-mono text-slate-800">
                        Mean: {result.mean_luminance.toFixed(1)}/255 • Underexposed: {(result.underexposure_ratio * 100).toFixed(1)}% • Overexposed: {(result.overexposure_ratio * 100).toFixed(1)}%
                      </td>
                    </tr>
                  )}
                  {result.illumination_asymmetry !== null && result.illumination_asymmetry !== undefined && (
                    <tr className="border-b border-slate-100">
                      <td className="py-2.5 font-medium text-slate-500">Illumination Asymmetry</td>
                      <td className="py-2.5 font-mono text-slate-800">
                        {result.illumination_asymmetry.toFixed(3)} <span className="text-[11px] text-slate-400 font-sans">(Left/Right half relative ratio)</span>
                      </td>
                    </tr>
                  )}
                  {result.gaze_status === 'OK' && result.gaze_pitch !== null && (
                    <tr className="border-b border-slate-100">
                      <td className="py-2.5 font-medium text-slate-500">MobileGaze Direction</td>
                      <td className="py-2.5 font-mono text-slate-800">
                        Pitch: {result.gaze_pitch.toFixed(1)}° • Yaw: {result.gaze_yaw.toFixed(1)}° <span className="text-[11px] text-slate-400 font-sans">(Eye line-of-sight)</span>
                      </td>
                    </tr>
                  )}
                  {result.detection_note && (
                    <tr className="border-b border-slate-100">
                      <td className="py-2.5 font-medium text-slate-500">Detection Note</td>
                      <td className="py-2.5 text-slate-600 font-mono text-[11px]">
                        {result.detection_note}
                      </td>
                    </tr>
                  )}
                  {result.error_reason && (
                    <tr className="border-b border-slate-100 bg-amber-50/50">
                      <td className="py-2.5 font-medium text-amber-800">Rejection / Reason</td>
                      <td className="py-2.5 text-amber-900 font-medium">
                        {result.error_reason}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Lightbox Modal */}
      {viewerOpen && (
        <ImageViewerModal
          isOpen={viewerOpen}
          onClose={() => setViewerOpen(false)}
          fullImageUrl={previewUrl}
          cropImageUrl={result?.face_crop_base64 ? (result.face_crop_base64.startsWith('data:') ? result.face_crop_base64 : `data:image/jpeg;base64,${result.face_crop_base64}`) : null}
          filename={imageMeta?.name || file?.name || 'image.jpg'}
          faces={allResults}
          initialFaceIndex={activeFaceIdx}
          onSelectFace={selectFace}
        />
      )}
    </div>
  );
}
