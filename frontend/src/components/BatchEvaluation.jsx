import React, { useState, useRef, useMemo, useEffect } from 'react';
import { 
  UploadCloud, 
  Trash2, 
  RefreshCw, 
  Play, 
  Sliders, 
  Download, 
  Search, 
  Filter, 
  ArrowUpDown, 
  FileCheck2, 
  FileText, 
  AlertCircle, 
  FileJson, 
  X, 
  ChevronDown, 
  ChevronRight, 
  ScanFace, 
  Sparkles, 
  Compass, 
  Clock, 
  CheckCircle2, 
  Layers, 
  Square, 
  Maximize2,
  Eye,
  Activity,
  FileArchive
} from 'lucide-react';
import { analyzeBatchImages, getExportUrl, getPassedZipUrl, saveBatchResult } from '../api';
import { ProcessingStatusBadge, DecisionBadge } from './StatusBadge';
import QualityScoreCard from './QualityScoreCard';
import BoundingBoxViewer from './BoundingBoxViewer';
import ImageViewerModal from './ImageViewerModal';

const MAX_BATCH_LIMIT = 100;
const CHUNK_SIZE = 2; // Process 2 images per request to prevent RAM spikes (512MB limit) and avoid proxy 502 timeouts

export default function BatchEvaluation({ onBatchComplete }) {
  const [files, setFiles] = useState([]);
  const [filePreviews, setFilePreviews] = useState([]);
  const fileInputRef = useRef(null);

  // Parameter options
  const [preset, setPreset] = useState('standard');
  const [detector, setDetector] = useState('scrfd');
  const [fallbackDetector, setFallbackDetector] = useState('retinaface');
  const [detectionMode, setDetectionMode] = useState('fallback');
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [scrfdInputSize, setScrfdInputSize] = useState(640);
  const [headPose, setHeadPose] = useState(true);
  const [gaze, setGaze] = useState(true);
  const [reviewPolicy, setReviewPolicy] = useState('manual_review'); // 'manual_review' | 'treat_as_pass' | 'treat_as_reject'

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

  // Batch execution & progress state
  const [loading, setLoading] = useState(false);
  const [progressPercent, setProgressPercent] = useState(0);
  const [currentStepText, setCurrentStepText] = useState('');
  const [currentFileName, setCurrentFileName] = useState('');
  const [processedCount, setProcessedCount] = useState(0);
  const [activeStage, setActiveStage] = useState('idle'); // 'uploading' | 'validating' | 'detecting' | 'scoring' | 'complete'
  const [batchStartTime, setBatchStartTime] = useState(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [batchId, setBatchId] = useState(null);
  const [batchResults, setBatchResults] = useState([]);
  const [error, setError] = useState(null);

  // Pagination & Virtualization state
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);

  // Lightbox Modal Viewer state
  const [viewerOpen, setViewerOpen] = useState(false);
  const [viewerIndex, setViewerIndex] = useState(0);

  // Row details expanded state (Set of row indexes)
  const [expandedRows, setExpandedRows] = useState(new Set());

  // Table filtering and sorting
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [sortField, setSortField] = useState('quality_score');
  const [sortDirection, setSortDirection] = useState('desc');

  // Cancel flag ref
  const cancelRequestedRef = useRef(false);

  // Timer for elapsed seconds
  useEffect(() => {
    let interval = null;
    if (loading && batchStartTime) {
      interval = setInterval(() => {
        setElapsedSeconds(Math.floor((Date.now() - batchStartTime) / 1000));
      }, 500);
    } else if (!loading) {
      clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [loading, batchStartTime]);

  // Full-resolution object URLs lookup map (cached & revoked on clear)
  const fileUrlMapRef = useRef(new Map());

  // Helper to obtain or cache full-resolution image URL for BoundingBoxViewer
  const getFullImageUrl = (fileOrName) => {
    if (!fileOrName) return null;
    if (typeof fileOrName === 'string') {
      const matched = files.find(f => f.name === fileOrName);
      if (!matched) return null;
      if (!fileUrlMapRef.current.has(matched)) {
        fileUrlMapRef.current.set(matched, URL.createObjectURL(matched));
      }
      return fileUrlMapRef.current.get(matched);
    }
    if (!fileUrlMapRef.current.has(fileOrName)) {
      fileUrlMapRef.current.set(fileOrName, URL.createObjectURL(fileOrName));
    }
    return fileUrlMapRef.current.get(fileOrName);
  };

  // Fast async thumbnail generator using background worker thread or offscreen canvas
  // Generates lightweight 100x100 JPEG (~8 KB) to completely eliminate batch queue scroll stutter
  const generateFastThumbnail = async (file) => {
    try {
      if (typeof window !== 'undefined' && 'createImageBitmap' in window) {
        const bitmap = await createImageBitmap(file, {
          resizeWidth: 100,
          resizeHeight: 100,
          resizeQuality: 'low',
        });
        const canvas = document.createElement('canvas');
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(bitmap, 0, 0);
        bitmap.close();
        return canvas.toDataURL('image/jpeg', 0.65);
      }
    } catch {
      // Fallback
    }

    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          const maxDim = 100;
          let w = img.width;
          let h = img.height;
          if (w > h) {
            h = Math.round((h * maxDim) / w);
            w = maxDim;
          } else {
            w = Math.round((w * maxDim) / h);
            h = maxDim;
          }
          const canvas = document.createElement('canvas');
          canvas.width = Math.max(1, w);
          canvas.height = Math.max(1, h);
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0, w, h);
          resolve(canvas.toDataURL('image/jpeg', 0.65));
        };
        img.onerror = () => resolve(null);
        img.src = e.target.result;
      };
      reader.onerror = () => resolve(null);
      reader.readAsDataURL(file);
    });
  };

  // Revoke object URLs on component unmount
  useEffect(() => {
    return () => {
      fileUrlMapRef.current.forEach(url => URL.revokeObjectURL(url));
      fileUrlMapRef.current.clear();
    };
  }, []);

  // Handle file additions up to 100 images
  const handleFilesAdded = (newFiles) => {
    if (!newFiles || newFiles.length === 0) return;
    const validFiles = Array.from(newFiles).filter(f => 
      ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/bmp'].includes(f.type)
    );

    const availableSlots = MAX_BATCH_LIMIT - files.length;
    if (availableSlots <= 0) {
      setError(`Batch maximum limit of ${MAX_BATCH_LIMIT} images reached.`);
      return;
    }

    const filesToAdd = validFiles.slice(0, availableSlots);
    if (validFiles.length > availableSlots) {
      setError(`Only added ${availableSlots} images to respect the ${MAX_BATCH_LIMIT} images/batch maximum.`);
    } else {
      setError(null);
    }

    // Pre-cache full object URLs in lookup map
    filesToAdd.forEach(file => {
      if (!fileUrlMapRef.current.has(file)) {
        fileUrlMapRef.current.set(file, URL.createObjectURL(file));
      }
    });

    setFiles(prev => [...prev, ...filesToAdd]);

    // Create immediate placeholder queue items so UI updates in 0ms
    const newItems = filesToAdd.map((file, i) => ({
      id: `${file.name}_${Date.now()}_${i}`,
      name: file.name,
      size: (file.size / 1024).toFixed(1) + ' KB',
      fileRef: file,
      thumbUrl: null,
    }));
    setFilePreviews(prev => [...prev, ...newItems]);

    // Asynchronously generate 100x100 thumbnails in small concurrent chunks so scrolling never lags
    (async () => {
      const chunkSize = 6;
      for (let i = 0; i < filesToAdd.length; i += chunkSize) {
        const slice = filesToAdd.slice(i, i + chunkSize);
        const thumbs = await Promise.all(slice.map(f => generateFastThumbnail(f)));
        setFilePreviews(prev => {
          const updated = [...prev];
          slice.forEach((f, idx) => {
            const foundIdx = updated.findIndex(p => p.fileRef === f);
            if (foundIdx !== -1) {
              updated[foundIdx] = {
                ...updated[foundIdx],
                thumbUrl: thumbs[idx],
              };
            }
          });
          return updated;
        });
      }
    })();
  };

  const handleRemoveFile = (index) => {
    const fileToRemove = files[index];
    if (fileToRemove && fileUrlMapRef.current.has(fileToRemove)) {
      URL.revokeObjectURL(fileUrlMapRef.current.get(fileToRemove));
      fileUrlMapRef.current.delete(fileToRemove);
    }
    setFiles(prev => prev.filter((_, i) => i !== index));
    setFilePreviews(prev => prev.filter((_, i) => i !== index));
  };

  const handleClearAll = () => {
    fileUrlMapRef.current.forEach(url => URL.revokeObjectURL(url));
    fileUrlMapRef.current.clear();
    setFiles([]);
    setFilePreviews([]);
    setBatchResults([]);
    setBatchId(null);
    setError(null);
    setExpandedRows(new Set());
    setProgressPercent(0);
    setProcessedCount(0);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const toggleRowExpanded = (rowIdx) => {
    setExpandedRows(prev => {
      const next = new Set(prev);
      if (next.has(rowIdx)) next.delete(rowIdx);
      else next.add(rowIdx);
      return next;
    });
  };

  const toggleExpandAll = () => {
    if (expandedRows.size === filteredResults.length) {
      setExpandedRows(new Set());
    } else {
      setExpandedRows(new Set(filteredResults.map((_, i) => i)));
    }
  };

  // High-speed Chunked Parallel Batch Processor
  const handleStartBatch = async () => {
    if (files.length === 0) return;
    setLoading(true);
    setError(null);
    cancelRequestedRef.current = false;
    setBatchStartTime(Date.now());
    setElapsedSeconds(0);
    setBatchResults([]);
    setExpandedRows(new Set());
    setProcessedCount(0);
    setProgressPercent(2);

    const generatedBatchId = crypto.randomUUID();
    setBatchId(generatedBatchId);

    const accumulatedResults = [];
    const totalFiles = files.length;

    // Split files into chunks of CHUNK_SIZE
    const chunks = [];
    for (let i = 0; i < totalFiles; i += CHUNK_SIZE) {
      chunks.push(files.slice(i, i + CHUNK_SIZE));
    }

    try {
      for (let chunkIdx = 0; chunkIdx < chunks.length; chunkIdx++) {
        if (cancelRequestedRef.current) {
          setError('Batch evaluation cancelled by user.');
          break;
        }

        const chunk = chunks[chunkIdx];
        const chunkStart = chunkIdx * CHUNK_SIZE;

        // Stage: Uploading & Initializing
        setActiveStage('uploading');
        setCurrentFileName(chunk.map(f => f.name).join(', '));
        setCurrentStepText(`Uploading chunk ${chunkIdx + 1} of ${chunks.length} (${chunk.length} images)...`);
        
        // Advance progress
        const basePercent = Math.round((chunkStart / totalFiles) * 90);
        setProgressPercent(Math.max(5, basePercent));

        // Short timeout for visual stage indicator
        await new Promise(r => setTimeout(r, 60));

        // Stage: Model Detection & Quality Scoring
        setActiveStage('detecting');
        setCurrentStepText(`Detecting faces with ${detector.toUpperCase()} & evaluating eDifFIQA-S quality...`);

        let chunkResults = [];
        try {
          const response = await analyzeBatchImages({
            files: chunk,
            detector,
            fallbackDetector: fallbackDetector || null,
            detectionMode,
            confidenceThreshold: Number(confidenceThreshold),
            scrfdInputSize,
            headPose,
            gaze,
            reviewPolicy,
          });
          chunkResults = response?.results || [];
        } catch (chunkErr) {
          console.warn(`Chunk ${chunkIdx + 1} encountered an error:`, chunkErr);
          // Gracefully fallback to per-image error records so the batch can proceed without stopping
          chunkResults = chunk.map((file, i) => ({
            request_id: `err-${Date.now()}-${i}`,
            image_name: file.name,
            image_width: null,
            image_height: null,
            detector: detector,
            detection_mode: detectionMode,
            fallback_used: false,
            face_index: null,
            confidence: null,
            bbox: null,
            quality_score: null,
            status: 'READ_ERROR',
            decision: 'NOT_EVALUATED',
            gate_decision: 'ERROR',
            effective_decision: 'ERROR',
            decision_reason: `Batch chunk error: ${chunkErr.message || 'Server timeout or memory limit'}.`,
            error_reason: chunkErr.message || 'Chunk network or gateway timeout',
            detection_note: 'Request timed out or encountered cloud container memory limit',
            processing_time_ms: 0,
          }));
        }

        // Stage: Processing finished for this chunk
        setActiveStage('scoring');
        const chunkResultsWithUrls = chunkResults.map((r, i) => {
          const matchedFile = chunk.find(f => f.name === r.image_name) || chunk[i];
          const fileUrl = matchedFile ? getFullImageUrl(matchedFile) : null;
          return {
            ...r,
            fileUrl,
          };
        });
        accumulatedResults.push(...chunkResultsWithUrls);
        
        // Update live table results immediately
        setBatchResults([...accumulatedResults]);
        setProcessedCount(accumulatedResults.length);
        
        const currentPercent = Math.min(95, Math.round((accumulatedResults.length / totalFiles) * 100));
        setProgressPercent(currentPercent);
      }

      // Final Stage: Save consolidated batch to backend for 1-click CSV/JSON export
      setActiveStage('complete');
      setCurrentStepText('Finalizing batch reports & session metrics...');
      setProgressPercent(100);

      const passCount = accumulatedResults.filter(r => (r.effective_decision || r.gate_decision) === 'PASS').length;
      const reviewCount = accumulatedResults.filter(r => (r.effective_decision || r.gate_decision) === 'REVIEW').length;
      const rejectCount = accumulatedResults.filter(r => (r.effective_decision || r.gate_decision) === 'REJECT').length;
      const noFaceCount = accumulatedResults.filter(r => r.status === 'NO_FACE').length;
      const errorCount = accumulatedResults.filter(r => ['READ_ERROR', 'ALIGNMENT_ERROR', 'MODEL_ERROR'].includes(r.status)).length;
      const totalFaces = accumulatedResults.filter(r => r.status === 'OK').length;

      const consolidatedBatch = {
        batch_id: generatedBatchId,
        total_images: totalFiles,
        processed_images: accumulatedResults.length,
        successful_evaluations: totalFaces,
        failures: totalFiles - totalFaces,
        pass_count: passCount,
        review_count: reviewCount,
        reject_count: rejectCount,
        no_face_count: noFaceCount,
        error_count: errorCount,
        total_faces: totalFaces,
        review_policy: reviewPolicy,
        results: accumulatedResults,
        created_at: new Date().toISOString(),
      };

      try {
        await saveBatchResult(consolidatedBatch);
      } catch (saveErr) {
        console.warn('Batch registration notice:', saveErr);
      }

      if (onBatchComplete) onBatchComplete();
    } catch (err) {
      setError(err.message || 'Batch evaluation error');
    } finally {
      setLoading(false);
      setActiveStage('idle');
    }
  };

  const handleCancel = () => {
    cancelRequestedRef.current = true;
    setCurrentStepText('Stopping batch worker...');
  };

  // Helper to compute effective decision respecting reviewPolicy
  const getItemEffectiveDecision = (item) => {
    let eff = item.effective_decision || item.gate_decision || (item.status === 'OK' ? (item.decision === 'ACCEPT' || item.decision === 'PASS' ? 'PASS' : item.decision) : 'ERROR');
    if (eff === 'REVIEW') {
      if (reviewPolicy === 'treat_as_pass') return 'PASS';
      if (reviewPolicy === 'treat_as_reject') return 'REJECT';
    }
    return eff;
  };

  // Filter and sort results
  const filteredResults = useMemo(() => {
    if (!batchResults || batchResults.length === 0) return [];

    return batchResults
      .filter(item => {
        const matchesQuery = item.image_name.toLowerCase().includes(searchQuery.toLowerCase());
        const eff = getItemEffectiveDecision(item);

        let matchesStatus = true;
        if (statusFilter !== 'ALL') {
          if (statusFilter === 'PASS') matchesStatus = (eff === 'PASS');
          else if (statusFilter === 'REVIEW') matchesStatus = (eff === 'REVIEW');
          else if (statusFilter === 'REJECT') matchesStatus = (eff === 'REJECT');
          else if (statusFilter === 'NO_FACE') matchesStatus = (item.status === 'NO_FACE' || eff === 'NO_FACE');
          else if (statusFilter === 'ERROR') matchesStatus = (['READ_ERROR', 'ALIGNMENT_ERROR', 'MODEL_ERROR'].includes(item.status) || eff === 'ERROR');
          else matchesStatus = (item.status === statusFilter || eff === statusFilter);
        }

        return matchesQuery && matchesStatus;
      })
      .sort((a, b) => {
        let valA = a[sortField];
        let valB = b[sortField];

        if (valA === null || valA === undefined) valA = sortDirection === 'desc' ? -Infinity : Infinity;
        if (valB === null || valB === undefined) valB = sortDirection === 'desc' ? -Infinity : Infinity;

        if (typeof valA === 'string') {
          return sortDirection === 'asc' ? valA.localeCompare(valB) : valB.localeCompare(valA);
        }
        return sortDirection === 'asc' ? valA - valB : valB - valA;
      });
  }, [batchResults, searchQuery, statusFilter, sortField, sortDirection, reviewPolicy]);

  // Reset page when filter or search changes
  useEffect(() => {
    setCurrentPage(1);
  }, [searchQuery, statusFilter, pageSize]);

  // Pagination chunking
  const totalPages = Math.max(1, Math.ceil(filteredResults.length / pageSize));
  const paginatedResults = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredResults.slice(start, start + pageSize);
  }, [filteredResults, currentPage, pageSize]);

  // Derived metrics from actual batch results covering all 6 discrete states
  const stats = useMemo(() => {
    if (batchResults.length === 0) return null;
    const totalFaces = batchResults.filter(r => r.status === 'OK').length;
    let passCount = 0;
    let reviewCount = 0;
    let rejectCount = 0;
    let noFaceCount = 0;
    let errorCount = 0;

    batchResults.forEach(r => {
      const eff = getItemEffectiveDecision(r);
      if (eff === 'PASS') passCount++;
      else if (eff === 'REVIEW') reviewCount++;
      else if (eff === 'REJECT') rejectCount++;
      else if (eff === 'NO_FACE' || r.status === 'NO_FACE') noFaceCount++;
      else if (eff === 'ERROR' || ['READ_ERROR', 'ALIGNMENT_ERROR', 'MODEL_ERROR'].includes(r.status)) errorCount++;
    });

    const scores = batchResults
      .map(r => r.quality_score)
      .filter(s => s !== null && s !== undefined && !isNaN(s));
    const avgScore = scores.length > 0 ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(4) : '-';
    const totalLatencyMs = batchResults.reduce((acc, r) => acc + (r.processing_time_ms || 0), 0);

    return {
      totalImages: files.length,
      processedImages: processedCount,
      totalFaces,
      passCount,
      reviewCount,
      rejectCount,
      noFaceCount,
      errorCount,
      avgScore,
      scoredCount: scores.length,
      totalLatencyMs,
    };
  }, [batchResults, reviewPolicy, files.length, processedCount]);

  const toggleSort = (field) => {
    if (sortField === field) {
      setSortDirection(prev => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  // Speed and ETA calculation
  const processingSpeed = useMemo(() => {
    if (elapsedSeconds <= 0 || processedCount <= 0) return null;
    return (processedCount / elapsedSeconds).toFixed(1);
  }, [elapsedSeconds, processedCount]);

  const etaSeconds = useMemo(() => {
    if (!processingSpeed || processingSpeed <= 0 || !files.length) return null;
    const remaining = files.length - processedCount;
    if (remaining <= 0) return 0;
    return Math.round(remaining / parseFloat(processingSpeed));
  }, [processingSpeed, files.length, processedCount]);

  // Helper for reference tier pill in diagnostics
  const getScoreTierPill = (score) => {
    if (score === null || score === undefined || isNaN(score)) return null;
    const num = Number(score);
    if (num >= 0.80) {
      return <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">Tier 1: High Quality</span>;
    }
    if (num >= 0.65) {
      return <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200">Tier 2: Standard Usable</span>;
    }
    if (num >= 0.45) {
      return <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200">Tier 3: Degraded</span>;
    }
    return <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-rose-50 text-rose-700 border border-rose-200">Tier 4: Poor</span>;
  };

  return (
    <div className="space-y-6">
      {/* Parameter Controls */}
      <div className="bg-white rounded-xl border border-slate-200/90 p-4 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 mb-3 border-b border-slate-100 gap-2">
          <div className="flex items-center space-x-2">
            <Sliders className="w-4 h-4 text-blue-600" />
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-700">
              High-Capacity Batch Settings (Up to 100 Images)
            </span>
          </div>
          {/* Preset Buttons */}
          <div className="flex items-center space-x-1.5 bg-slate-100 p-0.5 rounded-lg border border-slate-200">
            <button
              type="button"
              disabled={loading}
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
              disabled={loading}
              onClick={() => applyPreset('crowd')}
              className={`text-[11px] px-2.5 py-1 rounded-md font-medium transition-colors ${
                preset === 'crowd'
                  ? 'bg-blue-600 text-white shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
              title="Complementary ensemble (SCRFD + RetinaFace) at 960px with 0.35 confidence for crowded multi-face images"
            >
              👥 Crowded Scene (High Recall)
            </button>
            <button
              type="button"
              disabled={loading}
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
              disabled={loading}
              className="w-full text-xs rounded-lg border-slate-200 bg-slate-50/50 text-slate-800 p-2 focus:ring-1 focus:ring-blue-500 font-medium"
            >
              <option value="manual_review">Manual review (default)</option>
              <option value="treat_as_pass">Treat as PASS</option>
              <option value="treat_as_reject">Treat as REJECT</option>
            </select>
          </div>

          {/* Detection Mode */}
          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1">Detection Mode</label>
            <select
              value={detectionMode}
              onChange={(e) => {
                setDetectionMode(e.target.value);
                setPreset('custom');
              }}
              disabled={loading}
              className="w-full text-xs rounded-lg border-slate-200 bg-slate-50/50 text-slate-800 p-2 focus:ring-1 focus:ring-blue-500"
            >
              <option value="fallback">Fallback (SCRFD → Retina)</option>
              <option value="complementary">Complementary Ensemble (Merge Both)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1">Primary Detector</label>
            <select
              value={detector}
              onChange={(e) => {
                setDetector(e.target.value);
                setPreset('custom');
              }}
              disabled={loading}
              className="w-full text-xs rounded-lg border-slate-200 bg-slate-50/50 text-slate-800 p-2 focus:ring-1 focus:ring-blue-500"
            >
              <option value="scrfd">UniFace SCRFD</option>
              <option value="retinaface">UniFace RetinaFace</option>
            </select>
          </div>

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
              disabled={loading}
              className="w-full text-xs rounded-lg border-slate-200 bg-slate-50/50 text-slate-800 p-2 focus:ring-1 focus:ring-blue-500"
            >
              <option value="retinaface">RetinaFace</option>
              <option value="">None (Primary Only)</option>
              {detector !== 'scrfd' && <option value="scrfd">SCRFD</option>}
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1">Resolution Scale</label>
            <select
              value={scrfdInputSize}
              onChange={(e) => {
                setScrfdInputSize(Number(e.target.value));
                setPreset('custom');
              }}
              disabled={loading}
              className="w-full text-xs rounded-lg border-slate-200 bg-slate-50/50 text-slate-800 p-2 focus:ring-1 focus:ring-blue-500"
            >
              <option value={640}>640×640 (Standard)</option>
              <option value={960}>960×960 (High Recall / Crowd)</option>
              <option value={1280}>1280×1280 (Ultra-high Res)</option>
              <option value={320}>320×320 (Enhanced Low-res)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1">Min Confidence</label>
            <select
              value={confidenceThreshold}
              onChange={(e) => {
                setConfidenceThreshold(Number(e.target.value));
                setPreset('custom');
              }}
              disabled={loading}
              className="w-full text-xs rounded-lg border-slate-200 bg-slate-50/50 text-slate-800 p-2 focus:ring-1 focus:ring-blue-500"
            >
              <option value={0.5}>0.50 (Standard)</option>
              <option value={0.35}>0.35 (Crowd Balanced)</option>
              <option value={0.25}>0.25 (High Recall)</option>
            </select>
          </div>

          <div className="flex items-center space-x-2 pt-5">
            <label className="flex items-center space-x-1.5 p-1.5 rounded-lg border border-slate-200 bg-slate-50/40 cursor-pointer hover:bg-slate-50">
              <input
                type="checkbox"
                checked={headPose}
                onChange={(e) => setHeadPose(e.target.checked)}
                disabled={loading}
                className="rounded border-slate-300 text-blue-600 focus:ring-blue-500 h-3.5 w-3.5"
              />
              <span className="text-[11px] text-slate-700 font-medium">Pose</span>
            </label>

            <label className="flex items-center space-x-1.5 p-1.5 rounded-lg border border-slate-200 bg-slate-50/40 cursor-pointer hover:bg-slate-50">
              <input
                type="checkbox"
                checked={gaze}
                onChange={(e) => setGaze(e.target.checked)}
                disabled={loading}
                className="rounded border-slate-300 text-blue-600 focus:ring-blue-500 h-3.5 w-3.5"
              />
              <span className="text-[11px] text-slate-700 font-medium">Gaze</span>
            </label>
          </div>
        </div>
      </div>

      {/* Upload Zone & Queued Thumbnails */}
      <div className="bg-white rounded-xl border border-slate-200/90 p-5 shadow-xs">
        <div
          onClick={() => !loading && fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
          onDrop={(e) => {
            e.preventDefault();
            e.stopPropagation();
            if (!loading && e.dataTransfer.files) handleFilesAdded(e.dataTransfer.files);
          }}
          className={`border-2 border-dashed rounded-xl p-6 text-center transition-all ${
            loading 
              ? 'border-slate-200 bg-slate-50/50 cursor-not-allowed opacity-60' 
              : 'border-slate-300 hover:border-blue-500 hover:bg-blue-50/30 cursor-pointer'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".jpg,.jpeg,.png,.webp,.bmp"
            onChange={(e) => handleFilesAdded(e.target.files)}
            disabled={loading}
            className="hidden"
          />
          <div className="flex flex-col items-center">
            <div className="p-2.5 bg-blue-50 text-blue-600 rounded-full mb-2">
              <UploadCloud className="w-7 h-7" />
            </div>
            <p className="text-xs font-semibold text-slate-800">
              Click to select up to 100 face images or drag and drop
            </p>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Supports JPEG, JPG, PNG, WEBP, and BMP batches
            </p>
          </div>
        </div>

        {/* Selected files preview header & action buttons */}
        {filePreviews.length > 0 && (
          <div className="mt-4 pt-4 border-t border-slate-100">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
              <div className="flex items-center space-x-2">
                <span className="text-xs font-semibold text-slate-800">
                  Batch Queue ({filePreviews.length} / {MAX_BATCH_LIMIT} images)
                </span>
                <span className="text-[10px] bg-slate-100 text-slate-600 px-2 py-0.5 rounded-full font-mono font-medium">
                  Chunk Size: {CHUNK_SIZE}
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <button
                  onClick={handleClearAll}
                  disabled={loading}
                  className="text-xs text-slate-500 hover:text-rose-600 font-medium transition-colors px-2.5 py-1.5 rounded-lg hover:bg-rose-50"
                >
                  Clear Queue
                </button>
                {loading ? (
                  <button
                    onClick={handleCancel}
                    className="flex items-center space-x-1.5 px-4 py-1.5 rounded-lg text-xs font-medium bg-rose-600 text-white hover:bg-rose-700 shadow-xs transition-colors"
                  >
                    <Square className="w-3.5 h-3.5 fill-current" />
                    <span>Stop Evaluation</span>
                  </button>
                ) : (
                  <button
                    onClick={handleStartBatch}
                    className="flex items-center space-x-1.5 px-4 py-1.5 rounded-lg text-xs font-medium bg-blue-600 text-white hover:bg-blue-700 shadow-xs transition-colors"
                  >
                    <Play className="w-3.5 h-3.5 fill-current" />
                    <span>Start Batch Analysis ({filePreviews.length})</span>
                  </button>
                )}
              </div>
            </div>

            {/* Thumbnail scroll strip */}
            <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-10 gap-2 max-h-52 overflow-y-auto p-1.5 bg-slate-50/50 rounded-lg border border-slate-100 will-change-scroll overscroll-contain">
              {filePreviews.map((item, idx) => (
                <div key={item.id || idx} className="relative group bg-white rounded-lg p-1.5 border border-slate-200/90 shadow-2xs">
                  <div className="w-full aspect-square bg-slate-100 rounded overflow-hidden flex items-center justify-center">
                    {item.thumbUrl ? (
                      <img 
                        src={item.thumbUrl} 
                        alt={item.name} 
                        loading="lazy"
                        decoding="async"
                        className="w-full h-full object-cover" 
                      />
                    ) : (
                      <div className="flex flex-col items-center justify-center p-1 text-slate-300">
                        <RefreshCw className="w-3.5 h-3.5 animate-spin text-blue-400 mb-1" />
                        <span className="text-[9px] text-slate-400 font-mono">Loading</span>
                      </div>
                    )}
                  </div>
                  <p className="text-[10px] text-slate-700 truncate mt-1 font-medium" title={item.name}>{item.name}</p>
                  <p className="text-[9px] text-slate-400">{item.size}</p>
                  {!loading && (
                    <button
                      onClick={() => handleRemoveFile(idx)}
                      className="absolute top-1 right-1 p-0.5 bg-white text-slate-400 hover:text-rose-600 rounded-full shadow-xs opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Live Multi-Stage Animated Progress Bar */}
        {loading && (
          <div className="mt-4 p-4 rounded-xl bg-blue-50/60 border border-blue-200/80 space-y-3">
            {/* Stage Indicators */}
            <div className="grid grid-cols-3 gap-2 text-center text-xs">
              <div className={`p-2 rounded-lg border transition-all ${
                activeStage === 'uploading' 
                  ? 'bg-blue-600 text-white border-blue-600 shadow-xs' 
                  : 'bg-white/80 text-slate-600 border-slate-200'
              }`}>
                <div className="flex items-center justify-center space-x-1.5">
                  <UploadCloud className="w-3.5 h-3.5" />
                  <span className="font-semibold">1. Uploading Chunk</span>
                </div>
              </div>

              <div className={`p-2 rounded-lg border transition-all ${
                activeStage === 'detecting' 
                  ? 'bg-blue-600 text-white border-blue-600 shadow-xs' 
                  : 'bg-white/80 text-slate-600 border-slate-200'
              }`}>
                <div className="flex items-center justify-center space-x-1.5">
                  <ScanFace className="w-3.5 h-3.5" />
                  <span className="font-semibold">2. Face Detection</span>
                </div>
              </div>

              <div className={`p-2 rounded-lg border transition-all ${
                activeStage === 'scoring' 
                  ? 'bg-blue-600 text-white border-blue-600 shadow-xs' 
                  : 'bg-white/80 text-slate-600 border-slate-200'
              }`}>
                <div className="flex items-center justify-center space-x-1.5">
                  <Sparkles className="w-3.5 h-3.5" />
                  <span className="font-semibold">3. eDifFIQA Scoring</span>
                </div>
              </div>
            </div>

            {/* Progress status & percentage */}
            <div className="flex items-center justify-between text-xs text-slate-700">
              <div className="flex items-center space-x-2 truncate">
                <RefreshCw className="w-3.5 h-3.5 animate-spin text-blue-600 shrink-0" />
                <span className="font-medium truncate">{currentStepText}</span>
              </div>
              <span className="font-mono font-bold text-blue-700 shrink-0 ml-2">
                {progressPercent}% ({processedCount}/{files.length})
              </span>
            </div>

            {/* Smooth Progress Bar */}
            <div className="w-full bg-slate-200/80 rounded-full h-2.5 overflow-hidden">
              <div 
                className="bg-blue-600 h-full rounded-full transition-all duration-300 ease-out" 
                style={{ width: `${progressPercent}%` }} 
              />
            </div>

            {/* Real-time Processing Stats Ticker */}
            <div className="flex flex-wrap items-center justify-between text-[11px] text-slate-500 pt-1">
              <div className="flex items-center space-x-3">
                <span className="flex items-center space-x-1">
                  <Clock className="w-3 h-3 text-slate-400" />
                  <span>Elapsed: <strong>{elapsedSeconds}s</strong></span>
                </span>
                {processingSpeed && (
                  <span>Speed: <strong>{processingSpeed} img/sec</strong></span>
                )}
                {etaSeconds !== null && etaSeconds > 0 && (
                  <span>Estimated remaining: <strong>~{etaSeconds}s</strong></span>
                )}
              </div>
              <span className="truncate max-w-xs text-slate-400 italic font-mono">
                {currentFileName ? `Active: ${currentFileName}` : ''}
              </span>
            </div>
          </div>
        )}

        {error && (
          <div className="mt-3 p-3 rounded-lg bg-rose-50 border border-rose-200 text-xs text-rose-800 flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Batch Results & Reports Section */}
      {batchResults.length > 0 && stats && (
        <div className="space-y-4">
          {/* Summary Metric Cards (6 Discrete States + Latency) */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <div className="bg-white rounded-xl border border-slate-200/90 p-3.5 shadow-xs">
              <span className="text-[11px] font-medium text-slate-500 block">Uploaded / Faces</span>
              <div className="flex items-baseline space-x-1 mt-0.5">
                <span className="text-xl font-bold text-slate-800">{stats.totalImages}</span>
                <span className="text-xs text-slate-400 font-normal">img / {stats.totalFaces} faces</span>
              </div>
              <span className="text-[10px] text-slate-400 block mt-1">
                {stats.processedImages} images processed
              </span>
            </div>

            <div className="bg-white rounded-xl border border-emerald-200/80 bg-emerald-50/20 p-3.5 shadow-xs">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-semibold text-emerald-700 block">PASS</span>
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              </div>
              <span className="text-xl font-bold text-emerald-700 block mt-0.5">{stats.passCount}</span>
              <span className="text-[10px] text-emerald-600 font-medium block mt-1">
                {stats.totalFaces > 0 ? `${Math.round((stats.passCount / stats.totalFaces) * 100)}% of faces` : '0%'}
              </span>
            </div>

            <div className="bg-white rounded-xl border border-amber-200/80 bg-amber-50/20 p-3.5 shadow-xs">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-semibold text-amber-700 block">REVIEW</span>
                <AlertCircle className="w-3.5 h-3.5 text-amber-600" />
              </div>
              <span className="text-xl font-bold text-amber-700 block mt-0.5">{stats.reviewCount}</span>
              <span className="text-[10px] text-amber-600 font-medium block mt-1">
                Manual review queue
              </span>
            </div>

            <div className="bg-white rounded-xl border border-rose-200/80 bg-rose-50/20 p-3.5 shadow-xs">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-semibold text-rose-700 block">REJECT</span>
                <X className="w-3.5 h-3.5 text-rose-600" />
              </div>
              <span className="text-xl font-bold text-rose-700 block mt-0.5">{stats.rejectCount}</span>
              <span className="text-[10px] text-rose-600 font-medium block mt-1">
                Below threshold
              </span>
            </div>

            <div className="bg-white rounded-xl border border-slate-200/90 p-3.5 shadow-xs">
              <span className="text-[11px] font-medium text-slate-500 block">NO_FACE / ERROR</span>
              <span className="text-xl font-bold text-slate-700 block mt-0.5">
                {stats.noFaceCount + stats.errorCount}
              </span>
              <span className="text-[10px] text-slate-400 block mt-1">
                {stats.noFaceCount} miss, {stats.errorCount} err
              </span>
            </div>

            <div className="bg-white rounded-xl border border-slate-200/90 p-3.5 shadow-xs">
              <span className="text-[11px] font-medium text-blue-600 block">Avg Score & Latency</span>
              <span className="text-xl font-bold text-slate-800 font-mono block mt-0.5">{stats.avgScore}</span>
              <span className="text-[10px] text-slate-400 block mt-1">
                Total: {(stats.totalLatencyMs / 1000).toFixed(1)}s
              </span>
            </div>
          </div>

          {/* Table Container with Search, Filter & Exports */}
          <div className="bg-white rounded-xl border border-slate-200/90 shadow-xs overflow-hidden">
            <div className="p-4 border-b border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="flex items-center space-x-2 flex-1 max-w-md">
                <div className="relative w-full">
                  <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-2.5" />
                  <input
                    type="text"
                    placeholder="Search by filename..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full text-xs pl-8 pr-3 py-1.5 rounded-lg border border-slate-200 bg-slate-50 focus:bg-white focus:ring-1 focus:ring-blue-500"
                  />
                </div>
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  className="text-xs rounded-lg border border-slate-200 bg-slate-50 py-1.5 px-2.5 text-slate-700 shrink-0 font-medium"
                >
                  <option value="ALL">All Items ({batchResults.length})</option>
                  <option value="PASS">PASS Only ({stats.passCount})</option>
                  <option value="REVIEW">REVIEW Only ({stats.reviewCount})</option>
                  <option value="REJECT">REJECT Only ({stats.rejectCount})</option>
                  <option value="NO_FACE">NO_FACE Only ({stats.noFaceCount})</option>
                  <option value="ERROR">Errors Only ({stats.errorCount})</option>
                </select>
              </div>

              {/* Action Buttons: Expand All + Passed ZIP + CSV & JSON Downloads */}
              <div className="flex flex-wrap items-center gap-2">
                <button
                  onClick={toggleExpandAll}
                  className="px-2.5 py-1.5 rounded-lg text-xs font-medium text-slate-600 hover:text-slate-900 bg-slate-100 hover:bg-slate-200 border border-slate-200 transition-colors"
                >
                  {expandedRows.size === filteredResults.length ? 'Collapse All Details' : 'Expand All Details'}
                </button>
                {batchId && (
                  <>
                    <a
                      href={stats.passCount > 0 ? getPassedZipUrl(batchId, reviewPolicy) : undefined}
                      download={stats.passCount > 0 ? `face_quality_passed_${batchId.slice(0, 8)}.zip` : undefined}
                      onClick={(e) => {
                        if (stats.passCount === 0) e.preventDefault();
                      }}
                      className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all border ${
                        stats.passCount > 0
                          ? 'bg-emerald-600 hover:bg-emerald-700 text-white border-emerald-600 shadow-xs cursor-pointer'
                          : 'bg-slate-100 text-slate-400 border-slate-200 cursor-not-allowed pointer-events-none'
                      }`}
                      title={stats.passCount > 0 ? `Download ${stats.passCount} passed face crops with manifest files in a ZIP archive` : 'No passed faces in this batch'}
                    >
                      <Download className="w-3.5 h-3.5" />
                      <span>Download Passed Faces ({stats.passCount}) (.zip)</span>
                    </a>
                    <a
                      href={getExportUrl(batchId, 'csv')}
                      download
                      className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 border border-slate-200 transition-colors"
                    >
                      <FileText className="w-3.5 h-3.5 text-blue-600" />
                      <span>CSV</span>
                    </a>
                    <a
                      href={getExportUrl(batchId, 'json')}
                      download
                      className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 border border-slate-200 transition-colors"
                    >
                      <FileJson className="w-3.5 h-3.5 text-indigo-600" />
                      <span>JSON</span>
                    </a>
                  </>
                )}
              </div>
            </div>

            {/* Results Table with Diagnostic Row Expanders */}
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left border-collapse">
                <thead className="bg-slate-50 text-slate-500 uppercase tracking-wider font-semibold border-b border-slate-200">
                  <tr>
                    <th className="py-3 px-3 w-8"></th>
                    <th className="py-3 px-3">Face Crop</th>
                    <th className="py-3 px-4">Filename</th>
                    <th className="py-3 px-4">Detector</th>
                    <th className="py-3 px-4">Status</th>
                    <th 
                      onClick={() => toggleSort('quality_score')}
                      className="py-3 px-4 cursor-pointer hover:text-slate-900"
                    >
                      <div className="flex items-center space-x-1">
                        <span>Quality Score</span>
                        <ArrowUpDown className="w-3 h-3" />
                      </div>
                    </th>
                    <th className="py-3 px-4">Decision</th>
                    <th 
                      onClick={() => toggleSort('processing_time_ms')}
                      className="py-3 px-4 cursor-pointer hover:text-slate-900 text-right"
                    >
                      <div className="flex items-center justify-end space-x-1">
                        <span>Latency</span>
                        <ArrowUpDown className="w-3 h-3" />
                      </div>
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {filteredResults.length === 0 ? (
                    <tr>
                      <td colSpan={8} className="py-8 text-center text-slate-400 text-xs">
                        No batch results match the current filters.
                      </td>
                    </tr>
                  ) : (
                    paginatedResults.map((row, localIdx) => {
                      const globalIdx = (currentPage - 1) * pageSize + localIdx;
                      const isExpanded = expandedRows.has(globalIdx);
                      const hasScore = row.quality_score !== null && row.quality_score !== undefined;
                      const eff = getItemEffectiveDecision(row);

                      return (
                        <React.Fragment key={row.request_id || globalIdx}>
                          <tr 
                            onClick={() => toggleRowExpanded(globalIdx)}
                            className={`cursor-pointer transition-colors ${
                              isExpanded ? 'bg-blue-50/40' : 'hover:bg-slate-50/70'
                            }`}
                          >
                            {/* Expand toggle icon */}
                            <td className="py-3 px-3 text-slate-400">
                              {isExpanded ? (
                                <ChevronDown className="w-4 h-4 text-blue-600" />
                              ) : (
                                <ChevronRight className="w-4 h-4" />
                              )}
                            </td>

                            {/* Aligned Face Crop Thumbnail with Click-to-Zoom */}
                            <td className="py-2 px-3">
                              {row.face_crop_base64 ? (
                                <div
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setViewerIndex(globalIdx);
                                    setViewerOpen(true);
                                  }}
                                  className="w-10 h-10 rounded-md overflow-hidden bg-slate-200 border border-slate-300 shrink-0 shadow-2xs hover:ring-2 hover:ring-indigo-500 cursor-zoom-in relative group transition-all"
                                  title="Click to zoom in lightbox"
                                >
                                  <img 
                                    src={row.face_crop_base64} 
                                    alt="Face crop" 
                                    className="w-full h-full object-cover" 
                                  />
                                  <div className="absolute inset-0 bg-black/35 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                                    <Maximize2 className="w-3.5 h-3.5 text-white" />
                                  </div>
                                </div>
                              ) : (
                                <div className="w-10 h-10 rounded-md bg-slate-100 border border-dashed border-slate-200 flex items-center justify-center text-[10px] text-slate-400 shrink-0">
                                  No face
                                </div>
                              )}
                            </td>

                            {/* Filename */}
                            <td className="py-3 px-4 font-medium text-slate-800 max-w-xs truncate" title={row.image_name}>
                              {row.image_name}
                              {row.face_index && row.face_index > 1 && (
                                <span className="ml-1 text-[10px] bg-slate-100 text-slate-600 px-1 py-0.2 rounded font-mono">
                                  #{row.face_index}
                                </span>
                              )}
                            </td>

                            {/* Detector */}
                            <td className="py-3 px-4 font-mono text-slate-600">
                              {row.detector}
                              {row.fallback_used && (
                                <span className="ml-1 text-[10px] text-indigo-600 font-semibold bg-indigo-50 px-1.5 py-0.2 rounded">
                                  fallback
                                </span>
                              )}
                            </td>

                            {/* Processing Status */}
                            <td className="py-3 px-4">
                              <ProcessingStatusBadge status={row.status} />
                              {row.error_reason && (
                                <div className="text-[10px] text-slate-500 mt-0.5 truncate max-w-xs" title={row.error_reason}>
                                  {row.error_reason}
                                </div>
                              )}
                            </td>

                            {/* Quality Score */}
                            <td className="py-3 px-4">
                              <div className="flex items-center space-x-2">
                                <span className="font-mono font-bold text-slate-900">
                                  {hasScore ? Number(row.quality_score).toFixed(4) : '-'}
                                </span>
                                {getScoreTierPill(row.quality_score)}
                              </div>
                            </td>

                            {/* Quality Decision */}
                            <td className="py-3 px-4">
                              <DecisionBadge
                                decision={eff}
                                policyOverride={row.policy_override}
                                originalDecision={row.gate_decision}
                              />
                            </td>

                            {/* Latency */}
                            <td className="py-3 px-4 text-right font-mono text-slate-500">
                              {Math.round(row.processing_time_ms)} ms
                            </td>
                          </tr>

                          {/* Professional Processing & Diagnostic Details Drawer */}
                          {isExpanded && (
                            <tr className="bg-slate-50/90 border-b border-blue-200">
                              <td colSpan={8} className="p-4 sm:p-6">
                                <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-6">
                                  {/* Drawer Header */}
                                  <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-3">
                                    <div className="flex items-center space-x-2.5">
                                      <div className="p-2 bg-blue-50 text-blue-600 rounded-lg">
                                        <ScanFace className="w-5 h-5" />
                                      </div>
                                      <div>
                                        <h4 className="text-sm font-bold text-slate-800">
                                          Processing & Diagnostic Details — {row.image_name}
                                        </h4>
                                        <p className="text-[11px] text-slate-400 font-mono">
                                          Request ID: {row.request_id || 'batch-item'}
                                        </p>
                                      </div>
                                    </div>
                                    <div className="flex items-center space-x-2">
                                      <ProcessingStatusBadge status={row.status} />
                                      {row.fallback_used && (
                                        <span className="text-[10px] bg-indigo-50 text-indigo-700 font-semibold px-2 py-0.5 rounded-full border border-indigo-200">
                                          Fallback Used
                                        </span>
                                      )}
                                      <button
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          toggleRowExpanded(globalIdx);
                                        }}
                                        className="text-xs text-slate-500 hover:text-slate-800 px-2.5 py-1 rounded-md hover:bg-slate-100 border border-slate-200 transition-colors"
                                      >
                                        Close
                                      </button>
                                    </div>
                                  </div>

                                  {/* Top Split: Detection Visualizer (Left) + Quality Score & Pose (Right) */}
                                  <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                                    {/* Left Column: Image Preview with Blue Square Overlay + Face Crop */}
                                    <div className="lg:col-span-7 bg-white rounded-xl border border-slate-200/90 p-4 shadow-2xs">
                                      <div className="flex items-center justify-between mb-3">
                                        <span className="text-xs font-semibold uppercase tracking-wider text-slate-700">
                                          Detection Visualizer
                                        </span>
                                        <div className="flex items-center space-x-2">
                                          <ProcessingStatusBadge status={row.status} />
                                          {row.image_width && row.image_height && (
                                            <span className="text-[11px] text-slate-500 font-mono">
                                              {row.image_width} × {row.image_height} px
                                            </span>
                                          )}
                                        </div>
                                      </div>

                                      <BoundingBoxViewer
                                        imageSrc={row.fileUrl || getFullImageUrl(row.image_name)}
                                        bbox={row.bbox}
                                        originalWidth={row.image_width}
                                        originalHeight={row.image_height}
                                        confidence={row.confidence}
                                        detector={row.detector}
                                        faceCropBase64={row.face_crop_base64}
                                        maxHeight={400}
                                      />
                                    </div>

                                    {/* Right Column: eDifFIQA Score Card, UniFace Head-Pose Angles, & Latency */}
                                    <div className="lg:col-span-5 space-y-4">
                                      <QualityScoreCard
                                        score={row.quality_score}
                                        decision={row.decision}
                                        decisionReason={row.decision_reason}
                                        status={row.status}
                                      />

                                      {/* UniFace Head-Pose Angles Card */}
                                      <div className="bg-white rounded-xl border border-slate-200/90 p-4 shadow-xs">
                                        <div className="flex items-center justify-between mb-3">
                                          <div className="flex items-center space-x-2">
                                            <Compass className="w-4 h-4 text-slate-600" />
                                            <span className="text-xs font-semibold text-slate-800">
                                              UniFace Head-Pose Angles
                                            </span>
                                          </div>
                                          <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                                            row.pose_status === 'OK' 
                                              ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' 
                                              : 'bg-slate-100 text-slate-500'
                                          }`}>
                                            {row.pose_status || 'NOT_CHECKED'}
                                          </span>
                                        </div>

                                        {row.pose_status === 'OK' ? (
                                          <div className="grid grid-cols-3 gap-2 text-center">
                                            <div className="bg-slate-50 p-2 rounded-lg border border-slate-100">
                                              <span className="text-[10px] text-slate-400 block font-medium">Pitch</span>
                                              <span className="text-sm font-bold text-slate-800 font-mono">
                                                {row.pitch !== null && row.pitch !== undefined ? `${row.pitch.toFixed(1)}°` : '-'}
                                              </span>
                                            </div>
                                            <div className="bg-slate-50 p-2 rounded-lg border border-slate-100">
                                              <span className="text-[10px] text-slate-400 block font-medium">Yaw</span>
                                              <span className="text-sm font-bold text-slate-800 font-mono">
                                                {row.yaw !== null && row.yaw !== undefined ? `${row.yaw.toFixed(1)}°` : '-'}
                                              </span>
                                            </div>
                                            <div className="bg-slate-50 p-2 rounded-lg border border-slate-100">
                                              <span className="text-[10px] text-slate-400 block font-medium">Roll</span>
                                              <span className="text-sm font-bold text-slate-800 font-mono">
                                                {row.roll !== null && row.roll !== undefined ? `${row.roll.toFixed(1)}°` : '-'}
                                              </span>
                                            </div>
                                          </div>
                                        ) : (
                                          <p className="text-xs text-slate-500 py-1">
                                            Head-pose estimation was {row.pose_status ? row.pose_status.toLowerCase() : 'not evaluated'} for this item.
                                          </p>
                                        )}
                                      </div>

                                      {/* UniFace MobileGaze Direction Card */}
                                      <div className="bg-white rounded-xl border border-slate-200/90 p-4 shadow-xs">
                                        <div className="flex items-center justify-between mb-3">
                                          <div className="flex items-center space-x-2">
                                            <Eye className="w-4 h-4 text-indigo-600" />
                                            <div>
                                              <span className="text-xs font-semibold text-slate-800 block">
                                                UniFace MobileGaze
                                              </span>
                                              <span className="text-[10px] text-slate-400 font-normal">
                                                Eye line-of-sight
                                              </span>
                                            </div>
                                          </div>
                                          <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                                            row.gaze_status === 'OK' 
                                              ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' 
                                              : 'bg-slate-100 text-slate-500'
                                          }`}>
                                            {row.gaze_status || 'DISABLED'}
                                          </span>
                                        </div>

                                        {row.gaze_status === 'OK' ? (
                                          <div className="grid grid-cols-2 gap-2 text-center">
                                            <div className="bg-slate-50 p-2 rounded-lg border border-slate-100">
                                              <span className="text-[10px] text-slate-400 block font-medium">Pitch (Vertical)</span>
                                              <span className="text-sm font-bold text-slate-800 font-mono">
                                                {row.gaze_pitch !== null && row.gaze_pitch !== undefined ? `${row.gaze_pitch.toFixed(1)}°` : '-'}
                                              </span>
                                            </div>
                                            <div className="bg-slate-50 p-2 rounded-lg border border-slate-100">
                                              <span className="text-[10px] text-slate-400 block font-medium">Yaw (Horizontal)</span>
                                              <span className="text-sm font-bold text-slate-800 font-mono">
                                                {row.gaze_yaw !== null && row.gaze_yaw !== undefined ? `${row.gaze_yaw.toFixed(1)}°` : '-'}
                                              </span>
                                            </div>
                                          </div>
                                        ) : (
                                          <p className="text-xs text-slate-500 py-1">
                                            Gaze estimation was {row.gaze_status ? row.gaze_status.toLowerCase() : 'disabled'} for this item.
                                          </p>
                                        )}
                                      </div>

                                      {/* Photometric & Clarity Diagnostics Card */}
                                      <div className="bg-white rounded-xl border border-slate-200/90 p-4 shadow-xs">
                                        <div className="flex items-center justify-between mb-3">
                                          <div className="flex items-center space-x-2">
                                            <Activity className="w-4 h-4 text-blue-600" />
                                            <div>
                                              <span className="text-xs font-semibold text-slate-800 block">Photometric Diagnostics</span>
                                              <span className="text-[10px] text-slate-400 font-normal">Explainable crop signals</span>
                                            </div>
                                          </div>
                                          <span className="text-[10px] bg-slate-100 text-slate-600 font-medium px-2 py-0.5 rounded-full">
                                            Advisory
                                          </span>
                                        </div>

                                        <div className="grid grid-cols-2 gap-2 text-xs">
                                          <div className="p-2 rounded-lg bg-slate-50 border border-slate-100">
                                            <span className="text-[10px] text-slate-400 block font-medium">Laplacian Sharpness</span>
                                            <span className="text-sm font-bold font-mono text-slate-800">
                                              {row.sharpness_laplacian_var !== null && row.sharpness_laplacian_var !== undefined
                                                ? row.sharpness_laplacian_var.toFixed(1)
                                                : '-'}
                                            </span>
                                          </div>
                                          <div className="p-2 rounded-lg bg-slate-50 border border-slate-100">
                                            <span className="text-[10px] text-slate-400 block font-medium">Mean Luminance</span>
                                            <span className="text-sm font-bold font-mono text-slate-800">
                                              {row.mean_luminance !== null && row.mean_luminance !== undefined
                                                ? `${row.mean_luminance.toFixed(1)}/255`
                                                : '-'}
                                            </span>
                                          </div>
                                          <div className="p-2 rounded-lg bg-slate-50 border border-slate-100">
                                            <span className="text-[10px] text-slate-400 block font-medium">Under/Over Ratios</span>
                                            <span className="text-xs font-bold font-mono text-slate-800">
                                              {row.underexposure_ratio !== null && row.overexposure_ratio !== null
                                                ? `U: ${(row.underexposure_ratio * 100).toFixed(0)}% | O: ${(row.overexposure_ratio * 100).toFixed(0)}%`
                                                : '-'}
                                            </span>
                                          </div>
                                          <div className="p-2 rounded-lg bg-slate-50 border border-slate-100">
                                            <span className="text-[10px] text-slate-400 block font-medium">Asymmetry</span>
                                            <span className="text-sm font-bold font-mono text-slate-800">
                                              {row.illumination_asymmetry !== null && row.illumination_asymmetry !== undefined
                                                ? row.illumination_asymmetry.toFixed(3)
                                                : '-'}
                                            </span>
                                          </div>
                                        </div>
                                      </div>

                                      {/* Total Inference Latency Card */}
                                      <div className="bg-white rounded-xl border border-slate-200/90 p-3 shadow-xs flex items-center justify-between text-xs">
                                        <div className="flex items-center space-x-2 text-slate-600">
                                          <Clock className="w-4 h-4 text-slate-400" />
                                          <span>Total Inference Latency</span>
                                        </div>
                                        <div className="space-x-2 font-mono text-slate-800">
                                          {row.detection_time_ms && (
                                            <span className="text-slate-400 text-[11px]">(Det: {Math.round(row.detection_time_ms)}ms)</span>
                                          )}
                                          <span className="font-semibold">{Math.round(row.processing_time_ms)} ms</span>
                                        </div>
                                      </div>
                                    </div>
                                  </div>

                                  {/* Bottom Section: Processing & Diagnostic Details Table (Matches Screenshot 2) */}
                                  <div className="bg-white rounded-xl border border-slate-200/90 p-4 shadow-2xs">
                                    <h5 className="text-xs font-semibold uppercase tracking-wider text-slate-700 mb-3">
                                      Processing & Diagnostic Details
                                    </h5>
                                    <div className="overflow-x-auto rounded-lg border border-slate-100">
                                      <table className="w-full text-xs text-left border-collapse">
                                        <tbody className="divide-y divide-slate-100">
                                          <tr>
                                            <td className="py-2.5 px-3 font-medium text-slate-500 w-1/3 bg-slate-50/50">Processing Status</td>
                                            <td className="py-2.5 px-3 text-slate-800">
                                              <ProcessingStatusBadge status={row.status} />
                                            </td>
                                          </tr>
                                          <tr>
                                            <td className="py-2.5 px-3 font-medium text-slate-500 bg-slate-50/50">Detector & Method</td>
                                            <td className="py-2.5 px-3 font-mono text-slate-800">
                                              {row.detector} {row.fallback_used && '(fallback invoked)'}
                                            </td>
                                          </tr>
                                          <tr>
                                            <td className="py-2.5 px-3 font-medium text-slate-500 bg-slate-50/50">Bounding Box [x1, y1, x2, y2]</td>
                                            <td className="py-2.5 px-3 font-mono text-slate-800">
                                              {row.bbox ? JSON.stringify(row.bbox.map(c => Math.round(c * 10) / 10)) : 'None'}
                                            </td>
                                          </tr>
                                          <tr>
                                            <td className="py-2.5 px-3 font-medium text-slate-500 bg-slate-50/50">Face Area Ratio</td>
                                            <td className="py-2.5 px-3 font-mono text-slate-800">
                                              {row.face_area_ratio !== null && row.face_area_ratio !== undefined ? `${(row.face_area_ratio * 100).toFixed(2)}% of image area` : 'None'}
                                            </td>
                                          </tr>
                                          {row.inter_eye_distance_px !== null && row.inter_eye_distance_px !== undefined && (
                                            <tr>
                                              <td className="py-2.5 px-3 font-medium text-slate-500 bg-slate-50/50">Inter-Eye Distance</td>
                                              <td className="py-2.5 px-3 font-mono text-slate-800">
                                                {row.inter_eye_distance_px.toFixed(1)} px
                                              </td>
                                            </tr>
                                          )}
                                          {row.sharpness_laplacian_var !== null && row.sharpness_laplacian_var !== undefined && (
                                            <tr>
                                              <td className="py-2.5 px-3 font-medium text-slate-500 bg-slate-50/50">Laplacian Sharpness</td>
                                              <td className="py-2.5 px-3 font-mono text-slate-800">
                                                {row.sharpness_laplacian_var.toFixed(2)} <span className="text-[11px] text-slate-400 font-sans">(Advisory clarity indicator)</span>
                                              </td>
                                            </tr>
                                          )}
                                          {row.mean_luminance !== null && row.mean_luminance !== undefined && (
                                            <tr>
                                              <td className="py-2.5 px-3 font-medium text-slate-500 bg-slate-50/50">Exposure Statistics</td>
                                              <td className="py-2.5 px-3 font-mono text-slate-800">
                                                Mean: {row.mean_luminance.toFixed(1)}/255 • Underexposed: {(row.underexposure_ratio * 100).toFixed(1)}% • Overexposed: {(row.overexposure_ratio * 100).toFixed(1)}%
                                              </td>
                                            </tr>
                                          )}
                                          {row.illumination_asymmetry !== null && row.illumination_asymmetry !== undefined && (
                                            <tr>
                                              <td className="py-2.5 px-3 font-medium text-slate-500 bg-slate-50/50">Illumination Asymmetry</td>
                                              <td className="py-2.5 px-3 font-mono text-slate-800">
                                                {row.illumination_asymmetry.toFixed(3)} <span className="text-[11px] text-slate-400 font-sans">(Left/Right half relative ratio)</span>
                                              </td>
                                            </tr>
                                          )}
                                          {row.gaze_status === 'OK' && row.gaze_pitch !== null && (
                                            <tr>
                                              <td className="py-2.5 px-3 font-medium text-slate-500 bg-slate-50/50">MobileGaze Direction</td>
                                              <td className="py-2.5 px-3 font-mono text-slate-800">
                                                Pitch: {row.gaze_pitch.toFixed(1)}° • Yaw: {row.gaze_yaw.toFixed(1)}° <span className="text-[11px] text-slate-400 font-sans">(Eye line-of-sight)</span>
                                              </td>
                                            </tr>
                                          )}
                                          {row.detection_note && (
                                            <tr>
                                              <td className="py-2.5 px-3 font-medium text-slate-500 bg-slate-50/50">Detection Note</td>
                                              <td className="py-2.5 px-3 text-slate-600 font-mono text-[11px]">
                                                {row.detection_note}
                                              </td>
                                            </tr>
                                          )}
                                          {row.error_reason && (
                                            <tr className="bg-amber-50/60">
                                              <td className="py-2.5 px-3 font-medium text-amber-800">Rejection / Reason</td>
                                              <td className="py-2.5 px-3 text-amber-900 font-medium">
                                                {row.error_reason}
                                              </td>
                                            </tr>
                                          )}
                                        </tbody>
                                      </table>
                                    </div>
                                  </div>
                                </div>
                              </td>
                            </tr>
                          )}
                        </React.Fragment>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>

            {/* Pagination Controls Bar */}
            {filteredResults.length > 0 && (
              <div className="p-3 border-t border-slate-100 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-600 bg-slate-50/50">
                <div className="flex items-center space-x-2">
                  <span>
                    Showing {filteredResults.length === 0 ? 0 : (currentPage - 1) * pageSize + 1} to{' '}
                    {Math.min(currentPage * pageSize, filteredResults.length)} of {filteredResults.length} faces
                  </span>
                  <div className="flex items-center space-x-1.5 ml-4">
                    <span className="text-slate-400">Per page:</span>
                    <select
                      value={pageSize}
                      onChange={(e) => setPageSize(Number(e.target.value))}
                      className="text-xs rounded border border-slate-200 bg-white py-0.5 px-2 font-medium"
                    >
                      <option value={10}>10</option>
                      <option value={25}>25</option>
                      <option value={50}>50</option>
                      <option value={100}>100</option>
                    </select>
                  </div>
                </div>

                <div className="flex items-center space-x-1.5">
                  <button
                    type="button"
                    onClick={() => setCurrentPage((p) => Math.max(p - 1, 1))}
                    disabled={currentPage <= 1}
                    className="px-2.5 py-1 rounded border border-slate-200 bg-white hover:bg-slate-100 disabled:opacity-40 disabled:cursor-not-allowed transition font-medium"
                  >
                    Previous
                  </button>
                  <span className="px-2 font-semibold text-slate-700">
                    Page {currentPage} of {totalPages}
                  </span>
                  <button
                    type="button"
                    onClick={() => setCurrentPage((p) => Math.min(p + 1, totalPages))}
                    disabled={currentPage >= totalPages}
                    className="px-2.5 py-1 rounded border border-slate-200 bg-white hover:bg-slate-100 disabled:opacity-40 disabled:cursor-not-allowed transition font-medium"
                  >
                    Next
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Lightbox Modal Viewer */}
      <ImageViewerModal
        isOpen={viewerOpen}
        onClose={() => setViewerOpen(false)}
        items={filteredResults}
        currentIndex={viewerIndex}
        onNavigate={(newIdx) => setViewerIndex(newIdx)}
      />
    </div>
  );
}
