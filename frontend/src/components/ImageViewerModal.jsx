import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  X, 
  ZoomIn, 
  ZoomOut, 
  Maximize2, 
  RotateCcw, 
  ChevronLeft, 
  ChevronRight, 
  Layers, 
  Crop, 
  ScanFace 
} from 'lucide-react';
import { DecisionBadge } from './StatusBadge';

export default function ImageViewerModal({
  isOpen,
  onClose,
  items = [],
  currentIndex = 0,
  onNavigate,
}) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [showOverlay, setShowOverlay] = useState(true);
  const [viewMode, setViewMode] = useState('full'); // 'full' | 'crop'

  const containerRef = useRef(null);

  const currentItem = items[currentIndex] || null;

  // Reset zoom & pan when item changes
  const resetTransform = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  useEffect(() => {
    resetTransform();
    // Default to crop view if no full image available
    if (currentItem && !currentItem.fileUrl && currentItem.face_crop_base64) {
      setViewMode('crop');
    }
  }, [currentIndex, currentItem, resetTransform]);

  // Keyboard navigation & controls
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        onClose();
      } else if (e.key === 'ArrowLeft') {
        if (onNavigate && currentIndex > 0) onNavigate(currentIndex - 1);
      } else if (e.key === 'ArrowRight') {
        if (onNavigate && currentIndex < items.length - 1) onNavigate(currentIndex + 1);
      } else if (e.key === '+' || e.key === '=') {
        setZoom((prev) => Math.min(prev + 0.25, 4));
      } else if (e.key === '-') {
        setZoom((prev) => Math.max(prev - 0.25, 0.5));
      } else if (e.key === '0') {
        resetTransform();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose, currentIndex, items.length, onNavigate, resetTransform]);

  if (!isOpen || !currentItem) return null;

  const handleZoomIn = () => setZoom((prev) => Math.min(prev + 0.25, 4));
  const handleZoomOut = () => setZoom((prev) => Math.max(prev - 0.25, 0.5));
  const handleFitScreen = () => resetTransform();
  const handleOriginalSize = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const handleMouseDown = (e) => {
    if (zoom <= 1) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    setPan({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  };

  const handleMouseUp = () => setIsDragging(false);

  const hasCrop = Boolean(currentItem.face_crop_base64);
  const hasFull = Boolean(currentItem.fileUrl);

  const displayImageSrc =
    viewMode === 'crop' && hasCrop
      ? currentItem.face_crop_base64
      : currentItem.fileUrl || currentItem.face_crop_base64;

  const decisionState = currentItem.effective_decision || currentItem.gate_decision || currentItem.decision;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-sm p-2 sm:p-4 animate-in fade-in duration-150"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      role="dialog"
      aria-modal="true"
    >
      <div className="relative w-full max-w-5xl h-[90vh] bg-slate-900 border border-slate-700 rounded-xl shadow-2xl flex flex-col overflow-hidden">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-4 py-3 bg-slate-800/90 border-b border-slate-700 z-10">
          <div className="flex items-center space-x-3 overflow-hidden">
            <span className="font-semibold text-slate-100 truncate text-sm sm:text-base">
              {currentItem.image_name}
            </span>
            {currentItem.face_index && (
              <span className="px-2 py-0.5 rounded text-xs bg-slate-700 text-slate-300 font-medium">
                Face #{currentItem.face_index}
              </span>
            )}
            <DecisionBadge
              decision={decisionState}
              policyOverride={currentItem.policy_override}
              originalDecision={currentItem.gate_decision}
            />
            {currentItem.quality_score !== null && currentItem.quality_score !== undefined && (
              <span className="text-xs text-indigo-300 font-mono hidden sm:inline">
                Score: {currentItem.quality_score.toFixed(4)}
              </span>
            )}
          </div>

          <div className="flex items-center space-x-2">
            {/* View Mode Toggle */}
            {hasCrop && hasFull && (
              <div className="flex bg-slate-700/80 rounded-lg p-0.5 border border-slate-600 text-xs">
                <button
                  type="button"
                  onClick={() => {
                    setViewMode('full');
                    resetTransform();
                  }}
                  className={`flex items-center space-x-1 px-2.5 py-1 rounded-md transition ${
                    viewMode === 'full'
                      ? 'bg-indigo-600 text-white font-medium shadow-sm'
                      : 'text-slate-300 hover:text-white'
                  }`}
                >
                  <ScanFace className="w-3.5 h-3.5" />
                  <span>Full Image</span>
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setViewMode('crop');
                    resetTransform();
                  }}
                  className={`flex items-center space-x-1 px-2.5 py-1 rounded-md transition ${
                    viewMode === 'crop'
                      ? 'bg-indigo-600 text-white font-medium shadow-sm'
                      : 'text-slate-300 hover:text-white'
                  }`}
                >
                  <Crop className="w-3.5 h-3.5" />
                  <span>Face Crop</span>
                </button>
              </div>
            )}

            {/* Close Button */}
            <button
              type="button"
              onClick={onClose}
              className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition"
              title="Close (ESC)"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Modal Viewport */}
        <div
          ref={containerRef}
          className="relative flex-1 bg-slate-950 flex items-center justify-center overflow-hidden cursor-grab active:cursor-grabbing select-none"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          {displayImageSrc ? (
            <div
              style={{
                transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                transition: isDragging ? 'none' : 'transform 0.15s ease-out',
                position: 'relative',
              }}
              className="max-w-full max-h-full flex items-center justify-center"
            >
              <img
                src={displayImageSrc}
                alt={currentItem.image_name}
                className="max-h-[72vh] max-w-full object-contain pointer-events-none rounded shadow-md"
                draggable={false}
              />

              {/* Bounding box overlay on full image */}
              {viewMode === 'full' && showOverlay && currentItem.bbox && currentItem.image_width && currentItem.image_height && (
                <div
                  className="absolute pointer-events-none border-2 border-indigo-400 bg-indigo-500/20 rounded-sm"
                  style={{
                    left: `${(currentItem.bbox[0] / currentItem.image_width) * 100}%`,
                    top: `${(currentItem.bbox[1] / currentItem.image_height) * 100}%`,
                    width: `${((currentItem.bbox[2] - currentItem.bbox[0]) / currentItem.image_width) * 100}%`,
                    height: `${((currentItem.bbox[3] - currentItem.bbox[1]) / currentItem.image_height) * 100}%`,
                  }}
                >
                  <div className="absolute -top-5 left-0 px-1.5 py-0.5 bg-indigo-600 text-white text-[10px] font-bold rounded-sm whitespace-nowrap shadow">
                    Face #{currentItem.face_index || 1}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-slate-400 text-sm">Image preview unavailable</div>
          )}

          {/* Navigation Arrows */}
          {items.length > 1 && currentIndex > 0 && (
            <button
              type="button"
              onClick={() => onNavigate(currentIndex - 1)}
              className="absolute left-3 top-1/2 -translate-y-1/2 p-2 rounded-full bg-slate-800/80 text-white hover:bg-slate-700 shadow-lg border border-slate-600 transition"
              title="Previous (Left Arrow)"
            >
              <ChevronLeft className="w-5 h-5" />
            </button>
          )}

          {items.length > 1 && currentIndex < items.length - 1 && (
            <button
              type="button"
              onClick={() => onNavigate(currentIndex + 1)}
              className="absolute right-3 top-1/2 -translate-y-1/2 p-2 rounded-full bg-slate-800/80 text-white hover:bg-slate-700 shadow-lg border border-slate-600 transition"
              title="Next (Right Arrow)"
            >
              <ChevronRight className="w-5 h-5" />
            </button>
          )}
        </div>

        {/* Modal Controls Bar */}
        <div className="flex items-center justify-between px-4 py-2.5 bg-slate-800/90 border-t border-slate-700 text-xs text-slate-300">
          <div className="flex items-center space-x-2">
            <span>
              {currentIndex + 1} of {items.length}
            </span>
            {viewMode === 'full' && currentItem.bbox && (
              <button
                type="button"
                onClick={() => setShowOverlay((prev) => !prev)}
                className={`flex items-center space-x-1 px-2 py-1 rounded border transition ${
                  showOverlay
                    ? 'bg-indigo-600/30 border-indigo-500 text-indigo-300'
                    : 'bg-slate-700 border-slate-600 text-slate-400'
                }`}
                title="Toggle Face Box Overlay"
              >
                <Layers className="w-3.5 h-3.5" />
                <span>Box Overlay</span>
              </button>
            )}
          </div>

          {/* Zoom controls */}
          <div className="flex items-center space-x-1 sm:space-x-2">
            <button
              type="button"
              onClick={handleZoomOut}
              className="p-1.5 hover:bg-slate-700 rounded text-slate-300 hover:text-white transition"
              title="Zoom Out (-)"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <span className="w-12 text-center font-mono font-medium text-slate-200">
              {Math.round(zoom * 100)}%
            </span>
            <button
              type="button"
              onClick={handleZoomIn}
              className="p-1.5 hover:bg-slate-700 rounded text-slate-300 hover:text-white transition"
              title="Zoom In (+)"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            <div className="h-4 w-px bg-slate-600 mx-1" />
            <button
              type="button"
              onClick={handleOriginalSize}
              className="p-1.5 hover:bg-slate-700 rounded text-slate-300 hover:text-white transition"
              title="100% Size (0)"
            >
              <Maximize2 className="w-4 h-4" />
            </button>
            <button
              type="button"
              onClick={handleFitScreen}
              className="p-1.5 hover:bg-slate-700 rounded text-slate-300 hover:text-white transition"
              title="Reset View"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
