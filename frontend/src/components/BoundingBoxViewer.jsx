import React, { useRef, useState, useEffect } from 'react';
import { Maximize2, ShieldCheck, UserCheck } from 'lucide-react';

export default function BoundingBoxViewer({ 
  imageSrc, 
  bbox, 
  originalWidth, 
  originalHeight, 
  confidence, 
  detector, 
  faceCropBase64,
  allFaces = [],
  activeFaceIndex = 0,
  onSelectFace = null,
  maxHeight = 460,
  onZoom = null,
}) {
  const containerRef = useRef(null);
  const imgRef = useRef(null);
  const [scale, setScale] = useState({ scaleX: 1, scaleY: 1, offsetX: 0, offsetY: 0, ready: false });

  // Update box scaling whenever image loads or container resizes
  const updateScaling = () => {
    if (!imgRef.current) return;
    const { clientWidth, clientHeight, naturalWidth, naturalHeight } = imgRef.current;
    const realWidth = originalWidth || naturalWidth;
    const realHeight = originalHeight || naturalHeight;
    if (!realWidth || !realHeight) return;
    
    setScale({
      scaleX: clientWidth / realWidth,
      scaleY: clientHeight / realHeight,
      offsetX: imgRef.current.offsetLeft,
      offsetY: imgRef.current.offsetTop,
      ready: true,
    });
  };

  useEffect(() => {
    if (!imgRef.current) return;
    
    if (imgRef.current.complete) {
      updateScaling();
    }

    let observer = null;
    if (typeof ResizeObserver !== 'undefined') {
      observer = new ResizeObserver(() => {
        updateScaling();
      });
      observer.observe(imgRef.current);
    }
    
    window.addEventListener('resize', updateScaling);
    return () => {
      if (observer) observer.disconnect();
      window.removeEventListener('resize', updateScaling);
    };
  }, [imageSrc, originalWidth, originalHeight]);

  // Compute CSS box coordinates from a bbox array [x1, y1, x2, y2]
  const computeBoxStyle = (box) => {
    if (!box || box.length !== 4 || !scale.ready) return null;
    const [x1, y1, x2, y2] = box;
    const left = scale.offsetX + x1 * scale.scaleX;
    const top = scale.offsetY + y1 * scale.scaleY;
    const width = (x2 - x1) * scale.scaleX;
    const height = (y2 - y1) * scale.scaleY;
    return {
      left: `${left}px`,
      top: `${top}px`,
      width: `${width}px`,
      height: `${height}px`,
    };
  };

  const hasMultipleFaces = allFaces && allFaces.length > 1;

  return (
    <div className="flex flex-col space-y-4">
      {/* Main Image with Responsive Bounding Box Overlay */}
      <div 
        ref={containerRef}
        style={{ maxHeight: `${maxHeight}px` }}
        className="relative bg-slate-900/90 rounded-xl overflow-hidden flex items-center justify-center min-h-[260px] border border-slate-200"
      >
        <img
          ref={imgRef}
          src={imageSrc}
          alt="Analysis preview"
          onLoad={updateScaling}
          style={{ maxHeight: `${maxHeight - 10}px` }}
          className="w-auto max-w-full object-contain mx-auto"
        />

        {/* Total faces detected overlay badge */}
        {hasMultipleFaces && (
          <div className="absolute top-2.5 left-2.5 bg-slate-900/80 text-white text-[11px] px-2.5 py-1 rounded-md backdrop-blur-xs font-medium flex items-center space-x-1.5 shadow-sm border border-slate-700/60 z-20">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span>{allFaces.length} Faces Detected</span>
          </div>
        )}

        {/* Render bounding boxes for all detected faces */}
        {hasMultipleFaces ? (
          allFaces.map((faceItem, idx) => {
            const isSelected = idx === activeFaceIndex;
            const style = computeBoxStyle(faceItem.bbox);
            if (!style) return null;

            const isNearTop = faceItem.bbox && (faceItem.bbox[1] * scale.scaleY < 32);

            return (
              <div
                key={idx}
                style={style}
                onClick={() => onSelectFace && onSelectFace(idx)}
                className={`absolute rounded-sm transition-all duration-150 cursor-pointer ${
                  isSelected
                    ? 'border-2 border-blue-400 bg-blue-500/20 z-10 shadow-sm'
                    : 'border border-slate-300/80 bg-white/5 hover:border-blue-300 z-0'
                }`}
                title={`Face #${idx + 1} (Click to inspect)`}
              >
                <div className={`absolute text-[10px] font-medium px-1.5 py-0.5 rounded shadow-xs whitespace-nowrap flex items-center space-x-1 ${
                  isNearTop ? 'top-1 left-1' : '-top-6 left-0'
                } ${
                  isSelected ? 'bg-blue-600 text-white font-semibold' : 'bg-slate-800/80 text-slate-200'
                }`}>
                  <span>#{idx + 1}</span>
                  {faceItem.quality_score !== null && faceItem.quality_score !== undefined && (
                    <span className="font-mono">({faceItem.quality_score.toFixed(2)})</span>
                  )}
                </div>
              </div>
            );
          })
        ) : (
          bbox && bbox.length === 4 && scale.ready && (
            <div
              style={computeBoxStyle(bbox)}
              className="absolute border-2 border-blue-400 bg-blue-500/10 rounded-sm pointer-events-none transition-all duration-150"
            >
              <div className={`absolute bg-blue-600 text-white text-[11px] font-medium px-2 py-0.5 rounded shadow-sm whitespace-nowrap flex items-center space-x-1 ${
                (bbox[1] * scale.scaleY < 32) ? 'top-1 left-1' : '-top-7 left-0'
              }`}>
                <span>Face</span>
                {confidence !== null && confidence !== undefined && (
                  <span className="opacity-90 font-mono">
                    {(confidence * 100).toFixed(1)}%
                  </span>
                )}
              </div>
            </div>
          )
        )}
      </div>

      {/* Detected Face Crop Thumbnail and Geometry Summary */}
      {faceCropBase64 && (
        <div className="flex items-center space-x-4 bg-slate-50 p-3 rounded-lg border border-slate-200">
          <div 
            onClick={onZoom || undefined}
            className={`relative shrink-0 w-16 h-16 rounded-md overflow-hidden bg-slate-200 border border-slate-300 shadow-xs ${
              onZoom ? 'cursor-pointer group' : ''
            }`}
            title={onZoom ? 'Click to inspect in Zoom Lightbox' : undefined}
          >
            <img 
              src={faceCropBase64} 
              alt="Detected Face Crop" 
              className={`w-full h-full object-cover ${onZoom ? 'transition-transform duration-200 group-hover:scale-105' : ''}`}
            />
            {onZoom && (
              <div className="absolute inset-0 bg-slate-900/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity rounded-md">
                <Maximize2 className="w-4 h-4 text-white" />
              </div>
            )}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center space-x-2">
              <span className="text-xs font-semibold text-slate-800">Detected Face Crop</span>
              <span className="text-[10px] bg-blue-100 text-blue-700 font-mono px-1.5 py-0.2 rounded font-medium">
                112×112 Aligned
              </span>
            </div>
            <p className="text-xs text-slate-500 truncate mt-0.5">
              Detector: <span className="font-medium text-slate-700">{detector}</span>
            </p>
            {confidence !== null && confidence !== undefined && (
              <p className="text-xs text-slate-500">
                Confidence: <span className="font-mono text-slate-700 font-medium">{Number(confidence).toFixed(6)}</span>
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
