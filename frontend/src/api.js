/**
 * API client module for Face Quality Lab.
 * Connects directly to the FastAPI backend.
 */

const BASE_URL = import.meta.env.VITE_API_URL || '';

export async function fetchHealth() {
  const res = await fetch(`${BASE_URL}/api/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchConfig() {
  const res = await fetch(`${BASE_URL}/api/config`);
  if (!res.ok) throw new Error(`Config fetch failed: ${res.statusText}`);
  return res.json();
}

export async function updateThresholds(thresholds) {
  const res = await fetch(`${BASE_URL}/api/config/thresholds`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(thresholds),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Threshold update failed (${res.status})`);
  }
  return res.json();
}

export async function fetchSessionStats() {
  const res = await fetch(`${BASE_URL}/api/session/stats`);
  if (!res.ok) throw new Error(`Session stats fetch failed: ${res.statusText}`);
  return res.json();
}

export async function analyzeSingleImage({
  file,
  detector = 'scrfd',
  fallbackDetector = null,
  detectionMode = 'fallback',
  confidenceThreshold = null,
  scrfdInputSize = 640,
  headPose = false,
  gaze = false,
  reviewPolicy = 'manual_review',
}) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('detector', detector);
  if (fallbackDetector) {
    formData.append('fallback_detector', fallbackDetector);
  }
  formData.append('detection_mode', detectionMode);
  if (confidenceThreshold !== null && confidenceThreshold !== undefined) {
    formData.append('confidence_threshold', confidenceThreshold.toString());
  }
  formData.append('scrfd_input_size', scrfdInputSize.toString());
  formData.append('head_pose', headPose.toString());
  formData.append('gaze', gaze.toString());
  formData.append('review_policy', reviewPolicy);

  const res = await fetch(`${BASE_URL}/api/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Analysis failed (${res.status})`);
  }

  return res.json();
}

export async function analyzeBatchImages({
  files,
  detector = 'scrfd',
  fallbackDetector = null,
  detectionMode = 'fallback',
  confidenceThreshold = null,
  scrfdInputSize = 640,
  headPose = false,
  gaze = false,
  reviewPolicy = 'manual_review',
}) {
  const formData = new FormData();
  for (const file of files) {
    formData.append('files', file);
  }
  formData.append('detector', detector);
  if (fallbackDetector) {
    formData.append('fallback_detector', fallbackDetector);
  }
  formData.append('detection_mode', detectionMode);
  if (confidenceThreshold !== null && confidenceThreshold !== undefined) {
    formData.append('confidence_threshold', confidenceThreshold.toString());
  }
  formData.append('scrfd_input_size', scrfdInputSize.toString());
  formData.append('head_pose', headPose.toString());
  formData.append('gaze', gaze.toString());
  formData.append('review_policy', reviewPolicy);

  const res = await fetch(`${BASE_URL}/api/analyze/batch`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Batch analysis failed (${res.status})`);
  }

  return res.json();
}

export function getExportUrl(batchId, format = 'csv') {
  return `${BASE_URL}/api/results/${batchId}/export?format=${format}`;
}

export function getPassedZipUrl(batchId, reviewPolicy = 'manual_review') {
  return `${BASE_URL}/api/results/${batchId}/passed-zip?review_policy=${encodeURIComponent(reviewPolicy)}`;
}

export async function fetchPassedCount(batchId, reviewPolicy = 'manual_review') {
  const res = await fetch(`${BASE_URL}/api/results/${batchId}/passed-count?review_policy=${encodeURIComponent(reviewPolicy)}`);
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to fetch passed count (${res.status})`);
  }
  return res.json();
}

export async function saveBatchResult(batchData) {
  const res = await fetch(`${BASE_URL}/api/results/save`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(batchData),
  });
  if (!res.ok) throw new Error(`Failed to save batch (${res.status})`);
  return res.json();
}
