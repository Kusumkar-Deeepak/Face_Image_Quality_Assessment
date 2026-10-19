"""Explain detector behaviour on specific images.

For each image and each detector this reports:
  * the detections the PRODUCTION configuration returns (default confidence 0.5),
  * the best sub-threshold candidates, obtained with a low confidence floor used ONLY for this diagnosis,
  * face-to-image size ratio and the face size at the detector's input after UniFace's letterbox resize,
  * optional SCRFD input-size sweep (``--scrfd-input-sizes 640 320``),
  * bbox/landmark validity and alignment status.

The diagnostic floor never feeds the quality pipeline; it exists so a miss like "max score 0.34" is visible.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from face_detection import DetectorConfig, DetectorSettings, build_detector, validate_face
from uniface import face_alignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare UniFace SCRFD and RetinaFace detections')
    parser.add_argument('images', nargs='+', help='Image paths to inspect')
    parser.add_argument('--detector-config', default=None, help='Production detector JSON config')
    parser.add_argument('--diag-floor', type=float, default=0.01, help='Confidence floor for the sub-threshold probe (diagnosis only)')
    parser.add_argument('--scrfd-input-sizes', type=int, nargs='*', default=[], help='Extra square SCRFD input sizes to probe, e.g. 320')
    parser.add_argument('--top', type=int, default=3, help='Sub-threshold candidates to list')
    parser.add_argument('--json', default=None, help='Optional JSON report path')
    return parser.parse_args()


def letterbox_factor(image_shape: tuple[int, ...], input_size: tuple[int, int]) -> float:
    """Scale UniFace's resize_image applies (aspect-preserving fit; it also UPSCALES small images)."""
    height, width = image_shape[:2]
    return min(input_size[0] / width, input_size[1] / height)


def describe_face(image: np.ndarray, face: Any, input_size: tuple[int, int]) -> dict[str, Any]:
    height, width = image.shape[:2]
    bbox = np.asarray(face.bbox, dtype=np.float64)
    reason = validate_face(face, width, height)
    info: dict[str, Any] = {
        'confidence': float(face.confidence),
        'bbox': bbox.round(1).tolist(),
        'valid': reason is None,
        'invalid_reason': reason,
        'alignment_status': 'SKIPPED',
        'alignment_error': None,
    }
    if reason is None:
        x1, y1, x2, y2 = bbox
        cw, ch = min(x2, width) - max(x1, 0), min(y2, height) - max(y1, 0)
        factor = letterbox_factor(image.shape, input_size)
        info['face_area_ratio'] = round(float(cw * ch / (width * height)), 3)
        info['face_longest_side_px'] = round(float(max(cw, ch)), 1)
        info['face_longest_side_at_detector_input_px'] = round(float(max(cw, ch) * factor), 1)
        try:
            aligned, _ = face_alignment(image, np.asarray(face.landmarks, dtype=np.float32))
            if aligned is None or aligned.size == 0:
                raise ValueError('alignment returned an empty image')
            info['alignment_status'] = 'OK'
        except Exception as error:
            info.update(alignment_status='ALIGNMENT_ERROR', alignment_error=str(error))
    return info


def probe(detector: Any, image: np.ndarray, settings: DetectorSettings, top: int) -> dict[str, Any]:
    faces = sorted(detector.detect(image), key=lambda f: float(f.confidence), reverse=True)
    return {
        'confidence_threshold': settings.confidence_threshold,
        'input_size': list(settings.input_size),
        'n_detections': len(faces),
        'top': [describe_face(image, face, settings.input_size) for face in faces[:top]],
    }


def inspect_image(path: str, config: DetectorConfig, floor: float, sizes: list[int], top: int, cache: dict) -> dict[str, Any]:
    result: dict[str, Any] = {'image': path, 'shape': None, 'status': 'OK', 'error_reason': None, 'runs': []}
    image_path = Path(path).expanduser()
    if not image_path.is_file():
        result.update(status='READ_ERROR', error_reason=f'path is not a file: {image_path}')
        return result
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        result.update(status='READ_ERROR', error_reason=f'could not decode image: {image_path}')
        return result
    result['shape'] = list(image.shape)

    plans: list[tuple[str, str, DetectorSettings, str]] = []
    for name in ('scrfd', 'retinaface'):
        base = config.settings_for(name)
        plans.append((f'{name} production', name, base, 'production'))
        plans.append((f'{name} floor={floor}', name, replace(base, confidence_threshold=floor), 'diagnostic-only'))
    for size in sizes:
        base = config.settings_for('scrfd')
        plans.append((f'scrfd input={size}', 'scrfd', replace(base, input_size=(size, size)), 'production-threshold'))

    for label, name, settings, kind in plans:
        key = (name, settings)
        try:
            if key not in cache:
                cfg = replace(config, scrfd=settings) if name == 'scrfd' else replace(config, retinaface=settings)
                cache[key] = build_detector(name, cfg)
            run = probe(cache[key], image, settings, top)
            run.update(label=label, kind=kind, status='OK' if run['n_detections'] else 'NO_FACE')
        except Exception as error:
            run = {'label': label, 'kind': kind, 'status': 'MODEL_ERROR', 'error_reason': str(error), 'top': []}
        result['runs'].append(run)
    return result


def main() -> None:
    args = parse_args()
    config = DetectorConfig.from_file(args.detector_config)
    cache: dict = {}
    report = [inspect_image(p, config, args.diag_floor, args.scrfd_input_sizes, args.top, cache) for p in args.images]

    for item in report:
        print(f"\n{item['image']} shape={item['shape']} status={item['status']}")
        if item['error_reason']:
            print(f"  error={item['error_reason']}")
        for run in item['runs']:
            print(f"  [{run['label']}] ({run['kind']}) status={run['status']} detections={run.get('n_detections')}")
            if run.get('error_reason'):
                print(f"    error={run['error_reason']}")
            for face in run['top']:
                line = (
                    f"    conf={face['confidence']:.6f} bbox={face['bbox']} valid={face['valid']} "
                    f"alignment={face['alignment_status']}"
                )
                if 'face_area_ratio' in face:
                    line += (
                        f" area_ratio={face['face_area_ratio']} side={face['face_longest_side_px']}px"
                        f" -> {face['face_longest_side_at_detector_input_px']}px at detector input"
                    )
                print(line)
                if face['invalid_reason']:
                    print(f"      invalid: {face['invalid_reason']}")
                if face['alignment_error']:
                    print(f"      alignment_error: {face['alignment_error']}")
    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'\nJSON: {output}')


if __name__ == '__main__':
    main()
