"""Reproducible batch evaluation: detection success, fallback usage, scores, timing, controlled degradations.

Outputs one row per detected face (or one row per image when nothing could be scored), plus a summary JSON.
Scores are raw eDifFIQA outputs; nothing here treats a detector miss as a quality verdict.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from pipeline_cli import add_pipeline_arguments, pipeline_from_args
from quality_pipeline import EvaluationResult, QualityPipeline

IMAGE_SUFFIXES = {'.bmp', '.jpeg', '.jpg', '.png', '.webp'}
DERIVED_SUFFIXES = {'_blur': 'blur', '_lowres': 'lowres'}  # naming used by the repo's sample images


# ----------------------------------------------------------------------------- inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Batch eDifFIQA evaluation with status-aware reporting')
    parser.add_argument('inputs', type=str, nargs='+', help='Image paths or directories')
    add_pipeline_arguments(parser)
    parser.set_defaults(device='cpu')
    parser.add_argument('--labels', default=None, help='Optional CSV with columns image[,category][,label]; category overrides inference')
    parser.add_argument('--degrade', action='append', default=[], metavar='KIND:VALUE',
                        help='Controlled in-memory variants of every original: blur:SIGMA, lowres:FACTOR, dark:SCALE (repeatable)')
    parser.add_argument('--save-variants', default=None, help='Directory to save generated variants for inspection')
    parser.add_argument('--yaw-bins', type=float, nargs='*', default=[], help='|yaw| bin edges for a score-vs-pose report (needs --head-pose)')
    parser.add_argument('--csv', default='results/batch_evaluation.csv')
    parser.add_argument('--json', default='results/batch_evaluation.json')
    parser.add_argument('--summary', default='results/batch_summary.json')
    return parser.parse_args()


def expand_inputs(inputs: list[str]) -> list[str]:
    paths: list[str] = []
    for item in inputs:
        path = Path(item).expanduser()
        if path.is_dir():
            paths.extend(str(child) for child in sorted(path.rglob('*')) if child.suffix.lower() in IMAGE_SUFFIXES)
        else:
            paths.append(item)
    return list(dict.fromkeys(paths))  # de-duplicate, keep order


def classify_path(path: str) -> tuple[str, str]:
    """Return ``(category, source_stem)`` from the file name: ``image2_blur.jpg`` -> ``('blur', 'image2')``."""
    stem = Path(path).stem
    for suffix, category in DERIVED_SUFFIXES.items():
        if stem.endswith(suffix):
            return category, stem[: -len(suffix)]
    return 'original', stem


def load_labels(path: str | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with Path(path).open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if 'image' not in (reader.fieldnames or []):
            raise ValueError(f'labels file {path} needs an "image" column')
        return {Path(row['image']).as_posix(): row for row in reader}


# ----------------------------------------------------------------------------- degradations


def degrade_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)


def degrade_lowres(image: np.ndarray, factor: float) -> np.ndarray:
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, round(w / factor)), max(1, round(h / factor))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def degrade_dark(image: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)


DEGRADATIONS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    'blur': degrade_blur,
    'lowres': degrade_lowres,
    'dark': degrade_dark,
}


def parse_degradations(specs: list[str]) -> list[tuple[str, float]]:
    parsed = []
    for spec in specs:
        kind, _, value = spec.partition(':')
        if kind not in DEGRADATIONS:
            raise ValueError(f'unknown degradation {kind!r}; choose from {sorted(DEGRADATIONS)}')
        try:
            number = float(value)
        except ValueError as error:
            raise ValueError(f'degradation {spec!r} needs a numeric value, e.g. {kind}:2.0') from error
        if number <= 0 or (kind == 'dark' and number >= 1):
            raise ValueError(f'invalid value for {spec!r}')
        parsed.append((kind, number))
    return parsed


# ----------------------------------------------------------------------------- evaluation


def annotate(result: EvaluationResult, category: str, source: str, variant: str) -> dict[str, Any]:
    row = result.to_dict()
    row.update(category=category, source_image=source, variant=variant)
    return row


def run_batch(
    pipeline: QualityPipeline,
    paths: list[str],
    *,
    aligned: bool = False,
    degradations: list[tuple[str, float]] | None = None,
    labels: dict[str, dict[str, str]] | None = None,
    save_variants: str | Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = labels or {}
    for path in paths:
        label_row = labels.get(Path(path).as_posix(), {})
        inferred, source = classify_path(path)
        category = label_row.get('category') or inferred
        results = pipeline.evaluate(path, aligned=aligned)
        rows.extend(annotate(r, category, source, 'file') for r in results)
        if label_row.get('label'):
            for row in rows[-len(results):]:
                row['label'] = label_row['label']

        if not degradations or category != 'original' or results[0].status == 'READ_ERROR':
            continue
        image = cv2.imread(str(Path(path).expanduser()), cv2.IMREAD_COLOR)
        if image is None:
            continue
        for kind, value in degradations:
            variant = f'{kind}_{value:g}'
            degraded = DEGRADATIONS[kind](image, value)
            if save_variants:
                out = Path(save_variants)
                out.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / f'{Path(path).stem}__{variant}.png'), degraded)
            name = f'{path}::{variant}'
            rows.extend(annotate(r, kind, source, variant) for r in pipeline.evaluate_array(degraded, name=name, aligned=aligned))
    return rows


# ----------------------------------------------------------------------------- summaries


def score_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {'n': 0, 'mean': None, 'std': None, 'min': None, 'p25': None, 'median': None, 'p75': None, 'max': None}
    a = np.asarray(values, dtype=np.float64)
    p25, med, p75 = np.percentile(a, [25, 50, 75])
    return {'n': int(a.size), 'mean': float(a.mean()), 'std': float(a.std()), 'min': float(a.min()),
            'p25': float(p25), 'median': float(med), 'p75': float(p75), 'max': float(a.max())}


def _count(rows: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        out[str(row[key])] = out.get(str(row[key]), 0) + 1
    return dict(sorted(out.items()))


def _primary_rows(rows: list[dict]) -> dict[str, dict]:
    """One row per image: the highest-detection-confidence face (or the single non-face row)."""
    best: dict[str, dict] = {}
    for row in rows:
        current = best.get(row['image'])
        if current is None or (row['confidence'] or -1) > (current['confidence'] or -1):
            best[row['image']] = row
    return best


def build_summary(rows: list[dict], yaw_bins: list[float] | None = None) -> dict[str, Any]:
    images = _primary_rows(rows)
    decoded = [r for r in images.values() if r['status'] != 'READ_ERROR']
    with_face = [r for r in decoded if r['status'] in ('OK', 'ALIGNMENT_ERROR', 'MODEL_ERROR') and r['face_index'] is not None]
    fallback_images = [r for r in decoded if r['fallback_used']]
    times = [r['processing_time_ms'] for r in rows]

    scored = [r for r in rows if r['status'] == 'OK' and r['quality_score'] is not None]
    by_category: dict[str, list[float]] = {}
    for r in scored:
        by_category.setdefault(r['category'], []).append(r['quality_score'])

    # Paired comparison: each variant against the same source image, using each image's most confident face.
    originals = {r['source_image']: r for r in images.values() if r['category'] == 'original' and r['status'] == 'OK'}
    pairs = []
    for r in images.values():
        base = originals.get(r['source_image'])
        if r['category'] == 'original' or base is None or r['status'] != 'OK':
            continue
        delta = r['quality_score'] - base['quality_score']
        pairs.append({'source': r['source_image'], 'variant': f"{r['category']}:{r['variant']}", 'original_score': base['quality_score'],
                      'variant_score': r['quality_score'], 'delta': delta, 'variant_lower': delta < 0})
    exceptions = [p for p in pairs if not p['variant_lower']]

    summary: dict[str, Any] = {
        'images_total': len(images),
        'images_unreadable': len(images) - len(decoded),
        'status_counts_per_face_row': _count(rows, 'status'),
        'decision_counts_per_face_row': _count(rows, 'decision'),
        'detector_counts': _count(rows, 'detector'),
        'detection': {
            'images_decoded': len(decoded),
            'images_with_usable_face': len(with_face),
            'detection_success_rate': (len(with_face) / len(decoded)) if decoded else None,
            'images_using_fallback': len(fallback_images),
            'fallback_rate': (len(fallback_images) / len(decoded)) if decoded else None,
            'images_no_face': sum(r['status'] == 'NO_FACE' for r in decoded),
        },
        'timing_ms': {'mean': float(np.mean(times)) if times else None, 'median': float(np.median(times)) if times else None,
                      'p95': float(np.percentile(times, 95)) if times else None,
                      'note': 'includes the first call, which can include model warm-up'},
        'score_distribution_by_category': {k: score_stats(v) for k, v in sorted(by_category.items())},
        'paired_comparisons': pairs,
        'paired_summary': {'n_pairs': len(pairs), 'variant_lower': len(pairs) - len(exceptions), 'exceptions': exceptions,
                           'note': 'A variant is not guaranteed to score lower; exceptions are listed, not hidden. '
                                   'Faces are re-detected per variant, so boxes can differ.'},
    }
    if yaw_bins:
        edges = [0.0, *sorted(yaw_bins), float('inf')]
        groups: dict[str, list[float]] = {}
        for r in scored:
            if r['yaw'] is None:
                continue
            for lo, hi in zip(edges[:-1], edges[1:]):
                if lo <= abs(r['yaw']) < hi:
                    groups.setdefault(f'|yaw| in [{lo:g}, {hi:g})', []).append(r['quality_score'])
        summary['score_by_abs_yaw'] = {k: score_stats(v) for k, v in groups.items()}
    return summary


# ----------------------------------------------------------------------------- reports


def _cell(value: Any) -> Any:
    return json.dumps(value) if isinstance(value, (list, dict)) else value


def write_reports(rows: list[dict], csv_path: str | Path, json_path: str | Path) -> None:
    csv_path, json_path = Path(csv_path), Path(json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    fields = fields or ['image', 'detector', 'status', 'decision', 'quality_score', 'error_reason', 'processing_time_ms']
    with csv_path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, restval='')
        writer.writeheader()
        writer.writerows({k: _cell(v) for k, v in row.items()} for row in rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding='utf-8')


def main() -> None:
    args = parse_args()
    degradations = parse_degradations(args.degrade)
    labels = load_labels(args.labels)
    pipeline = pipeline_from_args(args)
    rows = run_batch(pipeline, expand_inputs(args.inputs), aligned=args.aligned, degradations=degradations,
                     labels=labels, save_variants=args.save_variants)
    write_reports(rows, args.csv, args.json)
    summary = build_summary(rows, args.yaw_bins)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding='utf-8')

    for row in rows:
        score = '-' if row['quality_score'] is None else f"{row['quality_score']:.4f}"
        print(f"{row['image']} category={row['category']} detector={row['detector']} status={row['status']} "
              f"decision={row['decision']} score={score} time_ms={row['processing_time_ms']:.1f}")
    det = summary['detection']
    print(f"\nimages={summary['images_total']} detection_success={det['images_with_usable_face']}/{det['images_decoded']} "
          f"fallback_used={det['images_using_fallback']} no_face={det['images_no_face']}")
    print(f"paired: {summary['paired_summary']['variant_lower']}/{summary['paired_summary']['n_pairs']} variants scored lower; "
          f"exceptions={len(summary['paired_summary']['exceptions'])}")
    print(f'CSV: {args.csv}\nJSON: {args.json}\nSummary: {args.summary}')


if __name__ == '__main__':
    main()
