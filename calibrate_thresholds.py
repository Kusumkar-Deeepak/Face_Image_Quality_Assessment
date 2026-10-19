"""Threshold calibration from LABELED data. Produces evidence; never edits quality_thresholds.json.

Input: the CSV written by batch_evaluate.py (run it with --labels labels.csv so each row carries ``label``).
Labels: ``usable`` (face good enough for recognition) or ``unusable``. Labelling is YOUR ground truth; this tool
cannot create it. Rows whose status is not OK (NO_FACE, READ_ERROR, ...) are counted but never scored.

Positive class = ``usable``. A face is predicted usable when ``score >= t``.
  precision  = TP / (TP + FP)            recall = TP / (TP + FN)           F1
  false_rejection_rate = FN / (TP + FN)  (usable faces that would be rejected)
  false_acceptance_rate = FP / (FP + TN) (unusable faces that would be accepted)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

VALID_LABELS = {'usable', 'unusable'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Threshold sweep and provisional calibration from labeled results')
    parser.add_argument('results_csv', help='CSV from batch_evaluate.py containing a "label" column')
    parser.add_argument('--sweep-csv', default='results/threshold_sweep.csv')
    parser.add_argument('--proposal-json', default='results/threshold_proposal.json')
    parser.add_argument('--min-per-class', type=int, default=50, help='Minimum labeled samples per class before any proposal is made')
    parser.add_argument('--target-precision', type=float, default=None, help='Required precision of ACCEPT (usable) decisions')
    parser.add_argument('--max-false-rejection', type=float, default=None, help='Max fraction of usable faces allowed in the REJECT band')
    return parser.parse_args()


def load_samples(path: str | Path) -> tuple[list[tuple[float, str]], dict[str, int]]:
    """One (score, label) per image (most confident face). Returns samples and skip counters."""
    best: dict[str, dict] = {}
    skipped = {'no_label': 0, 'invalid_label': 0, 'not_scored': 0}
    with Path(path).open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            label = (row.get('label') or '').strip().lower()
            if not label:
                skipped['no_label'] += 1
            elif label not in VALID_LABELS:
                skipped['invalid_label'] += 1
            elif row.get('status') != 'OK' or not row.get('quality_score'):
                skipped['not_scored'] += 1
            else:
                conf = float(row['confidence']) if row.get('confidence') else -1.0
                if row['image'] not in best or conf > best[row['image']]['conf']:
                    best[row['image']] = {'conf': conf, 'score': float(row['quality_score']), 'label': label}
    return [(v['score'], v['label']) for v in best.values()], skipped


def sweep(samples: list[tuple[float, str]]) -> list[dict[str, Any]]:
    n_pos = sum(label == 'usable' for _, label in samples)
    n_neg = len(samples) - n_pos
    rows = []
    for t in sorted({score for score, _ in samples}):
        tp = sum(s >= t and l == 'usable' for s, l in samples)
        fp = sum(s >= t and l == 'unusable' for s, l in samples)
        fn, tn = n_pos - tp, n_neg - fp
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / n_pos if n_pos else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else (0.0 if precision is not None and recall is not None else None)
        rows.append({'threshold': t, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': precision, 'recall': recall, 'f1': f1,
                     'false_rejection_rate': fn / n_pos if n_pos else None,
                     'false_acceptance_rate': fp / n_neg if n_neg else None,
                     'pct_accepted': (tp + fp) / len(samples)})
    return rows


def propose(samples: list[tuple[float, str]], target_precision: float, max_false_rejection: float) -> dict[str, Any]:
    n_pos = sum(label == 'usable' for _, label in samples)
    rows = sweep(samples)
    ok = [r for r in rows if r['precision'] is not None and r['precision'] >= target_precision]
    accept_min = min((r['threshold'] for r in ok), default=None)  # lowest t meeting precision -> highest recall
    reject_max = None
    for t in sorted({s for s, _ in samples}):
        if sum(s <= t and l == 'usable' for s, l in samples) / n_pos <= max_false_rejection:
            reject_max = t
    if accept_min is not None and reject_max is not None and reject_max >= accept_min:
        reject_max = None  # bands would overlap; refuse rather than silently pick one
    n = len(samples)
    pct = lambda pred: sum(pred(s) for s, _ in samples) / n  # noqa: E731
    return {
        'accept_min': accept_min, 'reject_max': reject_max,
        'pct_accept': None if accept_min is None else pct(lambda s: s >= accept_min),
        'pct_reject': None if reject_max is None else pct(lambda s: s <= reject_max),
        'pct_review': None if accept_min is None or reject_max is None else pct(lambda s: reject_max < s < accept_min),
    }


def main() -> None:
    args = parse_args()
    samples, skipped = load_samples(args.results_csv)
    n_pos = sum(label == 'usable' for _, label in samples)
    n_neg = len(samples) - n_pos
    print(f'labeled+scored samples: {len(samples)} (usable={n_pos}, unusable={n_neg}); skipped={skipped}')
    if not samples:
        print('CALIBRATION PENDING: no labeled, scored samples. Provide --labels to batch_evaluate.py first.')
        return
    rows = sweep(samples)
    Path(args.sweep_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.sweep_csv).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'sweep written: {args.sweep_csv}')

    proposal: dict[str, Any] = {'status': 'PENDING', 'n_usable': n_pos, 'n_unusable': n_neg, 'skipped': skipped,
                                'min_per_class': args.min_per_class}
    if min(n_pos, n_neg) < args.min_per_class:
        proposal['reason'] = f'fewer than {args.min_per_class} samples in a class; thresholds not proposed'
    elif args.target_precision is None or args.max_false_rejection is None:
        proposal['reason'] = 'supply --target-precision and --max-false-rejection (operational requirements, not defaults)'
    else:
        proposal.update(propose(samples, args.target_precision, args.max_false_rejection),
                        status='PROVISIONAL', target_precision=args.target_precision, max_false_rejection=args.max_false_rejection,
                        caveat='Estimated on this labeled set only; validate on held-out surveillance data before use.')
    Path(args.proposal_json).write_text(json.dumps(proposal, indent=2), encoding='utf-8')
    print(json.dumps(proposal, indent=2))


if __name__ == '__main__':
    main()
