"""Run from the repository root:  python -m unittest discover -s tests -v

Tests needing the eDifFIQA-S weights or sample images are skipped (not failed) when those are absent.
Real-inference tests never assert a specific score value and never assume a blurred image scores lower.
"""

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import batch_evaluate  # noqa: E402
import calibrate_thresholds  # noqa: E402
from face_detection import (  # noqa: E402
    DetectorConfig,
    DetectorSettings,
    FaceDetectorChain,
    bbox_iou,
    dedupe_faces,
    validate_face,
)
from quality_pipeline import EvaluationResult, QualityPipeline, QualityThresholds  # noqa: E402

IMAGES = ROOT / 'assets' / 'test_images'
IMAGE = IMAGES / 'image2.jpg'
WEIGHTS = ROOT / 'weights' / 'ediffiqa_s.pth'
HAVE_REAL = WEIGHTS.exists() and IMAGE.exists()
needs_real = unittest.skipUnless(HAVE_REAL, 'needs weights/ediffiqa_s.pth and assets/test_images')


class FakeDetector:
    def __init__(self, faces):
        self.faces, self.calls = faces, 0

    def detect(self, image):
        self.calls += 1
        return self.faces


class FailingDetector:
    def detect(self, image):
        raise RuntimeError('detector unavailable')


def fake_face(**overrides):
    values = dict(
        confidence=0.9,
        bbox=np.array([47.0, 25.0, 181.0, 191.0], dtype=np.float32),
        landmarks=np.array([[87.0, 84.0], [141.0, 107.0], [100.0, 127.0], [71.0, 139.0], [118.0, 159.0]], dtype=np.float32),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def chain(primary_faces=None, fallback_faces=None, *, primary_fails=False, fallback_fails=False, fallback='retinaface', mode='fallback'):
    c = FaceDetectorChain(DetectorConfig(primary='scrfd', fallback=fallback, mode=mode))
    c.detectors['scrfd'] = FailingDetector() if primary_fails else FakeDetector(primary_faces or [])
    if fallback:
        c.detectors['retinaface'] = FailingDetector() if fallback_fails else FakeDetector(fallback_faces or [])
    return c


BLANK = np.full((221, 221, 3), 128, dtype=np.uint8)


# ----------------------------------------------------------------------------- no weights required


class ValidationTests(unittest.TestCase):
    def test_valid_face(self):
        self.assertIsNone(validate_face(fake_face(), 221, 221))

    def test_boundary_tolerance(self):
        # A face at image edge with landmarks centroid shifted by 1px outside bbox is accepted
        box = np.array([0.0, 0.0, 100.0, 100.0], dtype=np.float32)
        # Centroid (-1.0, 50.0) is within tolerance (tol_x = max(2.0, 100 * 0.05) = 5.0)
        kps = np.array([[-15.0, 30.0], [10.0, 30.0], [5.0, 50.0], [-5.0, 70.0], [0.0, 70.0]], dtype=np.float32)
        self.assertIsNone(validate_face(fake_face(bbox=box, landmarks=kps), 221, 221))

    def test_invalid_bboxes(self):
        for bbox in ([50, 50, 40, 90], [10, 10, 10, 90], [np.nan, 0, 10, 10], [1, 2, 3], [300, 300, 400, 400], [-50, -50, -1, -1]):
            with self.subTest(bbox=bbox):
                self.assertIsNotNone(validate_face(fake_face(bbox=np.array(bbox, dtype=np.float32)), 221, 221))

    def test_invalid_landmarks(self):
        bad = {
            'nan': np.full((5, 2), np.nan, dtype=np.float32),
            'wrong shape': np.zeros((4, 2), dtype=np.float32),
            'collapsed': np.tile(np.array([[100.0, 100.0]], dtype=np.float32), (5, 1)),
            'centroid outside bbox': fake_face().landmarks + 500,
        }
        for name, landmarks in bad.items():
            with self.subTest(case=name):
                self.assertIsNotNone(validate_face(fake_face(landmarks=landmarks), 221, 221))

    def test_invalid_confidence_and_malformed(self):
        self.assertIsNotNone(validate_face(fake_face(confidence=1.5), 221, 221))
        self.assertIsNotNone(validate_face(fake_face(confidence=float('nan')), 221, 221))
        self.assertIsNotNone(validate_face(SimpleNamespace(), 221, 221))

    def test_dedupe_keeps_highest_confidence(self):
        a, b = fake_face(confidence=0.7), fake_face(confidence=0.95)
        far = fake_face(bbox=np.array([0.0, 0.0, 20.0, 20.0], dtype=np.float32), confidence=0.6)
        kept, dropped = dedupe_faces([a, b, far], 0.5)
        self.assertEqual(dropped, 1)
        self.assertIn(b, kept)
        self.assertNotIn(a, kept)
        self.assertAlmostEqual(bbox_iou(np.array([0, 0, 10, 10.0]), np.array([0, 0, 10, 10.0])), 1.0)


class DetectorConfigTests(unittest.TestCase):
    def test_defaults_match_uniface_defaults(self):
        config = DetectorConfig()
        self.assertEqual((config.primary, config.fallback), ('scrfd', None))
        self.assertEqual(config.scrfd.confidence_threshold, 0.5)
        self.assertEqual(config.scrfd.input_size, (640, 640))

    def test_from_dict_and_errors(self):
        config = DetectorConfig.from_dict({'fallback': 'retinaface', 'scrfd': {'input_size': 320}})
        self.assertEqual(config.scrfd.input_size, (320, 320))
        for bad in ({'primary': 'mtcnn'}, {'fallback': 'scrfd'}, {'nonsense': 1}, {'scrfd': {'confidence_threshold': 2}},
                    {'scrfd': {'bogus': 1}}, {'scrfd': {'input_size': 0}}, {'dedupe_iou': 0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                DetectorConfig.from_dict(bad)

    def test_from_file_roundtrip(self):
        shipped = DetectorConfig.from_file(ROOT / 'detector_config.json')
        self.assertEqual(shipped, DetectorConfig())


class DetectorChainTests(unittest.TestCase):
    def test_primary_success_does_not_run_fallback(self):
        c = chain(primary_faces=[fake_face()], fallback_faces=[fake_face()])
        outcome = c.detect(BLANK)
        self.assertEqual(outcome.detector, 'scrfd')
        self.assertFalse(outcome.fallback_used)
        self.assertEqual(c.detectors['retinaface'].calls, 0)

    def test_fallback_after_primary_miss(self):
        outcome = chain(primary_faces=[], fallback_faces=[fake_face()]).detect(BLANK)
        self.assertEqual(outcome.detector, 'retinaface(fallback)')
        self.assertTrue(outcome.fallback_used)
        self.assertEqual(len(outcome.faces), 1)

    def test_fallback_after_primary_crash(self):
        outcome = chain(primary_fails=True, fallback_faces=[fake_face()]).detect(BLANK)
        self.assertEqual(outcome.detector, 'retinaface(fallback)')
        self.assertIsNone(outcome.error)
        self.assertIn('detection failed', outcome.note)

    def test_invalid_primary_detections_trigger_fallback(self):
        bad = fake_face(landmarks=np.full((5, 2), np.nan, dtype=np.float32))
        outcome = chain(primary_faces=[bad], fallback_faces=[fake_face()]).detect(BLANK)
        self.assertEqual(outcome.detector, 'retinaface(fallback)')
        self.assertIn('invalid', outcome.note)

    def test_neither_detector_finds_a_face(self):
        outcome = chain().detect(BLANK)
        self.assertEqual(outcome.faces, [])
        self.assertIsNone(outcome.error)  # a clean miss is not an error

    def test_errors_are_surfaced(self):
        self.assertIn('scrfd detection failed', chain(primary_fails=True, fallback=None).detect(BLANK).error)
        self.assertIn('fallback detection failed', chain(fallback_fails=True).detect(BLANK).error)
        both = chain(primary_fails=True, fallback_fails=True).detect(BLANK)
        self.assertIn('scrfd detection failed', both.error)
        self.assertIn('retinaface fallback detection failed', both.error)

    def test_duplicates_removed_within_detector(self):
        outcome = chain(primary_faces=[fake_face(confidence=0.7), fake_face(confidence=0.9)]).detect(BLANK)
        self.assertEqual(len(outcome.faces), 1)
        self.assertAlmostEqual(outcome.faces[0].confidence, 0.9)

    def test_complementary_mode_merges_and_deduplicates(self):
        c = chain(
            primary_faces=[fake_face(confidence=0.8)],
            fallback_faces=[
                fake_face(bbox=np.array([48.0, 26.0, 180.0, 190.0], dtype=np.float32), confidence=0.95),
                fake_face(
                    bbox=np.array([5.0, 5.0, 30.0, 30.0], dtype=np.float32),
                    landmarks=np.array([[10.0, 10.0], [20.0, 10.0], [15.0, 15.0], [10.0, 25.0], [20.0, 25.0]], dtype=np.float32),
                    confidence=0.6,
                ),
            ],
            mode='complementary',
        )
        outcome = c.detect(BLANK)
        self.assertEqual(outcome.detector, 'scrfd+retinaface(complementary)')
        self.assertTrue(outcome.fallback_used)
        self.assertEqual(len(outcome.faces), 2)
        self.assertAlmostEqual(outcome.faces[0].confidence, 0.95)
        self.assertIn('complementary merge', outcome.note)


class ThresholdTests(unittest.TestCase):
    def test_boundaries(self):
        t = QualityThresholds(accept_min=0.8, reject_max=0.3)
        self.assertEqual(t.decision(0.8), 'ACCEPT')
        self.assertEqual(t.decision(0.8000001), 'ACCEPT')
        self.assertEqual(t.decision(0.7999999), 'REVIEW')
        self.assertEqual(t.decision(0.3), 'REJECT')
        self.assertEqual(t.decision(0.3000001), 'REVIEW')
        self.assertEqual(t.decision(None), 'NOT_EVALUATED')

    def test_uncalibrated_is_not_evaluated(self):
        self.assertEqual(QualityThresholds().decision(0.9), 'NOT_EVALUATED')

    def test_single_sided(self):
        self.assertEqual(QualityThresholds(accept_min=0.5).decision(0.4), 'REVIEW')
        self.assertEqual(QualityThresholds(reject_max=0.5).decision(0.6), 'REVIEW')

    def test_invalid_configuration(self):
        for kwargs in ({'accept_min': 0.3, 'reject_max': 0.3}, {'accept_min': 0.3, 'reject_max': 0.8},
                       {'accept_min': float('nan')}, {'accept_min': True}, {'max_abs_yaw': 0}, {'max_abs_yaw': -5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                QualityThresholds(**kwargs)

    def test_from_file(self):
        self.assertEqual(QualityThresholds.from_file(ROOT / 'quality_thresholds.json'), QualityThresholds())
        with tempfile.TemporaryDirectory() as d:
            for name, text in {'bad.json': '{not json', 'list.json': '[1]', 'typo.json': '{"accept_minn": 0.5}',
                               'str.json': '{"accept_min": "0.5"}', 'order.json': '{"accept_min": 0.2, "reject_max": 0.5}'}.items():
                path = Path(d) / name
                path.write_text(text, encoding='utf-8')
                with self.subTest(file=name), self.assertRaises(ValueError):
                    QualityThresholds.from_file(path)
            ok = Path(d) / 'ok.json'
            ok.write_text('{"accept_min": 0.7, "reject_max": 0.2, "max_abs_yaw": 40}', encoding='utf-8')
            self.assertEqual(QualityThresholds.from_file(ok).max_abs_yaw, 40)

    def test_pose_limits_require_head_pose_enabled(self):
        with self.assertRaisesRegex(ValueError, 'head-pose'):
            QualityPipeline(thresholds=QualityThresholds(max_abs_yaw=30))


class DegradationAndSummaryTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.image = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)

    def test_degradations_are_controlled_and_deterministic(self):
        lap = lambda im: cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()  # noqa: E731
        blurred = batch_evaluate.degrade_blur(self.image, 2.0)
        self.assertEqual(blurred.shape, self.image.shape)
        self.assertLess(lap(blurred), lap(self.image))
        np.testing.assert_array_equal(blurred, batch_evaluate.degrade_blur(self.image, 2.0))
        self.assertEqual(batch_evaluate.degrade_lowres(self.image, 4).shape, self.image.shape)
        self.assertLess(batch_evaluate.degrade_dark(self.image, 0.3).mean(), self.image.mean())

    def test_parse_degradations(self):
        self.assertEqual(batch_evaluate.parse_degradations(['blur:3', 'dark:0.5']), [('blur', 3.0), ('dark', 0.5)])
        for bad in ('sharpen:2', 'blur:x', 'blur:-1', 'dark:1.5', 'blur'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                batch_evaluate.parse_degradations([bad])

    def test_classify_path(self):
        self.assertEqual(batch_evaluate.classify_path('a/image2_blur.jpg'), ('blur', 'image2'))
        self.assertEqual(batch_evaluate.classify_path('a/image2_lowres.jpg'), ('lowres', 'image2'))
        self.assertEqual(batch_evaluate.classify_path('a/image2.jpg'), ('original', 'image2'))

    @staticmethod
    def _row(image, category, score, *, detector='scrfd', fallback=False, status='OK', source=None):
        return dict(image=image, category=category, source_image=source or image.split('::')[0], variant='v', quality_score=score,
                    status=status, decision='NOT_EVALUATED', detector=detector, fallback_used=fallback, confidence=0.9 if status == 'OK' else None,
                    face_index=1 if status == 'OK' else None, processing_time_ms=10.0, yaw=None)

    def test_summary_counts_exceptions_and_does_not_hide_them(self):
        rows = [
            self._row('a', 'original', 0.8, source='a'),
            self._row('a::b', 'blur', 0.4, source='a'),
            self._row('a::l', 'lowres', 0.85, source='a'),  # exception: higher than original
            self._row('c', 'original', None, detector='retinaface(fallback)', fallback=True, status='NO_FACE', source='c'),
            dict(self._row('d', 'original', None, status='READ_ERROR', source='d'), face_index=None),
        ]
        s = batch_evaluate.build_summary(rows)
        self.assertEqual(s['paired_summary']['n_pairs'], 2)
        self.assertEqual(s['paired_summary']['variant_lower'], 1)
        self.assertEqual(len(s['paired_summary']['exceptions']), 1)
        self.assertEqual(s['images_unreadable'], 1)
        self.assertEqual(s['detection']['images_no_face'], 1)
        self.assertEqual(s['detection']['images_using_fallback'], 1)
        self.assertEqual(s['score_distribution_by_category']['original']['n'], 1)

    def test_csv_and_json_reports(self):
        rows = [dict(image='x.jpg', bbox=[1.0, 2.0, 3.0, 4.0], status='NO_FACE', decision='NOT_EVALUATED', quality_score=None)]
        with tempfile.TemporaryDirectory() as d:
            csv_path, json_path = Path(d) / 'sub' / 'r.csv', Path(d) / 'sub' / 'r.json'
            batch_evaluate.write_reports(rows, csv_path, json_path)
            with csv_path.open(newline='', encoding='utf-8') as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(json.loads(csv_rows[0]['bbox']), [1.0, 2.0, 3.0, 4.0])
            self.assertEqual(csv_rows[0]['status'], 'NO_FACE')
            self.assertEqual(json.loads(json_path.read_text(encoding='utf-8'))[0]['decision'], 'NOT_EVALUATED')
            batch_evaluate.write_reports([], Path(d) / 'e.csv', Path(d) / 'e.json')  # empty report still valid
            self.assertTrue((Path(d) / 'e.csv').read_text(encoding='utf-8').startswith('image'))

    def test_result_schema_rejects_unknown_labels(self):
        with self.assertRaises(ValueError):
            EvaluationResult('i', 1, 1, 'd', None, None, None, None, 'WEIRD', 'ACCEPT', None, 0.0)
        with self.assertRaises(ValueError):
            EvaluationResult('i', 1, 1, 'd', None, None, None, None, 'OK', 'MAYBE', None, 0.0)


class CalibrationTests(unittest.TestCase):
    SAMPLES = [(0.9, 'usable'), (0.8, 'usable'), (0.4, 'unusable'), (0.3, 'usable'), (0.2, 'unusable')]

    def test_sweep_metrics_by_hand(self):
        row = next(r for r in calibrate_thresholds.sweep(self.SAMPLES) if r['threshold'] == 0.8)
        self.assertEqual((row['tp'], row['fp'], row['fn'], row['tn']), (2, 0, 1, 2))
        self.assertAlmostEqual(row['precision'], 1.0)
        self.assertAlmostEqual(row['recall'], 2 / 3)
        self.assertAlmostEqual(row['f1'], 0.8)
        self.assertAlmostEqual(row['false_rejection_rate'], 1 / 3)
        self.assertAlmostEqual(row['false_acceptance_rate'], 0.0)
        self.assertAlmostEqual(row['pct_accepted'], 2 / 5)

    def test_proposal_bands_and_percentages(self):
        p = calibrate_thresholds.propose(self.SAMPLES, target_precision=1.0, max_false_rejection=0.0)
        # usable: 0.9, 0.8, 0.3 ; unusable: 0.4, 0.2.  Precision 1.0 first holds at t=0.8.
        # No usable face may fall in the reject band -> largest such t is 0.2.
        self.assertEqual((p['accept_min'], p['reject_max']), (0.8, 0.2))
        self.assertAlmostEqual(p['pct_accept'], 0.4)
        self.assertAlmostEqual(p['pct_reject'], 0.2)
        self.assertAlmostEqual(p['pct_review'], 0.4)

    def test_pending_without_labels_or_enough_samples(self):
        with tempfile.TemporaryDirectory() as d:
            results = Path(d) / 'r.csv'
            results.write_text('image,status,quality_score,confidence,label\na.jpg,OK,0.9,0.9,\nb.jpg,NO_FACE,,,usable\n', encoding='utf-8')
            samples, skipped = calibrate_thresholds.load_samples(results)
            self.assertEqual(samples, [])
            self.assertEqual((skipped['no_label'], skipped['not_scored']), (1, 1))
            good = Path(d) / 'g.csv'
            good.write_text('image,status,quality_score,confidence,label\n' + '\n'.join(
                f'{i}.jpg,OK,{0.1 * i},0.9,{"usable" if i % 2 else "unusable"}' for i in range(1, 10)) + '\n', encoding='utf-8')
            out = Path(d) / 'p.json'
            argv = ['x', str(good), '--sweep-csv', str(Path(d) / 's.csv'), '--proposal-json', str(out), '--target-precision', '0.9', '--max-false-rejection', '0.1']
            with patch.object(sys, 'argv', argv), patch('builtins.print'):
                calibrate_thresholds.main()
            proposal = json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual(proposal['status'], 'PENDING')  # 9 samples < min-per-class
            self.assertNotIn('accept_min', proposal)


# ----------------------------------------------------------------------------- need weights (+ images)


@needs_real
class PipelineMockedDetectorTests(unittest.TestCase):
    """Real eDifFIQA-S model, scripted detectors, to exercise every failure path."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = QualityPipeline(variant='s', device='cpu', fallback_detector='retinaface')
        cls.original_config = cls.pipeline.detection.config

    def setUp(self):
        self.pipeline.detection.config = self.original_config  # undo per-test config changes
        self.pipeline.detectors = {}
        self.pipeline.thresholds = QualityThresholds()
        self.pipeline.pose_checker = None

    def use(self, **kwargs):
        self.pipeline.detectors = chain(**kwargs).detectors

    def test_read_errors(self):
        missing = self.pipeline.evaluate(IMAGES / 'does_not_exist.jpg')[0]
        self.assertEqual((missing.status, missing.decision), ('READ_ERROR', 'NOT_EVALUATED'))
        self.assertIsNone(self.pipeline.evaluate(IMAGES)[0].quality_score)  # a directory
        with tempfile.NamedTemporaryFile(suffix='.jpg') as invalid:
            invalid.write(b'not an image')
            invalid.flush()
            self.assertEqual(self.pipeline.evaluate(invalid.name)[0].status, 'READ_ERROR')
        with tempfile.NamedTemporaryFile(suffix='.jpg') as empty:
            self.assertEqual(self.pipeline.evaluate(empty.name)[0].status, 'READ_ERROR')

    def test_fallback_after_scrfd_miss(self):
        self.use(primary_faces=[], fallback_faces=[fake_face()])
        r = self.pipeline.evaluate(IMAGE)[0]
        self.assertEqual((r.status, r.detector, r.fallback_used), ('OK', 'retinaface(fallback)', True))
        self.assertTrue(math.isfinite(r.quality_score))
        self.assertAlmostEqual(r.confidence, 0.9)
        self.assertEqual(r.bbox, [47.0, 25.0, 181.0, 191.0])

    def test_fallback_after_scrfd_crash(self):
        self.use(primary_fails=True, fallback_faces=[fake_face()])
        r = self.pipeline.evaluate(IMAGE)[0]
        self.assertEqual((r.status, r.detector), ('OK', 'retinaface(fallback)'))

    def test_no_face_is_not_a_quality_verdict(self):
        self.use()
        r = self.pipeline.evaluate_array(BLANK, name='blank')[0]
        self.assertEqual((r.status, r.decision), ('NO_FACE', 'NOT_EVALUATED'))
        self.assertIsNone(r.quality_score)
        self.assertIn('no usable face', r.error_reason)
        self.assertNotIn('blur', (r.error_reason + str(r.decision_reason)).lower())

    def test_detector_errors_are_model_errors_not_no_face(self):
        self.use(primary_fails=True, fallback_fails=True)
        r = self.pipeline.evaluate(IMAGE)[0]
        self.assertEqual((r.status, r.decision), ('MODEL_ERROR', 'NOT_EVALUATED'))

    def test_invalid_detections_are_rejected_before_scoring(self):
        bad = [fake_face(bbox=np.array([50, 50, 40, 90], dtype=np.float32)), fake_face(landmarks=np.full((5, 2), np.nan, dtype=np.float32))]
        self.use(primary_faces=bad, fallback=None)
        self.pipeline.detection.config = DetectorConfig()
        r = self.pipeline.evaluate(IMAGE)[0]
        self.assertEqual(r.status, 'NO_FACE')
        self.assertIn('invalid', r.error_reason)

    def test_alignment_failures(self):
        self.use(primary_faces=[fake_face()])
        with patch('quality_pipeline.face_alignment', side_effect=ValueError('bad landmarks')):
            r = self.pipeline.evaluate(IMAGE)[0]
        self.assertEqual((r.status, r.decision), ('ALIGNMENT_ERROR', 'NOT_EVALUATED'))
        self.assertIsNone(r.quality_score)
        self.assertEqual(r.bbox, [47.0, 25.0, 181.0, 191.0])  # detection info is preserved
        for output, label in ((np.zeros((112, 112, 3), np.uint8), 'blank'), (np.zeros((50, 50, 3), np.uint8), 'wrong size')):
            with self.subTest(case=label), patch('quality_pipeline.face_alignment', return_value=(output, np.eye(2, 3))):
                self.assertEqual(self.pipeline.evaluate(IMAGE)[0].status, 'ALIGNMENT_ERROR')
        with patch('quality_pipeline.face_alignment', return_value=(np.full((112, 112, 3), 7, np.uint8), np.full((2, 3), np.nan))):
            self.assertEqual(self.pipeline.evaluate(IMAGE)[0].status, 'ALIGNMENT_ERROR')

    def test_model_failures_are_reported(self):
        self.use(primary_faces=[fake_face()])
        with patch.object(self.pipeline, '_score_aligned', side_effect=RuntimeError('inference failed')):
            r = self.pipeline.evaluate(IMAGE)[0]
        self.assertEqual((r.status, r.decision, r.quality_score), ('MODEL_ERROR', 'NOT_EVALUATED', None))
        with patch.object(self.pipeline, 'model', lambda x: torch.tensor([[float('nan')]])):
            r = self.pipeline.evaluate(IMAGE)[0]
        self.assertEqual(r.status, 'MODEL_ERROR')
        self.assertIn('non-finite', r.error_reason)

    def test_model_loading_errors_are_reported_to_caller(self):
        with self.assertRaisesRegex(RuntimeError, 'Failed to load eDifFIQA-S'):
            QualityPipeline(variant='s', device='cpu', weights=ROOT / 'weights' / 'missing.pth')

    def test_aligned_input_requirements(self):
        good = np.random.default_rng(1).integers(0, 255, (112, 112, 3), dtype=np.uint8)
        ok = self.pipeline.evaluate_array(good, name='aligned', aligned=True)[0]
        self.assertEqual((ok.status, ok.detector), ('OK', 'aligned'))
        self.assertTrue(math.isfinite(ok.quality_score))
        for name, bad in {'224x224': np.zeros((224, 224, 3), np.uint8), 'blank': np.zeros((112, 112, 3), np.uint8),
                          'float': good.astype(np.float32), 'gray': good[:, :, 0]}.items():
            with self.subTest(case=name):
                self.assertEqual(self.pipeline.evaluate_array(bad, name=name, aligned=True)[0].status, 'ALIGNMENT_ERROR')

    def test_decisions_and_reasons_use_thresholds(self):
        self.use(primary_faces=[fake_face()])
        score = self.pipeline.evaluate(IMAGE)[0].quality_score
        for thresholds, expected in ((QualityThresholds(accept_min=score), 'ACCEPT'), (QualityThresholds(reject_max=score), 'REJECT'),
                                     (QualityThresholds(accept_min=score + 0.01, reject_max=score - 0.01), 'REVIEW')):
            with self.subTest(expected=expected):
                self.pipeline.thresholds = thresholds
                r = self.pipeline.evaluate(IMAGE)[0]
                self.assertEqual(r.decision, expected)
                self.assertTrue(r.decision_reason)

    def test_pose_limits_review_but_never_override_reject_and_never_hide_score(self):
        self.use(primary_faces=[fake_face()])
        score = self.pipeline.evaluate(IMAGE)[0].quality_score
        pose = lambda yaw: SimpleNamespace(estimate=lambda crop: SimpleNamespace(pitch=0.0, yaw=yaw, roll=0.0))  # noqa: E731
        self.pipeline.thresholds = QualityThresholds(accept_min=score - 0.1, max_abs_yaw=30)
        from pose_check import HeadPoseChecker

        self.pipeline.pose_checker = HeadPoseChecker(estimator=pose(10.0))
        self.assertEqual(self.pipeline.evaluate(IMAGE)[0].decision, 'ACCEPT')
        self.pipeline.pose_checker = HeadPoseChecker(estimator=pose(45.0))
        r = self.pipeline.evaluate(IMAGE)[0]
        self.assertEqual((r.decision, r.pose_status, r.yaw), ('REVIEW', 'OK', 45.0))
        self.assertIn('yaw', r.decision_reason)
        self.assertEqual(r.quality_score, score)
        self.pipeline.thresholds = QualityThresholds(reject_max=score + 0.1, accept_min=score + 0.2, max_abs_yaw=30)
        self.assertEqual(self.pipeline.evaluate(IMAGE)[0].decision, 'REJECT')
        self.pipeline.thresholds = QualityThresholds(accept_min=score - 0.1, max_abs_yaw=30)
        self.pipeline.pose_checker = HeadPoseChecker(estimator=SimpleNamespace(estimate=lambda crop: (_ for _ in ()).throw(RuntimeError('boom'))))
        r = self.pipeline.evaluate(IMAGE)[0]
        self.assertEqual((r.status, r.pose_status, r.decision), ('OK', 'ERROR', 'REVIEW'))
        self.assertEqual(r.quality_score, score)

    def test_multi_face_rows_keep_separate_geometry(self):
        second = fake_face(bbox=np.array([5.0, 5.0, 40.0, 60.0], dtype=np.float32), confidence=0.8,
                           landmarks=np.array([[15, 20], [30, 20], [22, 32], [17, 45], [28, 45]], dtype=np.float32))
        self.use(primary_faces=[fake_face(), second])
        rows = self.pipeline.evaluate(IMAGE)
        self.assertEqual([r.face_index for r in rows], [1, 2])
        self.assertNotEqual(rows[0].bbox, rows[1].bbox)


@needs_real
class RealInferenceTests(unittest.TestCase):
    """Real detectors + real eDifFIQA-S on the sample images. Asserts behaviour, never exact scores."""

    @classmethod
    def setUpClass(cls):
        cls.default = QualityPipeline(variant='s', device='cpu')
        cls.fallback = QualityPipeline(variant='s', device='cpu', fallback_detector='retinaface')
        cls.small = QualityPipeline(variant='s', device='cpu', detector_config=DetectorConfig(scrfd=DetectorSettings(input_size=320)))

    def test_scrfd_640_miss_is_reported_as_no_face_not_as_quality(self):
        r = self.default.evaluate(IMAGE)[0]
        self.assertEqual((r.status, r.decision, r.quality_score), ('NO_FACE', 'NOT_EVALUATED', None))

    def test_retinaface_fallback_scores_the_portrait(self):
        r = self.fallback.evaluate(IMAGE)[0]
        self.assertEqual((r.status, r.detector, r.fallback_used), ('OK', 'retinaface(fallback)', True))
        self.assertTrue(math.isfinite(r.quality_score))
        self.assertGreater(r.confidence, 0.5)
        self.assertGreater(r.face_area_ratio, 0)

    def test_scrfd_input_size_320_detects_without_fallback(self):
        r = self.small.evaluate(IMAGE)[0]
        self.assertEqual((r.status, r.detector, r.fallback_used), ('OK', 'scrfd', False))

    def test_fallback_not_used_when_scrfd_succeeds(self):
        path = IMAGES / 'image5.jpg'
        if not path.exists():
            self.skipTest('image5.jpg missing')
        r = self.fallback.evaluate(path)[0]
        self.assertEqual((r.status, r.detector, r.fallback_used), ('OK', 'scrfd', False))

    def test_blurred_and_lowres_images_get_real_scores_without_assuming_order(self):
        for name in ('image2_blur.jpg', 'image2_lowres.jpg', 'image3_blur.jpg', 'image3_lowres.jpg'):
            path = IMAGES / name
            if not path.exists():
                continue
            with self.subTest(image=name):
                r = self.fallback.evaluate(path)[0]
                self.assertEqual(r.status, 'OK')  # a detector miss must never masquerade as "blurry"
                self.assertTrue(math.isfinite(r.quality_score))

    def test_blank_image_is_no_face(self):
        r = self.fallback.evaluate_array(BLANK, name='blank')[0]
        self.assertEqual((r.status, r.quality_score), ('NO_FACE', None))

    def test_aligned_crop_scores_identically_to_full_pipeline(self):
        from uniface import face_alignment

        image = cv2.imread(str(IMAGE))
        full = self.fallback.evaluate(IMAGE)[0]
        face = self.fallback.detection.detect(image).faces[0]
        crop, _ = face_alignment(image, np.asarray(face.landmarks, dtype=np.float32))
        aligned = self.fallback.evaluate_array(crop, name='crop', aligned=True)[0]
        self.assertAlmostEqual(full.quality_score, aligned.quality_score, places=5)

    def test_real_head_pose(self):
        pipeline = QualityPipeline(variant='s', device='cpu', fallback_detector='retinaface', head_pose=True)
        r = pipeline.evaluate(IMAGE)[0]
        self.assertEqual((r.status, r.pose_status), ('OK', 'OK'))
        self.assertTrue(all(math.isfinite(v) for v in (r.pitch, r.yaw, r.roll)))

    def test_photometrics_computed_on_real_image(self):
        pipeline = QualityPipeline(variant='s', device='cpu', fallback_detector='retinaface')
        r = pipeline.evaluate(IMAGE)[0]
        self.assertEqual(r.status, 'OK')
        self.assertIsNotNone(r.sharpness_laplacian_var)
        self.assertGreater(r.sharpness_laplacian_var, 0)
        self.assertIsNotNone(r.mean_luminance)
        self.assertGreater(r.mean_luminance, 0)
        self.assertIsNotNone(r.underexposure_ratio)
        self.assertIsNotNone(r.overexposure_ratio)
        self.assertIsNotNone(r.illumination_asymmetry)

    def test_real_gaze_estimation(self):
        pipeline = QualityPipeline(variant='s', device='cpu', fallback_detector='retinaface', gaze=True)
        r = pipeline.evaluate(IMAGE)[0]
        self.assertEqual(r.status, 'OK')
        self.assertEqual(r.gaze_status, 'OK')
        self.assertIsNotNone(r.gaze_pitch)
        self.assertIsNotNone(r.gaze_yaw)
        self.assertTrue(math.isfinite(r.gaze_pitch))
        self.assertTrue(math.isfinite(r.gaze_yaw))

    def test_gaze_disabled_by_default(self):
        pipeline = QualityPipeline(variant='s', device='cpu', fallback_detector='retinaface', gaze=False)
        r = pipeline.evaluate(IMAGE)[0]
        self.assertEqual(r.status, 'OK')
        self.assertEqual(r.gaze_status, 'DISABLED')
        self.assertIsNone(r.gaze_pitch)

    def test_batch_run_summary_and_reports(self):
        with tempfile.TemporaryDirectory() as d:
            rows = batch_evaluate.run_batch(self.fallback, [str(IMAGE), str(IMAGES / 'missing.jpg')],
                                            degradations=batch_evaluate.parse_degradations(['blur:3', 'lowres:4']))
            batch_evaluate.write_reports(rows, Path(d) / 'r.csv', Path(d) / 'r.json')
            summary = batch_evaluate.build_summary(rows)
            self.assertEqual(len(rows), 4)  # original + 2 variants + read error
            self.assertEqual(summary['images_unreadable'], 1)
            self.assertEqual(summary['paired_summary']['n_pairs'], 2)
            with (Path(d) / 'r.csv').open(newline='', encoding='utf-8') as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)


if __name__ == '__main__':
    unittest.main()
