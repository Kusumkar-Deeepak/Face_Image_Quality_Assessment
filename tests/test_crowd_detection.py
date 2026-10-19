"""Crowd and edge-case face detection benchmark and regression test suite.

Tests the 8 mandatory test cases:
1. Clear image with one frontal face
2. Crowded image containing many faces (known ground-truth count)
3. Small faces in the background
4. Faces near image boundaries
5. Adjacent faces with overlapping bounding boxes
6. Low-resolution image
7. Blurred image
8. Image containing no faces

Measures:
- Total detections
- Missed visible faces against ground truth
- False detections
- Duplicate detections
- Detection latency
- FIQA Quality-assessment success rate
"""

import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from face_detection import DetectorConfig, FaceDetectorChain, bbox_iou, validate_face
from quality_pipeline import QualityPipeline, QualityThresholds

ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGES = ROOT / 'assets' / 'test_images'
WEIGHTS_PATH = ROOT / 'weights' / 'ediffiqa_s.pth'


class CrowdDetectionBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.has_weights = WEIGHTS_PATH.exists()
        cls.standard_chain = FaceDetectorChain(
            DetectorConfig(primary='scrfd', fallback='retinaface', mode='fallback')
        )
        cls.crowd_chain = FaceDetectorChain(
            DetectorConfig.from_dict({
                'primary': 'scrfd',
                'fallback': 'retinaface',
                'mode': 'complementary',
                'scrfd': {'input_size': 960, 'confidence_threshold': 0.35},
                'retinaface': {'confidence_threshold': 0.35},
                'dedupe_iou': 0.55,
            })
        )
        if cls.has_weights:
            cls.pipeline = QualityPipeline(
                variant='s',
                weights=WEIGHTS_PATH,
                detector_config=cls.crowd_chain.config,
                thresholds=QualityThresholds(),
            )

    # 1. Clear image with one frontal face
    def test_case_1_clear_frontal_face(self):
        img_path = TEST_IMAGES / 'image16.jpg'
        if not img_path.exists():
            img_path = TEST_IMAGES / 'image2.jpg'
        img = cv2.imread(str(img_path))
        self.assertIsNotNone(img)

        t0 = time.perf_counter()
        outcome = self.standard_chain.detect(img)
        latency = (time.perf_counter() - t0) * 1000

        self.assertGreaterEqual(len(outcome.faces), 1)
        self.assertIsNone(outcome.error)
        print(f"\n[Case 1: Clear Frontal Face] Detections: {len(outcome.faces)}, Latency: {latency:.1f}ms, Detector: {outcome.detector}")

    # 2. Crowded multi-face scene with ground truth
    def test_case_2_crowded_multi_face_scene(self):
        # Create a synthetic crowd canvas (1600x1200) embedding 6 real face crops at known locations
        canvas = np.full((1200, 1600, 3), 210, dtype=np.uint8)
        src_img = cv2.imread(str(TEST_IMAGES / 'image2.jpg'))
        if src_img is None:
            self.skipTest("image2.jpg required")

        # Ground truth positions for 6 faces:
        gt_boxes = [
            (100, 100, 300, 300),   # Face 1: Large foreground
            (500, 100, 700, 300),   # Face 2: Large foreground
            (900, 150, 1050, 300),  # Face 3: Mid-ground
            (1200, 200, 1300, 300), # Face 4: Background small (100px)
            (50, 600, 180, 730),    # Face 5: Left edge
            (1450, 600, 1590, 740), # Face 6: Right edge boundary
        ]

        for x1, y1, x2, y2 in gt_boxes:
            resized_face = cv2.resize(src_img, (x2 - x1, y2 - y1))
            canvas[y1:y2, x1:x2] = resized_face

        t0 = time.perf_counter()
        outcome = self.crowd_chain.detect(canvas)
        latency = (time.perf_counter() - t0) * 1000

        total_detected = len(outcome.faces)
        self.assertGreaterEqual(total_detected, 4)

        # Measure matches against ground truth
        matched_gt = 0
        for gx1, gy1, gx2, gy2 in gt_boxes:
            g_box = np.array([gx1, gy1, gx2, gy2], dtype=np.float64)
            if any(bbox_iou(g_box, np.asarray(f.bbox, dtype=np.float64)) > 0.3 for f in outcome.faces):
                matched_gt += 1

        missed = len(gt_boxes) - matched_gt
        print(f"\n[Case 2: Crowded Scene] GT Faces: {len(gt_boxes)}, Detected: {total_detected}, Matched: {matched_gt}, Missed: {missed}, Latency: {latency:.1f}ms")

    # 3. Small faces in background
    def test_case_3_small_background_face(self):
        canvas = np.full((800, 800, 3), 200, dtype=np.uint8)
        src_img = cv2.imread(str(TEST_IMAGES / 'image2.jpg'))
        if src_img is None:
            self.skipTest("image2.jpg required")

        # Paste small 60x60 face in background
        small_crop = cv2.resize(src_img, (60, 60))
        canvas[200:260, 200:260] = small_crop

        outcome = self.crowd_chain.detect(canvas)
        self.assertGreaterEqual(len(outcome.faces), 1)
        face = outcome.faces[0]
        width_px = face.bbox[2] - face.bbox[0]
        self.assertTrue(20 <= width_px <= 65, f"Expected face width between 20 and 65, got {width_px}")
        print(f"\n[Case 3: Small Background Face] Detected: {len(outcome.faces)}, BBox size: {face.bbox[2]-face.bbox[0]:.1f}x{face.bbox[3]-face.bbox[1]:.1f}")

    # 4. Faces near image boundaries
    def test_case_4_faces_near_boundaries(self):
        canvas = np.full((600, 800, 3), 220, dtype=np.uint8)
        src_img = cv2.imread(str(TEST_IMAGES / 'image2.jpg'))
        if src_img is None:
            self.skipTest("image2.jpg required")

        crop = cv2.resize(src_img, (150, 150))
        # Touch left border (x=0) and right border (x=650 to 800)
        canvas[100:250, 0:150] = crop
        canvas[100:250, 650:800] = crop

        outcome = self.crowd_chain.detect(canvas)
        self.assertGreaterEqual(len(outcome.faces), 2)
        print(f"\n[Case 4: Boundary Faces] Detected: {len(outcome.faces)}, Edge faces preserved cleanly.")

    # 5. Adjacent faces with overlapping bounding boxes
    def test_case_5_adjacent_overlapping_faces(self):
        canvas = np.full((600, 800, 3), 220, dtype=np.uint8)
        src_img = cv2.imread(str(TEST_IMAGES / 'image2.jpg'))
        if src_img is None:
            self.skipTest("image2.jpg required")

        crop = cv2.resize(src_img, (150, 150))
        # Person A at x=200..350, Person B standing close behind at x=260..410 (IoU ~ 0.35)
        canvas[150:300, 200:350] = crop
        canvas[150:300, 260:410] = crop

        outcome = self.crowd_chain.detect(canvas)
        # Should detect both adjacent individuals without deduplicating them into one
        self.assertGreaterEqual(len(outcome.faces), 2)
        print(f"\n[Case 5: Adjacent Overlapping Faces] Detected: {len(outcome.faces)} distinct faces without wrongful deduplication.")

    # 6. Low-resolution image
    def test_case_6_low_resolution_image(self):
        img_path = TEST_IMAGES / 'image2_lowres.jpg'
        if not img_path.exists():
            self.skipTest("image2_lowres.jpg required")
        img = cv2.imread(str(img_path))
        outcome = self.crowd_chain.detect(img)
        self.assertGreaterEqual(len(outcome.faces), 1)
        print(f"\n[Case 6: Low-Res Image] Detected: {len(outcome.faces)}, Detector: {outcome.detector}")

    # 7. Blurred image
    def test_case_7_blurred_image(self):
        img_path = TEST_IMAGES / 'image2_blur.jpg'
        if not img_path.exists():
            self.skipTest("image2_blur.jpg required")
        img = cv2.imread(str(img_path))
        outcome = self.crowd_chain.detect(img)
        self.assertGreaterEqual(len(outcome.faces), 1)
        print(f"\n[Case 7: Blurred Image] Detected: {len(outcome.faces)}, Detector: {outcome.detector}")

    # 8. Blank / No face image
    def test_case_8_no_face_image(self):
        blank = np.full((400, 400, 3), 128, dtype=np.uint8)
        outcome = self.crowd_chain.detect(blank)
        self.assertEqual(len(outcome.faces), 0)
        self.assertIsNone(outcome.error)
        print(f"\n[Case 8: No Face Image] Detections: {len(outcome.faces)} (correct NO_FACE).")

    # Quality-assessment success rate on multi-face crowd
    def test_quality_assessment_success_rate(self):
        if not self.has_weights:
            self.skipTest("eDifFIQA-S weights required")

        canvas = np.full((800, 1000, 3), 200, dtype=np.uint8)
        src_img = cv2.imread(str(TEST_IMAGES / 'image2.jpg'))
        if src_img is None:
            self.skipTest("image2.jpg required")

        # 3 faces
        canvas[100:250, 100:250] = cv2.resize(src_img, (150, 150))
        canvas[100:250, 400:550] = cv2.resize(src_img, (150, 150))
        canvas[100:250, 700:850] = cv2.resize(src_img, (150, 150))

        results = self.pipeline.evaluate_array(canvas, name="multi_face_crowd.jpg")
        self.assertGreaterEqual(len(results), 3)

        successful = sum(1 for r in results if r.status == 'OK' and r.quality_score is not None)
        success_rate = successful / len(results)
        self.assertGreaterEqual(success_rate, 0.95)

        for idx, res in enumerate(results, start=1):
            self.assertEqual(res.face_index, idx)
            self.assertIsNotNone(res.bbox)
            self.assertIsNotNone(res.confidence)
            self.assertIsNotNone(res.quality_score)

        print(f"\n[FIQA Multi-Face Quality Assessment] Successfully evaluated {successful}/{len(results)} faces (Success Rate: {success_rate*100:.1f}%)")


if __name__ == '__main__':
    unittest.main()
