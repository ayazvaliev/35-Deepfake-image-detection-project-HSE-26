import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from src.preprocessing.faces import (
    YuNetFacePreprocessor, angle_bin, array_sha256, matrix_angles,
    square_bounds, square_face_crop, adaptive_square_bounds, adaptive_face_crop,
    crop_padding_fraction,
)


class Detector:
    def __init__(self, boxes):
        self.boxes = boxes

    def setInputSize(self, size):
        self.size = size

    def detect(self, image):
        if not self.boxes:
            return None, None
        rows = [list(box) + [0.0] * 10 + [0.99] for box in self.boxes]
        return None, np.asarray(rows, dtype=np.float32)


class Landmarker:
    def detect(self, image):
        points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
        from src.preprocessing.faces import FACE_OVAL

        for index, point_index in enumerate(FACE_OVAL):
            angle = 2 * np.pi * index / len(FACE_OVAL)
            points[point_index] = SimpleNamespace(
                x=0.5 + 0.25 * np.cos(angle), y=0.5 + 0.3 * np.sin(angle)
            )
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_euler("xyz", [10, 30, -5], degrees=True).as_matrix()
        return SimpleNamespace(
            face_landmarks=[points], facial_transformation_matrixes=[matrix]
        )


class ScaleDetector:
    def __init__(self, by_shape):
        self.by_shape = by_shape
        self.calls = []

    def setInputSize(self, size):
        self.size = size

    def detect(self, image):
        self.calls.append(self.size)
        rows = self.by_shape.get(self.size)
        return None, None if not rows else np.asarray(rows, dtype=np.float32)


def detection_row(bbox, score, points=None):
    if points is None:
        x, y, width, height = bbox
        points = [x + width / 2, y + height / 2] * 5
    return [*bbox, *points, score]


class FacePreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "frame.png"
        self.image = np.full((128, 128, 3), 80, np.uint8)
        cv2.imwrite(str(self.path), self.image)

    def tearDown(self):
        self.directory.cleanup()

    def processor(self, boxes, landmarks=True):
        return YuNetFacePreprocessor(
            detector=Detector(boxes), landmarker=Landmarker() if landmarks else None
        )

    def test_crop_padding_and_square_center(self):
        image = np.full((40, 60, 3), 230, np.uint8)
        crop, bounds, padding = square_face_crop(image, [0, 0, 40, 20], size=52)
        self.assertEqual(bounds, (-6, -16, 52, 52))
        np.testing.assert_array_equal(crop[:16], 127)
        np.testing.assert_array_equal(crop[16:40, 6:45], 230)
        self.assertGreater(padding, 0)
        self.assertEqual(square_bounds([10, 20, 33, 40]), (1, 14, 52, 52))

    def test_resize_interpolation_matches_contract(self):
        image = np.arange(200 * 200 * 3, dtype=np.uint8).reshape(200, 200, 3)
        crop, _, _ = square_face_crop(image, [20, 20, 100, 100], size=64, expansion=1)
        expected = cv2.resize(image[20:120, 20:120], (64, 64), interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(crop, expected)
        crop, _, _ = square_face_crop(image, [20, 20, 40, 40], size=64, expansion=1)
        expected = cv2.resize(image[20:60, 20:60], (64, 64), interpolation=cv2.INTER_CUBIC)
        np.testing.assert_array_equal(crop, expected)

    def test_rotation_angles_and_bin_boundaries(self):
        matrix = np.eye(4)
        matrix[:3, :3] = 1.7 * Rotation.from_euler("xyz", [10, 30, -5], degrees=True).as_matrix()
        np.testing.assert_allclose(matrix_angles(matrix), [30, 10, -5], atol=1e-6)
        self.assertEqual([angle_bin(x) for x in [14.9, 15, -45, 45.1, None]],
                         ["frontal", "three_quarter", "three_quarter", "profile", "unknown"])

    def test_multiple_faces_require_explicit_label_scope(self):
        processor = self.processor([[15, 15, 40, 40], [70, 70, 40, 40]])
        rows = processor.process(self.path, target=1)
        self.assertTrue(all(not row["eligible"] for row in rows))
        self.assertIn("ambiguous_multi_face_target", rows[0]["exclusion_flags"])
        rows = processor.process(self.path, target=1, image_label_all_faces=True)
        self.assertTrue(all(row["eligible"] for row in rows))

    def test_opposite_neighbor_and_same_face_conflict(self):
        processor = self.processor([])
        annotations = [
            {"bbox": [20, 20, 40, 40], "target": 0},
            {"bbox": [20, 20, 40, 40], "target": 1},
        ]
        rows = processor.process(self.path, annotations=annotations)
        self.assertIn("annotation_conflict", rows[0]["exclusion_flags"])
        self.assertIn("opposite_target_in_crop", rows[0]["exclusion_flags"])
        annotations[1]["bbox"] = [62, 25, 40, 40]
        row = processor.process(self.path, annotations=annotations)[0]
        self.assertNotIn("annotation_conflict", row["exclusion_flags"])
        self.assertIn("opposite_target_in_crop", row["exclusion_flags"])

    def test_small_faces_and_trusted_annotations(self):
        processor = self.processor([])
        row = processor.process(self.path, annotations=[{"bbox": [20, 20, 31, 60], "target": 0}])[0]
        self.assertIn("face_too_small", row["exclusion_flags"])
        row = processor.process(self.path, annotations=[{"bbox": [20, 20, 40, 40], "target": 0}])[0]
        self.assertIn("detector_missing", row["flags"])
        self.assertTrue(row["eligible"])

    def test_missing_landmarks_do_not_use_box_for_light(self):
        row = self.processor([[20, 20, 40, 40]], landmarks=False).process(self.path, target=0)[0]
        self.assertIsNone(row["lightness_lstar"])
        self.assertEqual(row["light_bin"], "unknown")
        self.assertEqual(row["yaw_bin"], "unknown")
        annotations = [{"bbox": [20, 20, 40, 40], "target": 0,
                        "segmentation": [[20, 20, 60, 20, 60, 60, 20, 60]]}]
        row = self.processor([], landmarks=False).process(self.path, annotations=annotations)[0]
        self.assertEqual(row["mask_source"], "annotation")
        self.assertIsNotNone(row["lightness_lstar"])

    def test_author_mask_has_priority_for_light_and_occupancy_but_pose_uses_landmarker(self):
        self.image[20:31, 20:31] = 230
        cv2.imwrite(str(self.path), self.image)
        processor = self.processor([[20, 20, 40, 40]])
        plain = processor.process(self.path, target=0)[0]
        annotations = [{"bbox": [20, 20, 40, 40], "target": 0,
                        "segmentation": [[20, 20, 30, 20, 30, 30, 20, 30]]}]
        annotated = processor.process(self.path, annotations=annotations)[0]
        self.assertEqual(annotated["mask_source"], "annotation")
        self.assertGreater(annotated["lightness_lstar"], plain["lightness_lstar"] + 10)
        self.assertAlmostEqual(annotated["face_mask_fraction_of_crop"], 121 / (52 * 52))
        self.assertAlmostEqual(annotated["yaw"], plain["yaw"])
        self.assertEqual(plain["mask_source"], "mediapipe_face_oval")

    def test_empty_and_invalid_author_masks_fall_back_to_landmarker(self):
        processor = self.processor([[20, 20, 40, 40]])
        for segmentation, flag in (([], "annotation_mask_empty"),
                                   (np.zeros((2, 2), dtype=np.uint8), "annotation_mask_invalid")):
            annotations = [{"bbox": [20, 20, 40, 40], "target": 0, "segmentation": segmentation}]
            row = processor.process(self.path, annotations=annotations)[0]
            self.assertEqual(row["mask_source"], "mediapipe_face_oval")
            self.assertIn(flag, row["flags"])
            self.assertGreater(row["face_mask_fraction_of_crop"], 0)
            self.assertLessEqual(row["face_mask_fraction_of_crop"], 1)

    def test_adaptive_square_inside_image_uses_maximum_allowed_side(self):
        bounds = adaptive_square_bounds([40, 40, 40, 40], self.image.shape)
        self.assertEqual(bounds, (34, 34, 52, 52))
        for bbox in ([0, 0, 80, 40], [90, 40, 30, 50], [10.3, 2.1, 45.7, 68.2]):
            bounds = adaptive_square_bounds(bbox, self.image.shape)
            left, top, side, _ = bounds
            x, y, width, height = bbox
            center_x, center_y = x + width / 2, y + height / 2
            capacity = 2 * min(center_x, center_y, 128 - center_x, 128 - center_y)
            self.assertEqual(side, int(np.floor(min(1.3 * max(width, height), capacity))))
            self.assertGreaterEqual(left, 0)
            self.assertGreaterEqual(top, 0)
            self.assertLessEqual(left + side, 128)
            self.assertLessEqual(top + side, 128)
            self.assertLessEqual(abs(left + side / 2 - center_x), 0.5)
            self.assertLessEqual(abs(top + side / 2 - center_y), 0.5)

    def test_adaptive_border_crop_has_no_padding_and_reports_truncation(self):
        row = self.processor([[0, 0, 80, 40]]).process(self.path, target=1)[0]
        self.assertEqual(row["crop_bounds"], [20, 0, 40, 40])
        self.assertEqual(row["padding_fraction"], 0)
        self.assertGreater(row["proposed_padding_fraction"], 0)
        self.assertEqual(row["proposed_crop_bounds"], [-12, -32, 104, 104])
        self.assertAlmostEqual(row["expansion_applied"], 0.5)
        self.assertAlmostEqual(row["face_coverage_fraction"], 0.5)
        self.assertEqual(row["visible_face_min_side"], 40)
        self.assertTrue(row["crop_truncates_bbox"])
        self.assertIn("crop_truncates_bbox", row["flags"])
        self.assertNotIn("crop_truncates_bbox", row["exclusion_flags"])
        self.assertTrue(row["eligible"])
        self.assertLessEqual(row["face_mask_fraction_of_crop"], 1)
        crop, _, padding = adaptive_face_crop(self.image, [0, 0, 80, 40])
        self.assertEqual(padding, 0)
        np.testing.assert_array_equal(crop, 80)

    def test_adaptive_too_little_visible_face_is_excluded(self):
        row = self.processor([[0, 0, 80, 32]]).process(self.path, target=0)[0]
        self.assertEqual(row["visible_face_min_side"], 32)
        self.assertNotIn("visible_face_too_small", row["exclusion_flags"])
        row = self.processor([[-1, -1, 80, 32]]).process(self.path, target=0)[0]
        self.assertEqual(row["visible_face_min_side"], 30)
        self.assertIn("visible_face_too_small", row["exclusion_flags"])
        self.assertNotIn("face_too_small", row["exclusion_flags"])

    def test_invalid_center_or_subpixel_capacity_produces_no_crop(self):
        for bbox in ([-50, 0, 40, 40], [-19.8, 0, 40, 40]):
            with self.assertRaises(ValueError):
                adaptive_square_bounds(bbox, self.image.shape)
            row = self.processor([]).process(self.path, annotations=[{"bbox": bbox, "target": 0}])[0]
            self.assertIn("invalid_bbox", row["exclusion_flags"])
            self.assertIsNone(row["crop_path"])

    def test_padded_baseline_remains_explicit_and_has_correct_area_fraction(self):
        processor = YuNetFacePreprocessor(
            detector=Detector([[0, 0, 80, 40]]), landmarker=Landmarker(), crop_policy="padded",
        )
        row = processor.process(self.path, target=0)[0]
        self.assertEqual(row["crop_bounds"], [-12, -32, 104, 104])
        self.assertGreater(row["padding_fraction"], 0)
        self.assertAlmostEqual(row["padding_fraction"], row["proposed_padding_fraction"])
        expected = 1 - (92 * 72) / (104 * 104)
        self.assertAlmostEqual(crop_padding_fraction(self.image.shape, row["crop_bounds"]), expected)

    def test_multiscale_detection_keeps_native_small_face_and_remaps_five_landmarks(self):
        image = np.full((501, 1001, 3), 80, dtype=np.uint8)
        scale_x, scale_y = 1001 / 320, 501 / 160
        large_bbox = [400, 100, 300, 250]
        large_points = [430, 150, 650, 150, 550, 210, 470, 300, 620, 300]
        scaled_bbox = [large_bbox[0] / scale_x, large_bbox[1] / scale_y,
                       large_bbox[2] / scale_x, large_bbox[3] / scale_y]
        scaled_points = [value / (scale_x if index % 2 == 0 else scale_y)
                         for index, value in enumerate(large_points)]
        small = detection_row([10, 10, 32, 32], 0.99)
        duplicate_small = detection_row([10 / scale_x, 10 / scale_y, 32 / scale_x, 32 / scale_y], 0.95)
        detector = ScaleDetector({
            (1001, 501): [small],
            (320, 160): [detection_row(scaled_bbox, 0.98, scaled_points), duplicate_small],
        })
        processor = YuNetFacePreprocessor(detector=detector, landmarker=Landmarker())
        rows = processor._detect(image)
        self.assertEqual(detector.calls, [(1001, 501), (320, 160)])
        self.assertEqual(len(rows), 2)
        native = min(rows, key=lambda row: row[2])
        large = max(rows, key=lambda row: row[2])
        np.testing.assert_allclose(native[:4], [10, 10, 32, 32], atol=1e-4)
        np.testing.assert_allclose(large[:4], large_bbox, atol=1e-4)
        np.testing.assert_allclose(large[4:14], large_points, atol=1e-4)
        self.assertEqual(processor._detection_audit["detector_native_faces"], 1)
        self.assertEqual(processor._detection_audit["detector_downscaled_faces"], 2)

    def test_small_image_uses_only_native_and_score_threshold_is_retained(self):
        detector = ScaleDetector({(128, 128): [
            detection_row([10, 10, 32, 32], 0.99),
            detection_row([70, 70, 40, 40], 0.5),
        ]})
        processor = YuNetFacePreprocessor(detector=detector, landmarker=Landmarker())
        rows = processor._detect(self.image)
        self.assertEqual(detector.calls, [(128, 128)])
        self.assertEqual(len(rows), 1)
        self.assertFalse(processor._detection_audit["detector_downscaled_applied"])

    def test_downscaled_detection_crops_original_pixels_and_records_coverage(self):
        image = np.arange(501 * 1001 * 3, dtype=np.uint8).reshape(501, 1001, 3)
        path = Path(self.directory.name) / "large.png"
        cv2.imwrite(str(path), image)
        detector = ScaleDetector({(320, 160): [detection_row([110, 30, 90, 80], 0.99)]})
        processor = YuNetFacePreprocessor(detector=detector, landmarker=Landmarker())
        row = processor.process(path, target=1, output_dir=Path(self.directory.name) / "large-crops")[0]
        expected, bounds, _ = adaptive_face_crop(image, row["bbox"])
        saved = cv2.imread(row["crop_path"])
        np.testing.assert_array_equal(saved, expected)
        self.assertEqual(row["crop_bounds"], list(bounds))
        self.assertEqual(row["detector_native_faces"], 0)
        self.assertEqual(row["detector_downscaled_faces"], 1)
        self.assertEqual(row["detected_faces"], 1)
        self.assertEqual(row["detector_policy"], "native_plus_downscaled_320")
        self.assertEqual(row["image_width"], 1001)
        self.assertEqual(row["image_height"], 501)

    def test_png_output_metadata_and_hashes(self):
        processor = self.processor([[20, 20, 40, 40]])
        row = processor.process_record(
            {"path": str(self.path), "target": 1, "dataset": "example"},
            Path(self.directory.name) / "crops",
        )[0]
        crop = cv2.imread(row["crop_path"])
        self.assertEqual(crop.shape, (256, 256, 3))
        self.assertEqual(row["crop_sha256"], array_sha256(crop))
        self.assertEqual(row["input_path"], str(self.path))
        self.assertEqual(row["dataset"], "example")
        self.assertEqual(row["source_format"], "PNG")
        self.assertTrue(row["source_format_verified"])
        self.assertEqual(row["source_width"], 128)
        self.assertEqual(row["yaw_bin"], "three_quarter")
        self.assertEqual(len(row["phash"]), 16)
        self.assertTrue(row["crop_path"].endswith(".png"))

    def test_empty_and_broken_inputs_are_auditable(self):
        row = self.processor([]).process(self.path, target=0)[0]
        self.assertEqual(row["exclusion_flags"], ["no_face"])
        broken = Path(self.directory.name) / "broken.png"
        broken.write_bytes(b"invalid image")
        row = self.processor([]).process(broken, target=0)[0]
        self.assertEqual(row["exclusion_flags"], ["decoding_failed"])


if __name__ == "__main__":
    unittest.main()
