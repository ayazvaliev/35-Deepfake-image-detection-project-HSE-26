import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.preprocessing.verification_views import (
    normalized_original_view, read_verification_view,
    save_crop_verification_view, save_original_verification_view, view_pixel_sha256,
)


class ExactVerificationViewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_original_view_exactly_matches_previous_resize_and_source_selection(self):
        image = np.random.default_rng(7).integers(0, 256, (480, 640, 3), dtype=np.uint8)
        success, source = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.assertTrue(success)
        image = cv2.imdecode(source, cv2.IMREAD_COLOR)
        expected = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        metadata = save_original_verification_view(image, source.tobytes(), self.root / "originals")
        actual = read_verification_view(
            metadata["original_verification_path"], metadata["original_verification_sha256"],
            metadata["original_verification_pixel_sha256"], metadata["original_verification_size"],
            metadata["original_verification_codec"],
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(metadata["original_aspect_ratio"], 640 / 480)
        self.assertEqual(metadata["original_verification_pixel_sha256"], view_pixel_sha256(expected))
        self.assertLessEqual(metadata["original_verification_bytes"], len(source))

    def test_source_bytes_are_preserved_when_smaller_and_crop_can_alias_original(self):
        image = np.random.default_rng(19).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        success, source = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.assertTrue(success)
        image = cv2.imdecode(source, cv2.IMREAD_COLOR)
        original = save_original_verification_view(image, source.tobytes(), self.root / "originals")
        self.assertEqual(original["original_verification_codec"], "source_bytes")
        self.assertEqual(Path(original["original_verification_path"]).read_bytes(), source.tobytes())
        crop = save_crop_verification_view(image, self.root / "crops", original)
        self.assertTrue(crop["crop_verification_aliases_original"])
        self.assertEqual(crop["crop_verification_path"], original["original_verification_path"])

    def test_lossless_crop_cache_and_file_or_pixel_tampering_are_checked(self):
        image = np.random.default_rng(21).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        metadata = save_crop_verification_view(image, self.root / "crops")
        args = [metadata["crop_verification_path"], metadata["crop_verification_sha256"],
                metadata["crop_verification_pixel_sha256"], metadata["crop_verification_size"],
                metadata["crop_verification_codec"]]
        np.testing.assert_array_equal(read_verification_view(*args), image)
        with self.assertRaisesRegex(ValueError, "pixel SHA256"):
            read_verification_view(args[0], args[1], "0" * 64, args[3], args[4])
        Path(args[0]).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "file SHA256"):
            read_verification_view(*args)

    def test_small_jpeg_fast_path_avoids_lossless_encoder_and_crop_uses_exact_source_descriptor(self):
        image = np.random.default_rng(31).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        success, source = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.assertTrue(success)
        self.assertLess(len(source), 64 * 1024)
        image = cv2.imdecode(source, cv2.IMREAD_COLOR)
        crop = cv2.resize(image[30:170, 40:180], (256, 256), interpolation=cv2.INTER_CUBIC)
        with patch("src.preprocessing.verification_views._lossless_encoding", side_effect=AssertionError("Unnecessary encoding")):
            original = save_original_verification_view(image, source.tobytes(), self.root / "fast-original")
            cached = save_crop_verification_view(crop, self.root / "fast-crop", original,
                                                 crop_bounds=[40, 30, 140, 140], source_pixels=image)
        self.assertEqual(original["original_verification_storage_policy"], "prefer_small_source_bytes_otherwise_min")
        self.assertEqual(cached["crop_verification_codec"], "source_crop")
        restored = read_verification_view(cached["crop_verification_path"], cached["crop_verification_sha256"],
                                          cached["crop_verification_pixel_sha256"], cached["crop_verification_size"],
                                          cached["crop_verification_codec"])
        np.testing.assert_array_equal(restored, crop)

    def test_small_original_resize_uses_the_same_cubic_interpolation(self):
        image = np.random.default_rng(6).integers(0, 256, (71, 111, 3), dtype=np.uint8)
        expected = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
        np.testing.assert_array_equal(normalized_original_view(image), expected)
        self.assertEqual(view_pixel_sha256(expected), hashlib.sha256(
            f"{expected.dtype}:{expected.shape}:".encode() + expected.tobytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
