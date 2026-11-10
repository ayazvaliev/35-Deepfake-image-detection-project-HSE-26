import hashlib
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd

from scripts.compact_source_crops import compact_source_crops
from src.preprocessing.faces import adaptive_face_crop, square_face_crop
from src.preprocessing.source_crops import read_source_crop, save_source_crop
from src.preprocessing.verification_views import (
    normalized_original_view, save_crop_verification_view, save_original_verification_view, view_pixel_sha256,
)


class ExactSourceCropTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        image = np.random.default_rng(29).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.assertTrue(success)
        self.image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        self.original = save_original_verification_view(self.image, encoded.tobytes(), self.root / "originals")
        self.assertEqual(self.original["original_verification_codec"], "source_bytes")

    def test_downscale_upscale_and_padded_crops_reconstruct_identical_pixels(self):
        cases = [([30, 30, 150, 150], False), ([30, 30, 40, 40], False), ([0, 0, 100, 60], True)]
        for bbox, padded in cases:
            crop, bounds, _ = (square_face_crop if padded else adaptive_face_crop)(self.image, bbox)
            metadata = save_source_crop(crop, bounds, self.original, self.root / "descriptors")
            restored = read_source_crop(metadata["crop_verification_path"], metadata["crop_verification_sha256"],
                                        metadata["crop_verification_pixel_sha256"], metadata["crop_verification_size"])
            np.testing.assert_array_equal(restored, crop)
            self.assertEqual(view_pixel_sha256(restored), view_pixel_sha256(crop))
            self.assertLess(metadata["crop_verification_bytes"], 2048)

    def test_source_and_descriptor_integrity_are_checked_on_every_read(self):
        crop, bounds, _ = adaptive_face_crop(self.image, [30, 30, 150, 150])
        metadata = save_source_crop(crop, bounds, self.original, self.root / "descriptors")
        arguments = [metadata["crop_verification_path"], metadata["crop_verification_sha256"],
                     metadata["crop_verification_pixel_sha256"], metadata["crop_verification_size"]]
        read_source_crop(*arguments)
        source = Path(self.original["original_verification_path"])
        content = source.read_bytes()
        source.write_bytes(b"tampered original")
        with self.assertRaisesRegex(ValueError, "original file SHA256"):
            read_source_crop(*arguments)
        source.write_bytes(content)
        descriptor = Path(arguments[0])
        descriptor.write_bytes(descriptor.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "descriptor file SHA256"):
            read_source_crop(*arguments)

    def test_identical_crops_from_different_sources_get_immutable_distinct_descriptors(self):
        first = self.image.copy()
        second = self.image.copy()
        first[64:128, 64:128] = 120
        second[64:128, 64:128] = 120
        second[:32] = 17
        bounds = [64, 64, 64, 64]
        metadata = []
        for index, image in enumerate((first, second)):
            path = self.root / f"source-{index}.png"
            cv2.imwrite(str(path), image)
            original = {"original_verification_codec": "source_bytes",
                        "original_verification_path": str(path),
                        "original_verification_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            crop = np.full((256, 256, 3), 120, np.uint8)
            metadata.append(save_source_crop(crop, bounds, original, self.root / "descriptors"))
        self.assertEqual(metadata[0]["crop_verification_pixel_sha256"], metadata[1]["crop_verification_pixel_sha256"])
        self.assertNotEqual(metadata[0]["crop_verification_path"], metadata[1]["crop_verification_path"])
        for row in metadata:
            self.assertEqual(hashlib.sha256(Path(row["crop_verification_path"]).read_bytes()).hexdigest(),
                             row["crop_verification_sha256"])
            read_source_crop(row["crop_verification_path"], row["crop_verification_sha256"],
                             row["crop_verification_pixel_sha256"], row["crop_verification_size"])

    def test_wrong_bounds_cannot_claim_the_same_pixel_identity(self):
        crop, bounds, _ = adaptive_face_crop(self.image, [30, 30, 150, 150])
        wrong = (bounds[0] + 1, *bounds[1:])
        with self.assertRaisesRegex(ValueError, "changes preprocessing pixels"):
            save_source_crop(crop, wrong, self.original, self.root / "descriptors")

    def test_postconversion_preserves_all_pixels_and_releases_only_old_crop_caches(self):
        crop, bounds, _ = adaptive_face_crop(self.image, [30, 30, 150, 150])
        old_cache = save_crop_verification_view(crop, self.root / "crops", self.original)
        metadata_path = self.root / "metadata.parquet"
        row = {"sample_id": "example", "target": 1, "crop_sha256": view_pixel_sha256(crop),
               "crop_bounds": list(bounds), "training_crop_removed": True, **self.original, **old_cache}
        pd.DataFrame([row]).to_parquet(metadata_path, index=False)
        audit = compact_source_crops(metadata_path, delete_old_caches=True, workers=2)
        current = pd.read_parquet(metadata_path).iloc[0]
        self.assertEqual(current.crop_verification_codec, "source_crop")
        self.assertFalse(Path(old_cache["crop_verification_path"]).exists())
        self.assertTrue(Path(self.original["original_verification_path"]).exists())
        self.assertTrue(audit["all_crop_pixels_equal"])
        self.assertTrue(audit["files"][0]["all_nonstorage_columns_equal"])
        self.assertEqual(audit["converted_crop_records"], 1)
        self.assertEqual(current.crop_sha256, row["crop_sha256"])
        restored = read_source_crop(current.crop_verification_path, current.crop_verification_sha256,
                                    current.crop_verification_pixel_sha256, current.crop_verification_size)
        np.testing.assert_array_equal(restored, crop)


if __name__ == "__main__":
    unittest.main()
