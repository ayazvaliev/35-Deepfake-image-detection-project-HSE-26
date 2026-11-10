import hashlib
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd

from scripts.compact_processed import compact
from src.preprocessing.verification_views import (
    read_verification_view, save_original_verification_view, view_pixel_sha256,
)


class ProcessedCompactionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = np.random.default_rng(43).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        self.raw = self.root / "raw.jpg"
        cv2.imwrite(str(self.raw), self.image)
        source = cv2.imread(str(self.raw))
        self.crop = self.root / "crops" / "crop.png"
        self.crop.parent.mkdir()
        cv2.imwrite(str(self.crop), self.image)
        self.metadata = self.root / "metadata.parquet"
        proxy = save_original_verification_view(source, self.raw.read_bytes(), self.root / "original-proxies")
        self.row = {
            "sample_id": "example", "target": 1, "crop_path": str(self.crop),
            "input_path": str(self.raw), "image_sha256": hashlib.sha256(self.raw.read_bytes()).hexdigest(),
            "crop_file_sha256": hashlib.sha256(self.crop.read_bytes()).hexdigest(),
            "crop_sha256": view_pixel_sha256(self.image), "training_crop_removed": False, **proxy,
        }
        pd.DataFrame([self.row]).to_parquet(self.metadata, index=False)

    def test_explicit_cleanup_occurs_only_after_verified_proxy_and_atomic_metadata(self):
        audit = compact(self.metadata, delete_crops=True)
        self.assertTrue(audit["original_views_verified"])
        self.assertTrue(audit["crop_views_verified"])
        self.assertEqual(audit["removed_png_crops"], 1)
        self.assertFalse(self.crop.exists())
        self.assertTrue(self.raw.exists())
        self.assertFalse(audit["raw_originals_removed"])
        row = pd.read_parquet(self.metadata).iloc[0]
        actual = read_verification_view(row.crop_verification_path, row.crop_verification_sha256,
                                        row.crop_verification_pixel_sha256, row.crop_verification_size,
                                        row.crop_verification_codec)
        np.testing.assert_array_equal(actual, self.image)
        self.assertEqual(audit["files"][0]["changed_storage_columns"], ["training_crop_removed"])
        again = compact(self.metadata, delete_crops=True)
        self.assertEqual(again["removed_png_crops"], 0)
        self.assertTrue(again["files"][0]["all_old_columns_equal"])

    def test_default_compaction_keeps_training_png_and_old_columns(self):
        audit = compact(self.metadata)
        self.assertTrue(self.crop.exists())
        self.assertEqual(audit["removed_png_crops"], 0)
        self.assertTrue(audit["files"][0]["all_old_columns_equal"])

    def test_changed_training_file_is_not_removed_or_accepted(self):
        before = self.metadata.read_bytes()
        self.crop.write_bytes(b"changed source")
        with self.assertRaises(ValueError):
            compact(self.metadata, delete_crops=True)
        self.assertTrue(self.crop.exists())
        self.assertEqual(self.metadata.read_bytes(), before)
        self.assertTrue(self.raw.exists())


if __name__ == "__main__":
    unittest.main()
