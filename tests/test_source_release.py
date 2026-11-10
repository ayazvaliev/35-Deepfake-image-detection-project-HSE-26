import hashlib
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd

from scripts.release_processed_sources import release_sources
from src.preprocessing.verification_views import (
    normalized_original_view, save_crop_verification_view, save_original_verification_view,
)


class SourceReleaseTests(unittest.TestCase):
    def prepare(self, root):
        raw = root / "raw/source"
        raw.mkdir(parents=True)
        output = root / "processed/source"
        output.mkdir(parents=True)
        image = np.random.default_rng(42).integers(0, 256, (128, 128, 3), dtype=np.uint8)
        success, encoded = cv2.imencode(".png", image)
        self.assertTrue(success)
        content = encoded.tobytes()
        source = raw / "image.png"
        source.write_bytes(content)
        original = save_original_verification_view(image, content, output / "views")
        crop = save_crop_verification_view(normalized_original_view(image), output / "crops", original)
        row = {"dataset": "source", "sample_id": "face", "input_path": str(source),
               "image_sha256": hashlib.sha256(content).hexdigest(), "decoded_sha256": "measured",
               "crop_sha256": "measured", "training_crop_removed": True, **original, **crop}
        metadata = output / "metadata.parquet"
        pd.DataFrame([row]).to_parquet(metadata, index=False)
        (output / "checkpoint.sqlite").write_bytes(b"completed checkpoint")
        return raw, source, metadata, row

    def test_release_checks_exact_views_and_preserves_cache_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw, source, metadata, row = self.prepare(Path(temporary))
            audit = release_sources(metadata, raw, delete_checkpoint=True)
            self.assertEqual(audit["removed_originals"], 1)
            self.assertTrue(audit["original_views_verified"])
            self.assertTrue(audit["crop_views_verified"])
            self.assertFalse(source.exists())
            self.assertTrue(metadata.exists())
            self.assertTrue(Path(row["original_verification_path"]).exists())
            self.assertFalse((metadata.parent / "checkpoint.sqlite").exists())

    def test_tampered_view_or_source_prevents_original_deletion(self):
        for tamper in ("view", "source"):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as temporary:
                raw, source, metadata, row = self.prepare(Path(temporary))
                changed = Path(row["crop_verification_path"]) if tamper == "view" else source
                changed.write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    release_sources(metadata, raw)
                self.assertTrue(source.exists())

    def test_a_valid_source_outside_owned_root_is_never_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw, source, metadata, row = self.prepare(Path(temporary))
            unrelated = Path(temporary) / "another-dataset"
            unrelated.mkdir()
            with self.assertRaisesRegex(ValueError, "outside"):
                release_sources(metadata, unrelated)
            self.assertTrue(source.exists())


if __name__ == "__main__":
    unittest.main()
