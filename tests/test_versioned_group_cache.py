import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from src.preprocessing.group_cache import _prepare_raw, versioned_cache_path
from src.preprocessing.near_duplicates import PixelNearDuplicateVerifier
from src.preprocessing.splits import build_composite_splits
from src.preprocessing.verification_views import view_pixel_sha256


class VersionedGroupCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "groups"
        cv2.setNumThreads(1)
        first = np.random.default_rng(72).integers(20, 230, (64, 64, 3), dtype=np.uint8)
        second = first + 1
        third = np.random.default_rng(8).integers(20, 230, (64, 64, 3), dtype=np.uint8)
        rows = []
        for index, image in enumerate((first, second, third)):
            source = self.root / f"source-{index}.jpg"
            cv2.imwrite(str(source), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            pixels = cv2.imread(str(source))
            crop = self.root / f"crop-{index}.png"
            cv2.imwrite(str(crop), pixels)
            rows.append({
                "dataset": "toy", "sample_id": f"row-{index}", "target": 1,
                "source_group": f"image-{index}", "face_mask_fraction_of_crop": 0.6,
                "input_path": str(source), "crop_path": str(crop), "path": str(crop),
                "image_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "decoded_sha256": view_pixel_sha256(pixels), "crop_sha256": view_pixel_sha256(pixels),
                "crop_file_sha256": hashlib.sha256(crop.read_bytes()).hexdigest(),
                "phash": "0000000000000000" if index < 2 else "ffffffffffffffff",
                "source_format": "JPEG", "source_jpeg_quality": 95,
                "source_jpeg_quality_method": "exact_ijg_qtables", "source_jpeg_quality_status": "exact",
                "source_jpeg_subsampling": "4:2:0", "author_split": "train",
            })
        self.raw = pd.DataFrame(rows)

    def build(self, frame=None, **kwargs):
        return build_composite_splits(self.raw if frame is None else frame,
                                      verified_group_cache=self.cache, group_cache_policy="versioned", **kwargs)

    def test_quality_only_author_and_row_order_changes_reuse_the_same_real_pixel_proof(self):
        first = self.build()
        self.assertEqual(first["report"]["near_duplicate_verification"]["verified_crop_pairs"], 1)
        modified = self.raw.iloc[[2, 0, 1]].copy()
        modified["source_jpeg_quality"] = 100
        modified["author_split"] = "test"
        for filename in self.root.glob("source-*.jpg"):
            filename.unlink()
        for filename in self.root.glob("crop-*.png"):
            filename.unlink()
        with patch.object(PixelNearDuplicateVerifier, "__call__", side_effect=AssertionError("Unexpected pixel read")):
            second = self.build(modified)
        old_cache = first["report"]["group_cache"]
        new_cache = second["report"]["group_cache"]
        self.assertEqual(old_cache["cache_key"], new_cache["cache_key"])
        self.assertEqual(old_cache["cache_path"], new_cache["cache_path"])
        self.assertTrue(new_cache["reused"])
        self.assertEqual(new_cache["policy"], "versioned")
        self.assertEqual(len(list(self.cache.glob("v1_*"))), 1)
        self.assertFalse(second["report"]["near_duplicate_verification"]["verified_now"])

    def test_changed_target_creates_fresh_child_and_keeps_the_previous_cache(self):
        first = self.build()
        old_path = Path(first["report"]["group_cache"]["cache_path"])
        old_manifest = (old_path / "manifest.json").read_bytes()
        modified = self.raw.copy()
        modified.at[0, "target"] = 0
        second = self.build(modified)
        new_path = Path(second["report"]["group_cache"]["cache_path"])
        self.assertNotEqual(old_path, new_path)
        self.assertEqual((old_path / "manifest.json").read_bytes(), old_manifest)
        self.assertTrue((new_path / "manifest.json").is_file())
        self.assertFalse(second["report"]["group_cache"]["reused"])
        self.assertTrue(second["report"]["near_duplicate_verification"]["verified_now"])
        self.assertEqual(len(list(self.cache.glob("v1_*"))), 2)

    def test_same_key_tampered_cache_fails_closed_without_creating_another_child(self):
        first = self.build()
        path = Path(first["report"]["group_cache"]["cache_path"])
        with (path / "groups.parquet").open("ab") as stream:
            stream.write(b"tampered")
        with patch.object(PixelNearDuplicateVerifier, "__call__", side_effect=AssertionError("Silent fallback is forbidden")):
            with self.assertRaisesRegex(ValueError, "file integrity"):
                self.build()
        self.assertEqual(len(list(self.cache.glob("v1_*"))), 1)

    def test_key_tracks_storage_logic_threshold_and_metric_but_ignores_runtime_tuning(self):
        prepared = _prepare_raw(self.raw)
        path, _ = versioned_cache_path(prepared, self.cache, 4)
        verifier = PixelNearDuplicateVerifier(pd.DataFrame())
        settings = dict(verifier.settings)
        verifier.close()
        runtime = {**settings, "cache_size": 999, "opencv_threads": 99}
        self.assertEqual(versioned_cache_path(prepared, self.cache, 4, runtime)[0], path)
        metric = {**settings, "ssim_min": 0.99}
        self.assertNotEqual(versioned_cache_path(prepared, self.cache, 4, metric)[0], path)
        self.assertNotEqual(versioned_cache_path(prepared, self.cache, 3)[0], path)
        changed = prepared.copy()
        changed["crop_verification_sha256"] = "changed-storage-binding"
        self.assertNotEqual(versioned_cache_path(changed, self.cache, 4)[0], path)
        with patch("src.preprocessing.group_cache.grouping_fingerprints", return_value={"logic": "changed"}):
            self.assertNotEqual(versioned_cache_path(prepared, self.cache, 4)[0], path)

    def test_cli_passes_versioned_policy_and_reuses_after_author_metadata_change(self):
        repository = Path(__file__).resolve().parents[1]
        source = self.root / "faces.parquet"
        output = self.root / "output"
        rows = pd.DataFrame([
            {"dataset": "source", "sample_id": f"face-{index}", "target": index % 2,
             "source_group": f"source-{index}", "author_split": "train",
             "face_mask_fraction_of_crop": 0.6, "training_crop_removed": False if index % 2 else None}
            for index in range(120)
        ])
        rows.to_parquet(source, index=False)
        config = self.root / "config.yml"
        config.write_text("splits:\n  verification_cache_size: 7\n")
        command = [sys.executable, str(repository / "scripts/build_composite_splits.py"),
                   "--manifest", str(source), "--output", str(output),
                   "--group-cache", str(self.cache), "--group-cache-policy", "versioned", "--config", str(config)]
        first = subprocess.run(command, cwd=repository, capture_output=True, text=True, check=False)
        self.assertEqual(first.returncode, 0, first.stderr)
        audit = json.loads((output / "audit.json").read_text())
        self.assertEqual(audit["group_cache"]["policy"], "versioned")
        self.assertFalse(audit["group_cache"]["reused"])
        self.assertEqual(audit["near_duplicate_verification"]["settings"]["cache_size"], 7)
        rows["author_split"] = "test"
        rows.to_parquet(source, index=False)
        second = subprocess.run(command + ["--verification-cache-size", "4096"], cwd=repository,
                                capture_output=True, text=True, check=False)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertNotIn("FutureWarning", second.stderr)
        reused = json.loads((output / "audit.json").read_text())
        self.assertTrue(reused["group_cache"]["reused"])
        self.assertEqual(reused["requested_verification_cache_size"], 4096)
        self.assertEqual(reused["group_cache"]["cache_key"], audit["group_cache"]["cache_key"])

    def test_operational_cache_size_is_forwarded_without_changing_grouping_results(self):
        small = build_composite_splits(self.raw, verification_cache_size=2)
        large = build_composite_splits(self.raw, verification_cache_size=16384)
        self.assertEqual(small["report"]["near_duplicate_verification"]["settings"]["cache_size"], 2)
        self.assertEqual(large["report"]["near_duplicate_verification"]["settings"]["cache_size"], 16384)
        for partition in ("train", "validate", "test", "excluded"):
            pd.testing.assert_frame_equal(small[partition], large[partition])

    def test_versioned_cache_reuses_historical_proof_after_runtime_cache_size_change(self):
        first = self.build(verification_cache_size=2)
        with patch.object(PixelNearDuplicateVerifier, "__call__", side_effect=AssertionError("Unnecessary verification")):
            second = self.build(verification_cache_size=16384)
        self.assertEqual(first["report"]["group_cache"]["cache_key"], second["report"]["group_cache"]["cache_key"])
        self.assertTrue(second["report"]["group_cache"]["reused"])
        self.assertEqual(second["report"]["requested_verification_cache_size"], 16384)
        self.assertEqual(second["report"]["near_duplicate_verification"]["settings"]["cache_size"], 2)

    def test_operational_cache_size_must_be_a_positive_integer(self):
        for value in (0, -1, 1.5, True, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    build_composite_splits(self.raw, verification_cache_size=value)

    def test_incomplete_same_key_cache_is_not_silently_replaced(self):
        prepared = _prepare_raw(self.raw)
        path, _ = versioned_cache_path(prepared, self.cache, 4)
        path.mkdir(parents=True)
        (path / "unfinished.tmp").write_text("partial cache")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.build()


if __name__ == "__main__":
    unittest.main()
