import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.preprocessing.quality import apply_quality_policy


ROOT = Path(__file__).resolve().parents[1]


class QualityPolicyTests(unittest.TestCase):
    def test_boundary_missing_and_nonfinite_fractions(self):
        values = [0, 0.25, 0.250001, None, np.nan, np.inf, -0.1, 1.1, 1.0]
        frame = pd.DataFrame({
            "face_mask_fraction_of_crop": values,
            "eligible": [True] * len(values),
            "flags": [[] for _ in values],
        })
        result, audit = apply_quality_policy(frame)
        self.assertEqual(result["eligible"].tolist(), [False, False, True, False, False, False, False, False, True])
        self.assertEqual(result.iloc[0]["exclusion_flags"], ["face_mask_too_small"])
        self.assertEqual(result.iloc[1]["exclusion_flags"], ["face_mask_too_small"])
        self.assertEqual(result.iloc[2]["exclusion_flags"], [])
        for index in [3, 4, 5, 6, 7]:
            self.assertEqual(result.iloc[index]["exclusion_flags"], ["face_mask_missing"])
        self.assertEqual(audit["missing_fraction_records"], 5)
        self.assertEqual(audit["too_small_fraction_records"], 2)
        self.assertTrue(frame["eligible"].all())

    def test_missing_column_is_excluded_and_unknown_generator_is_kept(self):
        result, audit = apply_quality_policy(pd.DataFrame({"eligible": [True]}))
        self.assertFalse(result.iloc[0]["eligible"])
        self.assertEqual(result.iloc[0]["flags"], ["face_mask_missing"])
        self.assertEqual(audit["missing_fraction_records"], 1)
        frame = pd.DataFrame({
            "generator": ["unknown"], "model": [None],
            "face_mask_fraction_of_crop": [0.5], "eligible": [True],
        })
        result, _ = apply_quality_policy(frame)
        self.assertTrue(result.iloc[0]["eligible"])
        self.assertEqual(result.iloc[0]["generator"], "unknown")

    def test_existing_flags_exclusions_and_idempotence(self):
        frame = pd.DataFrame({
            "face_mask_fraction_of_crop": [0.2, 0.8, 0.8],
            "eligible": [True, False, True],
            "flags": [np.array(["detector_missing"]), ["face_too_small"], []],
            "exclusion_flags": [[], ["face_too_small"], ["annotation_conflict"]],
        })
        first, _ = apply_quality_policy(frame)
        second, _ = apply_quality_policy(first)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.iloc[0]["flags"], ["detector_missing", "face_mask_too_small"])
        self.assertEqual(first.iloc[1]["exclusion_flags"], ["face_too_small"])
        self.assertEqual(first.iloc[2]["exclusion_flags"], ["annotation_conflict"])
        self.assertEqual(first["eligible"].tolist(), [False, False, False])

    def test_invalid_threshold_is_rejected(self):
        for threshold in [-0.1, 1.1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                apply_quality_policy(pd.DataFrame(), min_mask_fraction=threshold)

    def test_quality_only_cli_preserves_inference_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            frame = pd.DataFrame({
                "dataset": ["test"] * 5,
                "sample_id": ["a", "b", "c", "d", "e"],
                "face_index": [0, 0, 0, None, 1],
                "crop_path": ["a.png", "b.png", "c.png", None, "e.png"],
                "face_mask_fraction_of_crop": [0.25, 0.250001, None, 0.0, 0.8],
                "eligible": [True, True, True, False, False],
                "flags": [[], [], [], ["no_face"], ["annotation_conflict"]],
                "exclusion_flags": [[], [], [], ["no_face"], ["annotation_conflict"]],
            })
            frame.to_parquet(output / "metadata.parquet", index=False)
            original_audit = {
                "images": 4, "faces": 5, "eligible_faces": 3,
                "fingerprint": "original-inference",
                "provenance": {"settings": {"expansion": 1.3}, "model_sha256": {"yunet": "abc"}},
                "computed_images": 4, "reused_images": 0,
                "manifest_sha256": "manifest-hash",
            }
            audit_path = output / "preprocessing_audit.json"
            audit_path.write_text(json.dumps(original_audit))
            checkpoint = output / "checkpoint.sqlite"
            checkpoint.write_bytes(b"unchanged inference checkpoint")
            command = [
                sys.executable, str(ROOT / "scripts" / "preprocess.py"),
                "--output-dir", str(output), "--quality-only",
            ]
            result = subprocess.run(command, cwd=output, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            updated = pd.read_parquet(output / "metadata.parquet")
            audit = json.loads(audit_path.read_text())
            self.assertEqual(updated["eligible"].tolist(), [False, True, False, False, False])
            for key in ["images", "fingerprint", "provenance", "computed_images", "reused_images", "manifest_sha256"]:
                self.assertEqual(audit[key], original_audit[key])
            self.assertEqual(audit["faces"], 5)
            self.assertEqual(audit["processing_records"], 5)
            self.assertEqual(audit["face_records"], 4)
            self.assertEqual(audit["crop_records"], 4)
            self.assertEqual(audit["eligible_faces"], 1)
            self.assertFalse(audit["quality_provenance"]["inference_rerun"])
            self.assertEqual(audit["quality_policy"]["threshold_comparison"], "<=")
            self.assertEqual(checkpoint.read_bytes(), b"unchanged inference checkpoint")
            result = subprocess.run(command, cwd=output, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            repeated = pd.read_parquet(output / "metadata.parquet")
            pd.testing.assert_frame_equal(updated, repeated)


if __name__ == "__main__":
    unittest.main()
