import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from src.preprocessing.group_cache import (
    _prepare_raw, _seal, _sha256, bootstrap_verified_groups,
    grouping_fingerprints, load_verified_groups, write_verified_groups,
)
from src.preprocessing.near_duplicates import PixelNearDuplicateVerifier
from src.preprocessing.splits import _connect


class VerifiedGroupCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        cv2.setNumThreads(1)
        first = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        second = first.copy()
        second[0, 0, 0] ^= 1
        third = np.random.default_rng(10).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        rows = []
        for index, image in enumerate([first, second, third]):
            path = self.root / f"{index}.png"
            cv2.imwrite(str(path), image)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append({
                "dataset": "toy", "sample_id": chr(97 + index), "target": 1,
                "eligible": True, "source_group": "", "identity_group": "",
                "bbox": [0.0, 0.0, 64.0, 64.0], "image_sha256": digest,
                "decoded_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "crop_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "crop_file_sha256": digest, "phash": "0000000000000000" if index < 2 else "ffffffffffffffff",
                "input_path": str(path), "crop_path": str(path),
                "face_mask_fraction_of_crop": 0.5, "mask_source": "landmark_oval",
                "flags": [], "exclusion_flags": [],
            })
        self.raw = pd.DataFrame(rows)
        self.data = _prepare_raw(self.raw)
        verifier = PixelNearDuplicateVerifier(self.data)
        _connect(self.data, 4, verifier=verifier)
        self.verification = verifier.report()
        verifier.close()
        self.assertEqual(self.verification["verified_crop_pairs"], 1)
        self.path = self.root / "cache"

    def write(self):
        return write_verified_groups(self.data, self.path, 4, self.verification)

    def provenance(self):
        return {
            "source_commit": "a" * 40, "executed_notebook_sha256": "b" * 64,
            "historical_logic_fingerprints": grouping_fingerprints(),
        }

    def test_real_pixel_proof_roundtrip_order_independent_and_offline(self):
        written = self.write()
        self.assertTrue(written["verified_now"])
        shuffled = self.data.iloc[[2, 0, 1]].copy()
        for path in self.root.glob("*.png"):
            path.unlink()
        with patch.object(PixelNearDuplicateVerifier, "_load_view", side_effect=AssertionError("Pixels must not be read")):
            groups, previous, audit = load_verified_groups(shuffled, self.path, 4)
        self.assertEqual(groups.tolist(), ["c", "a", "a"])
        self.assertEqual(previous, self.verification)
        self.assertTrue(audit["reused"])
        self.assertFalse(audit["verified_now"])

    def test_changed_group_inputs_are_rejected(self):
        self.write()
        mutations = {
            "target": 0, "eligible": False, "source_group": "changed",
            "identity_group": "identity", "dataset": "other",
            "bbox": [1, 0, 64, 64], "image_sha256": "changed",
            "decoded_sha256": "changed", "crop_sha256": "changed",
            "crop_file_sha256": "changed", "phash": "1234567890abcdef",
            "sample_id": "new-id",
        }
        for column, value in mutations.items():
            with self.subTest(column=column):
                changed = self.data.copy(deep=True)
                changed.at[0, column] = value
                with self.assertRaises(ValueError):
                    load_verified_groups(changed, self.path, 4)
        with self.assertRaises(ValueError):
            load_verified_groups(self.data.iloc[:-1], self.path, 4)

    def test_threshold_and_grouping_logic_changes_are_rejected(self):
        self.write()
        with self.assertRaisesRegex(ValueError, "threshold"):
            load_verified_groups(self.data, self.path, 3)
        with patch("src.preprocessing.group_cache.grouping_fingerprints", return_value={"changed": "code"}):
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_verified_groups(self.data, self.path, 4)

    def test_file_and_manifest_tampering_are_rejected(self):
        self.write()
        manifest_path = self.path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["groups"] += 1
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "manifest integrity"):
            load_verified_groups(self.data, self.path, 4)
        manifest["groups"] -= 1
        manifest_path.write_text(json.dumps(manifest))
        table = pd.read_parquet(self.path / "groups.parquet")
        table.loc[1, "group_id"] = "c"
        table.to_parquet(self.path / "groups.parquet", index=False)
        with self.assertRaisesRegex(ValueError, "file integrity"):
            load_verified_groups(self.data, self.path, 4)
        manifest["cache_file_sha256"] = _sha256(self.path / "groups.parquet")
        manifest["manifest_sha256"] = _seal(manifest)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "record digest"):
            load_verified_groups(self.data, self.path, 4)

    def test_bootstrap_uses_raw_quality_and_only_previous_group_ids(self):
        old = self.data.copy()
        old["flags"] = old["flags"].map(list)
        old.at[1, "flags"] = ["exact_face_duplicate"]
        old.at[1, "eligible"] = False
        old["partition"] = ["test", "excluded", "train"]
        old["author_split"] = ["Test", "Test", "Train"]
        audit = {"input_rows": 3, "near_duplicate_verification": self.verification}
        result = bootstrap_verified_groups(self.raw, old, audit, self.path, 4, self.provenance())
        groups, previous, loaded = load_verified_groups(_prepare_raw(self.raw), self.path, 4)
        self.assertEqual(groups.tolist(), ["a", "a", "c"])
        self.assertFalse(result["verified_now"])
        self.assertTrue(loaded["bootstrap_from_previous_verified_run"])
        self.assertEqual(previous, self.verification)

    def test_bootstrap_rejects_unknown_postprocessing_or_changed_old_inputs(self):
        audit = {"input_rows": 3, "near_duplicate_verification": self.verification}
        old = self.data.copy()
        old["flags"] = old["flags"].map(list)
        old.at[1, "flags"] = ["unexpected_filter"]
        old.at[1, "eligible"] = False
        with self.assertRaisesRegex(ValueError, "do not match"):
            bootstrap_verified_groups(self.raw, old, audit, self.path, 4, self.provenance())
        old = self.data.copy()
        old.at[1, "source_group"] = "changed"
        with self.assertRaisesRegex(ValueError, "do not match"):
            bootstrap_verified_groups(self.raw, old, audit, self.path, 4, self.provenance())

    def test_missing_pixel_proof_and_unreadable_run_are_rejected(self):
        for audit in [{"mode": "injected_verifier"}, {**self.verification, "unreadable_rows": 1}]:
            with self.assertRaises(ValueError):
                write_verified_groups(self.data, self.path, 4, audit)
        mismatched = {**self.verification, "settings": {**self.verification["settings"], "ssim_min": 0.5}}
        with self.assertRaisesRegex(ValueError, "settings"):
            write_verified_groups(self.data, self.path, 4, mismatched)

    def test_cache_cannot_be_overwritten(self):
        self.write()
        with self.assertRaises(FileExistsError):
            self.write()


if __name__ == "__main__":
    unittest.main()
