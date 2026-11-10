import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class CompactSplitReportTests(unittest.TestCase):
    def test_compact_audit_has_valid_split_counts_and_does_not_claim_training_pngs_exist(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [
                {"dataset": "source", "sample_id": str(index), "target": index % 2,
                 "face_mask_fraction_of_crop": 0.6, "training_crop_removed": True}
                for index in range(80)
            ]
            manifest = root / "metadata.parquet"
            pd.DataFrame(records).to_parquet(manifest, index=False)
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/build_composite_splits.py"),
                 "--manifest", str(manifest), "--output", str(root / "splits")],
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            audit = json.loads((root / "splits/audit.json").read_text())
            self.assertTrue(audit["ready_for_split_analysis"])
            self.assertFalse(audit["ready_for_training"])
            self.assertEqual(audit["training_crops_removed_for_audit"], 80)
            self.assertEqual(sum(audit[part + "_rows"] for part in ("train", "validate", "test")), 80)


if __name__ == "__main__":
    unittest.main()
