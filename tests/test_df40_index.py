import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.preprocessing.datasets import index_df40

ROOT = Path(__file__).resolve().parents[1]


def acquired_sample(size=100):
    rows = []
    for index in range(size):
        method = "StyleGAN2" if index % 2 else "e4e"
        family = "EFS" if method == "StyleGAN2" else "FE"
        target = int(index % 5 != 0)
        rows.append({
            "dataset": "DF40" if index % 2 else "df40", "sample_id": f"native-image-{index:04}",
            "path": f"/offline-native/{index:04}.PNG" if index % 2 else f"/offline-native/{index:04}.jpg",
            "target": target, "native_method": method, "native_family": family,
            "generator": method if target else "real", "manipulation": family if target else "real",
            "domain": "ff" if target else "ffhq", "author_split": "Train" if index % 3 else "TEST",
            "source_format": "PNG" if index % 2 else "JPEG",
            "source_group": f"publisher-source-{index}", "identity_group": "",
            "author_split_inventory": np.array(["Train", "TEST"]),
            "publisher_field": f"Raw Value {index}",
        })
    return pd.DataFrame(rows)


class Df40IndexTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.raw = self.directory / "raw/df40"
        self.raw.mkdir(parents=True)
        acquired_sample().to_parquet(self.raw / "sample.parquet", index=False)
        self.source = pd.read_parquet(self.raw / "sample.parquet")

    def tearDown(self):
        self.temporary.cleanup()

    def run_cli(self, output, limit=20):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts/index_datasets.py"), "--dataset", "df40",
             "--root", str(self.raw), "--limit", str(limit), "--seed", "42", "--require-count",
             "--output", str(output)],
            capture_output=True, text=True, timeout=120,
        )

    def test_adapter_validates_but_does_not_normalize_acquired_values(self):
        result = index_df40(self.raw)
        pd.testing.assert_frame_equal(result, self.source)
        self.assertEqual(set(result["dataset"]), {"DF40", "df40"})
        self.assertEqual(set(result["author_split"]), {"Train", "TEST"})
        self.assertEqual(len(result), 100)

    def test_cli_caps_frozen_full_acquisition_and_preserves_parquet_array_metadata(self):
        output = self.directory / "processed/demo/manifests/df40.parquet"
        process = self.run_cli(output)
        self.assertEqual(process.returncode, 0, process.stderr)
        selected = pd.read_parquet(output)
        self.assertEqual(len(selected), 20)
        self.assertTrue(selected["sample_id"].is_unique)
        self.assertTrue(selected["target"].isin([0, 1]).all())
        self.assertTrue(set(selected["sample_id"]) <= set(self.source["sample_id"]))
        expected = self.source.set_index("sample_id").loc[selected["sample_id"]].reset_index()
        pd.testing.assert_frame_equal(selected.reset_index(drop=True), expected[selected.columns])
        self.assertTrue(all(isinstance(value, np.ndarray) for value in selected["author_split_inventory"]))
        self.assertTrue(all(value.tolist() == ["Train", "TEST"] for value in selected["author_split_inventory"]))
        sampling = json.loads(output.with_suffix(".sampling.json").read_text())
        self.assertEqual((sampling["population_n"], sampling["sample_n"]), (100, 20))
        self.assertEqual(sampling["strata"], ["native_method", "target", "generator", "manipulation",
                                              "domain", "author_split", "source_format"])
        self.assertFalse(output.with_name(output.name + ".tmp").exists())

    def test_require_count_fails_instead_of_silently_using_a_smaller_pool(self):
        output = self.directory / "insufficient.parquet"
        process = self.run_cli(output, limit=101)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("expected 101 images", process.stderr)
        self.assertIn("found 100", process.stderr)
        self.assertFalse(output.exists())

    def test_invalid_targets_video_rows_non_native_methods_and_foreign_dataset_fail(self):
        mutations = [
            ("dataset", "mffi", "another dataset"), ("target", 2, "binary targets"),
            ("path", "/movie.mp4", "native image paths"), ("native_method", "FaceSwap", "outside native"),
            ("native_family", "FS", "native_family"), ("manipulation", "FR", "manipulation"),
            ("sample_id", self.source.iloc[0]["sample_id"], "unique"),
        ]
        for column, value, message in mutations:
            with self.subTest(column=column):
                changed = self.source.copy()
                changed.at[1, column] = value
                changed.to_parquet(self.raw / "sample.parquet", index=False)
                with self.assertRaisesRegex(ValueError, message):
                    index_df40(self.raw)
        changed = self.source.assign(face_index=np.nan)
        changed.at[0, "face_index"] = 0
        changed.to_parquet(self.raw / "sample.parquet", index=False)
        with self.assertRaisesRegex(ValueError, "processed face rows"):
            index_df40(self.raw)

    def test_only_available_native_strata_are_used(self):
        minimal = self.source[["dataset", "sample_id", "path", "target", "native_method"]]
        minimal.to_parquet(self.raw / "sample.parquet", index=False)
        output = self.directory / "minimal.parquet"
        process = self.run_cli(output)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(len(pd.read_parquet(output)), 20)
        sampling = json.loads(output.with_suffix(".sampling.json").read_text())
        self.assertEqual(sampling["strata"], ["native_method", "target"])


if __name__ == "__main__":
    unittest.main()
