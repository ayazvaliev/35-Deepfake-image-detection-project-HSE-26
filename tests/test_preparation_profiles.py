from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
import yaml

from scripts.prepare_composite import ROOT, build_plan
from scripts.sample_processed import sample_processed


class PreparationProfileTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load((ROOT / "configs/composite.yml").read_text())

    def test_full_profile_uses_all_requested_images_and_separate_outputs(self):
        plan = build_plan(self.config)
        self.assertEqual(plan["image_limits"], {
            "df40": 100000, "140k": 140000, "openforensics": 70300, "mffi": 181947,
        })
        commands = plan["commands"]
        for dataset in ("df40", "140k", "mffi"):
            index = next(command for command in commands
                         if Path(command[1]).name == "index_datasets.py" and dataset in command)
            self.assertIn("--require-count", index)
            self.assertEqual(int(index[index.index("--limit") + 1]), plan["image_limits"][dataset])
        openforensics = next(command for command in commands
                             if Path(command[1]).name == "download_openforensics_parts.py")
        self.assertEqual(openforensics[openforensics.index("--limit") + 1], "70300")
        self.assertIn("--preprocessing-config", openforensics)
        mffi = next(command for command in commands
                    if Path(command[1]).name == "download_assets.py" and "mffi" in command)
        self.assertIn("--mffi-test-only", mffi)
        self.assertTrue(plan["split_output"].endswith("data/splits/composite_full"))
        self.assertTrue(plan["processed_root"].endswith("data/processed/full"))
        preprocessing = [command for command in commands if Path(command[1]).name == "preprocess.py"]
        self.assertEqual(len(preprocessing), 3)
        self.assertTrue(all("--min-mask-fraction" in command for command in preprocessing))

    def test_demo_is_20k_per_dataset_and_downloads_can_be_skipped(self):
        plan = build_plan(self.config, "demo", skip_download=True)
        self.assertEqual(set(plan["image_limits"].values()), {20000})
        self.assertTrue(plan["split_output"].endswith("data/splits/composite_demo"))
        self.assertTrue(plan["processed_root"].endswith("data/processed/demo"))
        self.assertFalse(any(Path(command[1]).name.startswith("download_") for command in plan["commands"]))
        df40_index = next(command for command in plan["commands"]
                          if Path(command[1]).name == "index_datasets.py" and "df40" in command)
        self.assertEqual(df40_index[df40_index.index("--limit") + 1], "20000")
        self.assertTrue(df40_index[df40_index.index("--output") + 1].endswith("df40.parquet"))
        self.assertNotEqual(plan["raw_root"], build_plan(self.config, "full")["raw_root"])
        self.assertTrue(plan["raw_root"].endswith("data/raw/demo"))

    def test_demo_can_reuse_full_preprocessing_without_downloading_or_inference(self):
        plan = build_plan(self.config, "demo", reuse_full=True)
        self.assertEqual([Path(command[1]).name for command in plan["commands"]],
                         ["sample_processed.py", "build_composite_splits.py"])
        self.assertTrue(plan["reuses_full_preprocessing"])
        self.assertEqual(plan["commands"][0][plan["commands"][0].index("--limit") + 1], "20000")
        with self.assertRaisesRegex(ValueError, "only to the demo"):
            build_plan(self.config, "full", reuse_full=True)

    def test_compact_storage_is_explicit_and_checks_sources_before_release(self):
        plan = build_plan(self.config, compact_storage=True)
        preprocessing = [command for command in plan["commands"] if Path(command[1]).name == "preprocess.py"]
        self.assertTrue(all("--compact-verification-crops" in command for command in preprocessing))
        cleanup = [command for command in plan["commands"] if Path(command[1]).name == "release_processed_sources.py"]
        self.assertEqual(len(cleanup), 3)
        self.assertTrue(all("--delete-checkpoint" in command for command in cleanup))

    def test_processed_sampling_preserves_all_faces_and_original_features_of_selected_images(self):
        records = []
        for image in range(80):
            for face in range(1 + image % 3):
                records.append({"dataset": "openforensics", "sample_id": f"image-{image}:face-{face}",
                                "image_sample_id": f"image-{image}", "target": (image + face) % 2,
                                "source_format": "JPEG", "source_jpeg_quality": 95 if image % 2 else 100,
                                "face_mask_fraction_of_crop": 0.6, "crop_sha256": f"pixels-{image}-{face}"})
        frame = pd.DataFrame(records)
        selected = sample_processed(frame, limit=20, seed=17)
        self.assertEqual(selected["image_sample_id"].nunique(), 20)
        expected = frame.loc[frame["image_sample_id"].isin(selected["image_sample_id"])].sort_values("sample_id")
        pd.testing.assert_frame_equal(selected.sort_values("sample_id").reset_index(drop=True),
                                      expected.reset_index(drop=True), check_like=True)
        repeat = sample_processed(frame, limit=20, seed=17)
        self.assertEqual(selected["sample_id"].tolist(), repeat["sample_id"].tolist())
        self.assertEqual(set(selected["source_jpeg_quality"]), {95, 100})

    def test_native_uppercase_df40_keeps_dataset_values_and_uses_the_canonical_output_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "full.parquet"
            frame = pd.DataFrame({"dataset": ["DF40"] * 30, "sample_id": [f"face-{i}" for i in range(30)],
                                  "image_sample_id": [f"image-{i}" for i in range(30)], "target": [i % 2 for i in range(30)]})
            frame.to_parquet(source, index=False)
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/sample_processed.py"), "--manifest", str(source),
                 "--limit", "20", "--output-root", str(root / "demo")],
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            selected = pd.read_parquet(root / "demo/df40/metadata.parquet")
            self.assertEqual(len(selected), 20)
            self.assertEqual(set(selected["dataset"]), {"DF40"})

    def test_incomplete_full_dataset_is_not_silently_called_a_20k_demo(self):
        frame = pd.DataFrame({"dataset": ["140k"], "sample_id": ["face"],
                              "image_sample_id": ["image"], "target": [0]})
        with self.assertRaisesRegex(ValueError, "only 1 available"):
            sample_processed(frame, limit=20)


if __name__ == "__main__":
    unittest.main()
