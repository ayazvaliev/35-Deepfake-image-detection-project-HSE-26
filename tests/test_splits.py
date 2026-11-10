import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from src.preprocessing.splits import build_composite_splits, proportional_sample


def manifest(rows):
    return pd.DataFrame([
        {"dataset": "example", "sample_id": f"image-{i}", "target": i % 2,
         "face_mask_fraction_of_crop": 0.5, **row}
        for i, row in enumerate(rows)
    ])


class SplitTests(unittest.TestCase):
    def test_proportional_sample_uses_largest_remainder_and_is_reproducible(self):
        data = pd.DataFrame({"generator": ["a"] * 71 + ["b"] * 29, "target": [1] * 100})
        first = proportional_sample(data, 17, seed=5)
        second = proportional_sample(data, 17, seed=5)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 17)
        self.assertEqual(first["generator"].value_counts().to_dict(), {"a": 12, "b": 5})
        self.assertFalse(first.index.duplicated().any())

    def test_sampling_does_not_silently_return_less_than_requested(self):
        with self.assertRaises(ValueError):
            proportional_sample(pd.DataFrame({"target": [0]}), 100_000)

    def test_source_groups_never_cross_partitions_and_marginals_balance(self):
        rows = []
        for group in range(40):
            rows.extend([
                {"source_group": f"video-{group}", "target": group % 2,
                 "generator": f"method-{group % 4}", "yaw_bin": str(group % 3),
                 "light_bin": str(group % 2)}
                for _ in range(3)
            ])
        data = manifest(rows)
        result = build_composite_splits(data, validation_fraction=0.25)
        combined = pd.concat([result["train"], result["validate"], result["test"]])
        self.assertEqual(len(combined), 120)
        self.assertEqual(combined.groupby("source_group")["partition"].nunique().max(), 1)
        self.assertLessEqual(abs(len(result["validate"]) - 30), 3)
        self.assertLess(abs(result["learn"]["target"].mean() - 0.5), 0.08)
        self.assertLess(abs(result["validate"]["target"].mean() - 0.5), 0.1)
        self.assertFalse(any(result["report"]["overlap_groups"].values()))
        again = build_composite_splits(data.sample(frac=1, random_state=2), validation_fraction=0.25)
        self.assertEqual(set(result["learn"]["sample_id"]), set(again["learn"]["sample_id"]))

    def test_author_test_is_reserved_and_lower_priority_overlap_is_removed(self):
        data = manifest([
            {"source_group": "shared", "author_split": "Train"},
            {"source_group": "shared", "author_split": "Val"},
            {"source_group": "heldout", "author_split": "train"},
            {"source_group": "heldout", "author_split": "test"},
            {"source_group": "onlytrain", "author_split": "train"},
        ])
        result = build_composite_splits(data, split_policy="author_preserved")
        self.assertEqual(result["learn"]["sample_id"].tolist(), ["image-4"])
        self.assertEqual(result["validate"]["sample_id"].tolist(), ["image-1"])
        self.assertEqual(result["test"]["sample_id"].tolist(), ["image-3"])
        self.assertEqual(set(result["excluded"]["sample_id"]), {"image-0", "image-2"})

    def test_mixed_face_image_hash_is_not_a_label_conflict(self):
        data = manifest([
            {"image_sha256": "same-image", "crop_sha256": "face-a", "target": 0},
            {"image_sha256": "same-image", "crop_sha256": "face-b", "target": 1},
        ])
        result = build_composite_splits(data)
        self.assertEqual(result["report"]["contradictory_crop_rows"], 0)
        self.assertEqual(len(result["excluded"]), 0)
        self.assertEqual(len(result["learn"]) + len(result["validate"]), 2)

    def test_same_face_contradictory_labels_exclude_every_copy(self):
        data = manifest([
            {"dataset": "a", "crop_sha256": "same-face", "target": 0},
            {"dataset": "b", "crop_sha256": "same-face", "target": 1},
            {"dataset": "b", "crop_sha256": "other-face", "target": 1},
        ])
        result = build_composite_splits(data)
        self.assertEqual(result["report"]["contradictory_crop_rows"], 2)
        self.assertEqual(set(result["excluded"]["sample_id"]), {"image-0", "image-1"})

    def test_near_duplicate_is_conservatively_grouped_and_author_train_is_purged(self):
        data = manifest([
            {"dataset": "a", "phash": "0000000000000001", "author_split": "train"},
            {"dataset": "b", "phash": "0000000000000003", "author_split": "val"},
            {"dataset": "c", "phash": "ffffffffffffffff", "author_split": "train"},
        ])
        result = build_composite_splits(data, near_duplicate_threshold=1, near_duplicate_verifier=lambda first, second: True, split_policy="author_preserved")
        self.assertEqual(result["excluded"]["sample_id"].tolist(), ["image-0"])
        self.assertEqual(result["report"]["near_duplicate_candidate_pairs"], 1)
        self.assertEqual(result["report"]["contradictory_crop_rows"], 0)
        self.assertEqual(result["learn"]["sample_id"].tolist(), ["image-2"])

    def test_ineligible_gray_crops_do_not_bridge_distinct_valid_sources(self):
        data = manifest([
            {"source_group": "source-a", "author_split": "train", "eligible": True,
             "crop_sha256": "valid-a", "phash": "0000000000000001", "target": 0},
            {"source_group": "source-a", "author_split": "train", "eligible": False,
             "crop_sha256": "gray", "phash": "00000000000000ff", "target": 0},
            {"source_group": "source-b", "author_split": "test", "eligible": True,
             "crop_sha256": "valid-b", "phash": "ffffffffffffffff", "target": 1},
            {"source_group": "source-b", "author_split": "test", "eligible": False,
             "crop_sha256": "gray", "phash": "00000000000000fe", "target": 1},
        ])
        result = build_composite_splits(data, near_duplicate_threshold=1, split_policy="author_preserved")
        self.assertEqual(result["train"]["sample_id"].tolist(), ["image-0"])
        self.assertEqual(result["test"]["sample_id"].tolist(), ["image-2"])
        self.assertEqual(set(result["excluded"]["sample_id"]), {"image-1", "image-3"})
        self.assertEqual(result["report"]["author_overlap_removed"], 0)
        self.assertEqual(result["report"]["contradictory_crop_rows"], 0)
        self.assertEqual(result["report"]["near_duplicate_candidate_pairs"], 0)

    def test_auxiliary_labels_are_canonical_and_multilabel_shares_use_faces(self):
        rows = []
        for partition in ("train", "valid", "test"):
            for index, tags in enumerate((["smile", "glasses"], ["smile"], [], [])):
                rows.append({
                    "target": 1, "author_split": partition,
                    "generator": "StyleGAN2" if index % 2 else "StyleGAN 2",
                    "manipulation": "EFS" if index % 2 else "EntireFaceSynthesis",
                    "native_method": "PublisherStyleGAN2",
                    "transformation_tags": tags,
                })
        source = manifest(rows)
        result = build_composite_splits(source, split_policy="author_preserved")
        combined = pd.concat([result[name] for name in ("train", "validate", "test")])
        self.assertEqual(combined["generator"].nunique(), 1)
        self.assertEqual(combined["manipulation"].nunique(), 1)
        self.assertEqual(set(combined["generator_raw"]), {"StyleGAN2", "StyleGAN 2"})
        self.assertEqual(set(combined["manipulation_raw"]), {"EFS", "EntireFaceSynthesis"})
        self.assertEqual(set(combined["native_method"]), {"PublisherStyleGAN2"})
        self.assertTrue(combined["target"].eq(1).all())
        tags = [record for record in result["report"]["balance"] if record["feature"] == "transformation_tags"]
        self.assertTrue(tags)
        for record in tags:
            self.assertTrue(record["multilabel"])
            for partition in ("train", "validate", "test"):
                self.assertEqual(record[partition + "_share"], record[partition + "_n"] / 4)
        self.assertIn("registry_version", result["report"]["label_normalization"])

    def test_df40_requires_real_source_video_groups(self):
        with self.assertRaisesRegex(ValueError, "source_group"):
            build_composite_splits(manifest([{"dataset": "df40"}]))

    def test_same_label_exact_face_duplicates_keep_one_copy(self):
        data = manifest([
            {"dataset": "a", "crop_sha256": "face", "target": 0},
            {"dataset": "b", "crop_sha256": "face", "target": 0},
        ])
        result = build_composite_splits(data)
        self.assertEqual(result["report"]["exact_face_duplicates_removed"], 1)
        self.assertEqual(len(result["learn"]) + len(result["validate"]), 1)

    def test_df40_validation_is_derived_only_from_author_train_groups(self):
        rows = [
            {"dataset": "df40", "source_group": f"source-{i}", "author_split": "train"}
            for i in range(40)
        ]
        rows.extend([
            {"dataset": "df40", "source_group": "held-out", "author_split": "test"},
            {"dataset": "df40", "source_group": "held-out", "author_split": "train"},
        ])
        result = build_composite_splits(manifest(rows), split_policy="author_preserved")
        self.assertGreater(len(result["learn"]), 0)
        self.assertGreater(len(result["validate"]), 0)
        self.assertEqual(set(result["validate"]["author_split"]), {"train"})
        self.assertEqual(result["test"]["sample_id"].tolist(), ["image-40"])
        self.assertIn("image-41", set(result["excluded"]["sample_id"]))

    def test_ineligible_unlabeled_records_remain_excluded(self):
        result = build_composite_splits(manifest([
            {"target": None, "eligible": False, "flags": ["decoding_failed"]},
            {"target": 1, "author_split": "train", "eligible": True},
        ]))
        self.assertEqual(len(result["excluded"]), 1)
        self.assertEqual(len(result["learn"]), 1)

    def test_empty_manifest_has_empty_outputs(self):
        result = build_composite_splits(pd.DataFrame(columns=["dataset", "sample_id", "target"]))
        self.assertTrue(all(len(result[key]) == 0 for key in ("learn", "validate", "test", "excluded")))

    def test_source_formats_are_balanced_within_generator(self):
        data = manifest([
            {"dataset": "df40", "source_group": f"source-{i}", "author_split": "train",
             "generator": f"method-{i % 4}", "target": i % 2,
             "source_format": "JPEG" if (i // 4) % 2 else "PNG"}
            for i in range(128)
        ])
        result = build_composite_splits(data, validation_fraction=0.25)
        formats = [row for row in result["report"]["balance"] if row["feature"] == "source_format"]
        self.assertTrue(formats)
        self.assertLessEqual(max(row["abs_gap"] for row in formats), 0.15)
        self.assertTrue(all("generator" in row["condition"] for row in formats))
        self.assertEqual(result["report"]["final_crop_format"], "PNG")

    def test_cli_exports_real_parquet_csv_and_audit(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            data = manifest([
                {"dataset": "df40", "source_group": f"source-{i}", "author_split": "train"}
                for i in range(40)
            ])
            source = directory / "faces.parquet"
            data.to_parquet(source, index=False)
            output = directory / "splits"
            process = subprocess.run([
                sys.executable, "-B", str(root / "scripts/build_composite_splits.py"),
                "--manifest", str(source), "--output", str(output),
            ], capture_output=True, text=True, check=False)
            self.assertEqual(process.returncode, 0, process.stderr)
            audit = json.loads((output / "audit.json").read_text())
            self.assertTrue(audit["ready_for_training"])
            self.assertEqual(audit["split_policy"], "global_balanced")
            self.assertEqual(audit["author_overlap_removed"], 0)
            learn = pd.read_parquet(output / "train.parquet")
            validate = pd.read_csv(output / "validate.csv")
            self.assertEqual(len(learn) + len(validate) + audit["test_rows"], len(data))
            self.assertFalse(set(learn["sample_id"]) & set(validate["sample_id"]))
            self.assertTrue((output / "excluded.parquet").is_file())

    def test_test_only_input_is_not_repurposed_as_training(self):
        data = manifest([{"author_split": "Test-Dev"} for _ in range(8)])
        result = build_composite_splits(data, split_policy="author_preserved")
        self.assertEqual(len(result["learn"]), 0)
        self.assertEqual(len(result["validate"]), 0)
        self.assertEqual(len(result["test"]), 8)


if __name__ == "__main__":
    unittest.main()
