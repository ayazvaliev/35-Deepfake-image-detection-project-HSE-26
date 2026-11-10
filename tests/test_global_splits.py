from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.preprocessing.splits import build_composite_splits


def population():
    records = []
    for dataset in ("first", "second"):
        for target in (0, 1):
            for model in (0, 1):
                for domain in (0, 1):
                    for small_face in (0, 1):
                        for image_format in ("JPEG", "PNG"):
                            for light in ("dark", "bright"):
                                for fraction in (0.3, 0.6, 0.9):
                                    for repeat in range(8):
                                        index = len(records)
                                        author = "test" if domain else "valid" if small_face else "train"
                                        records.append({
                                            "dataset": dataset, "sample_id": f"row-{index}", "target": target,
                                            "source_group": f"pair-{index // 2}", "author_split": author,
                                            "author_split_inventory": ["train", "validate", "test"],
                                            "generator": ("StyleGAN2" if model else "StyleGAN3") if target else "real",
                                            "manipulation": ("EFS" if model else "FE") if target else "real",
                                            "domain": "cdf" if domain else "ff", "source_format": image_format,
                                            "face_size_bin": "64-128" if small_face else "128-256",
                                            "source_resolution_bin": ">=512" if domain else "256-512",
                                            "yaw_bin": "frontal" if small_face else "profile",
                                            "pitch_bin": "frontal", "roll_bin": "three_quarter",
                                            "light_bin": light, "face_mask_fraction_of_crop": fraction,
                                            "transformation_tags": ["smile", "glasses"] if repeat % 2 else ["smile"],
                                        })
    return pd.DataFrame(records)


class GlobalBalancedSplitTests(unittest.TestCase):
    def test_skewed_author_parts_are_ignored_while_feature_marginals_and_groups_are_preserved(self):
        source = population()
        result = build_composite_splits(source, seed=19)
        for partition, fraction in (("train", 0.7), ("validate", 0.15), ("test", 0.15)):
            self.assertLessEqual(abs(len(result[partition]) - fraction * len(source)), 2)
        combined = pd.concat([result[name] for name in ("train", "validate", "test")])
        self.assertEqual(set(combined["sample_id"]), set(source["sample_id"]))
        self.assertEqual(combined.groupby("source_group")["partition"].nunique().max(), 1)
        self.assertEqual(len(result["excluded"]), 0)
        self.assertEqual(result["report"]["author_overlap_removed"], 0)
        self.assertTrue(result["report"]["author_parts_ignored"])
        self.assertFalse(any(result["report"]["overlap_groups"].values()))
        fields = {"generator", "manipulation", "domain", "source_format", "yaw_bin", "light_bin",
                  "face_size_bin", "source_resolution_bin", "face_mask_fraction_of_crop_bin", "transformation_tags"}
        rows = [row for row in result["report"]["balance"] if row["feature"] in fields and row["common_support"]]
        self.assertLessEqual(max(row["max_pairwise_gap"] for row in rows), 0.04)
        for partition in ("train", "validate", "test"):
            self.assertEqual(set(result[partition]["author_split"]), {"train", "valid", "test"})
        global_rows = result["report"]["global_balance"]
        self.assertEqual({row["feature"] for row in global_rows}, {"dataset", "target", "source_format"})
        self.assertLessEqual(max(row["max_pairwise_gap"] for row in global_rows), 0.02)

    def test_groups_crossing_author_boundaries_stay_intact_without_author_purges(self):
        records = []
        for group in range(20):
            for part in ("train", "valid", "test"):
                records.append({"dataset": "source", "sample_id": f"{group}-{part}", "target": group % 2,
                                "source_group": f"group-{group}", "author_split": part,
                                "generator": "unknown", "face_mask_fraction_of_crop": 0.6})
        result = build_composite_splits(pd.DataFrame(records))
        rows = pd.concat([result[name] for name in ("train", "validate", "test")])
        self.assertEqual(len(rows), 60)
        self.assertEqual(rows.groupby("source_group")["partition"].nunique().max(), 1)
        self.assertEqual(set(rows["generator"]), {"unknown"})
        self.assertEqual(result["report"]["author_overlap_removed"], 0)
        self.assertEqual(len(result["excluded"]), 0)
        self.assertFalse(rows["flags"].map(lambda flags: "author_split_overlap" in flags).any())

    def test_exact_face_deduplication_runs_once_globally_without_author_priority(self):
        source = pd.DataFrame([
            {"dataset": "first", "sample_id": "a-training-copy", "target": 1, "author_split": "train",
             "crop_sha256": "verified-identical", "face_mask_fraction_of_crop": 0.6},
            {"dataset": "second", "sample_id": "z-test-copy", "target": 1, "author_split": "test",
             "crop_sha256": "verified-identical", "face_mask_fraction_of_crop": 0.6},
        ])
        result = build_composite_splits(source)
        kept = pd.concat([result[name] for name in ("train", "validate", "test")])
        self.assertEqual(kept["sample_id"].tolist(), ["a-training-copy"])
        self.assertEqual(result["report"]["exact_face_duplicates_removed"], 1)
        self.assertEqual(result["report"]["author_overlap_removed"], 0)
        self.assertIn("exact_face_duplicate", result["excluded"].iloc[0]["flags"])

    def test_known_global_identity_links_are_preserved_across_datasets_and_author_roles(self):
        source = pd.DataFrame([
            {"dataset": f"dataset-{index % 2}", "sample_id": f"row-{index}", "target": index % 2,
             "author_split": "train" if index % 2 else "test", "identity_group": f"global:person-{index // 2}",
             "source_group": f"image-{index}", "face_mask_fraction_of_crop": 0.6}
            for index in range(80)
        ])
        result = build_composite_splits(source)
        kept = pd.concat([result[name] for name in ("train", "validate", "test")])
        self.assertEqual(len(kept), 80)
        self.assertEqual(kept.groupby("identity_group")["partition"].nunique().max(), 1)
        self.assertFalse(any(result["report"]["overlap_groups"].values()))

    def test_verified_cache_rebuild_ignores_changed_author_roles_and_rejects_changed_evidence(self):
        source = population().iloc[:200].copy()
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "verified-groups"
            first = build_composite_splits(source, verified_group_cache=cache)
            modified = source.copy()
            modified["author_split"] = "test"
            reused = build_composite_splits(modified, verified_group_cache=cache)
            self.assertFalse(first["report"]["group_cache"]["reused"])
            self.assertTrue(reused["report"]["group_cache"]["reused"])
            self.assertEqual(reused["report"]["near_duplicate_verification"]["execution_mode"], "reused_cache")
            self.assertFalse(reused["report"]["near_duplicate_verification"]["verified_now"])
            for partition in ("train", "validate", "test", "excluded"):
                self.assertEqual(set(first[partition]["sample_id"]), set(reused[partition]["sample_id"]))
            modified.loc[modified.index[0], "source_group"] = "changed-source-evidence"
            with self.assertRaises(ValueError):
                build_composite_splits(modified, verified_group_cache=cache)

    def test_unsupported_conditional_strata_do_not_create_artificial_maximum_gaps(self):
        records = []
        for part in ("train", "valid", "test"):
            for index in range(12):
                records.append({"dataset": "source", "sample_id": f"{part}-{index}", "target": 1,
                                "author_split": part, "generator": "StyleGAN2" if part == "train" else "StyleGAN3",
                                "source_format": "PNG", "face_mask_fraction_of_crop": 0.6})
        result = build_composite_splits(pd.DataFrame(records), split_policy="author_preserved")
        formats = [row for row in result["report"]["balance"] if row["feature"] == "source_format"]
        self.assertTrue(formats)
        self.assertTrue(all(not row["common_support"] for row in formats))
        self.assertTrue(all(row["max_pairwise_gap"] is None for row in formats))
        self.assertTrue(all(row["missing_in_partition"] for row in formats))
        self.assertTrue(result["report"]["missing_strata"])


if __name__ == "__main__":
    unittest.main()
