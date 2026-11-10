import unittest

import pandas as pd

from src.preprocessing.splits import build_composite_splits


class JpegSplitTests(unittest.TestCase):
    def test_exact_quality_levels_and_subsampling_remain_balanced_with_model_and_format(self):
        records = []
        for dataset in ("first", "second"):
            for target in (0, 1):
                for quality in (90, 94, 95, 96, 99, 100, None):
                    for sampling in ("4:4:4", "4:2:0"):
                        for repeat in range(40):
                            index = len(records)
                            records.append({
                                "dataset": dataset, "sample_id": f"row-{index}", "target": target,
                                "source_group": f"group-{index // 2}", "source_format": "JPEG",
                                "source_jpeg_quality": quality, "source_jpeg_subsampling": sampling,
                                "generator": "StyleGAN2" if target else "real",
                                "author_split": "test" if quality == 100 else "train",
                                "face_mask_fraction_of_crop": 0.6,
                            })
        result = build_composite_splits(pd.DataFrame(records), seed=17)
        features = {"source_jpeg_quality_level", "source_jpeg_quality_bin", "source_jpeg_subsampling"}
        balance = [row for row in result["report"]["balance"] if row["feature"] in features]
        self.assertTrue(balance)
        self.assertLessEqual(max(row["max_pairwise_gap"] for row in balance), 0.035)
        self.assertEqual({tuple(row["condition"]) for row in balance},
                         {("dataset", "target", "generator", "source_format")})
        levels = {row["category"] for row in balance if row["feature"] == "source_jpeg_quality_level"}
        self.assertEqual(levels, {"90", "94", "95", "96", "99", "100", "unknown"})
        bins = {row["category"] for row in balance if row["feature"] == "source_jpeg_quality_bin"}
        self.assertTrue({"95", "100", "unknown"} <= bins)
        self.assertEqual(len(result["excluded"]), 0)
        self.assertFalse(any(result["report"]["overlap_groups"].values()))
        for partition, fraction in (("train", 0.7), ("validate", 0.15), ("test", 0.15)):
            self.assertLessEqual(abs(len(result[partition]) - len(records) * fraction), 2)
            self.assertTrue(result[partition]["source_jpeg_quality"].isna().any())

    def test_non_jpeg_invalid_and_missing_quality_never_become_a_fabricated_level(self):
        records = []
        for image_format, quality in (("PNG", 100), ("JPEG", 0), ("JPEG", 101),
                                      ("JPEG", 95.5), ("JPEG", None), ("JPEG", 95), ("unknown", None)):
            index = len(records)
            records.append({"dataset": "source", "sample_id": str(index), "target": 1,
                            "source_format": image_format, "source_jpeg_quality": quality,
                            "source_jpeg_quality_level": "stale", "source_jpeg_quality_bin": "stale",
                            "source_jpeg_subsampling": "4:2:0", "face_mask_fraction_of_crop": 0.6})
        result = build_composite_splits(pd.DataFrame(records))
        rows = pd.concat([result[name] for name in ("train", "validate", "test")]).set_index("sample_id")
        self.assertEqual(rows.loc["0", "source_jpeg_quality_level"], "not_jpeg")
        self.assertEqual(rows.loc["0", "source_jpeg_subsampling"], "not_jpeg")
        for index in ("1", "2", "3", "4"):
            self.assertEqual(rows.loc[index, "source_jpeg_quality_level"], "unknown")
            self.assertEqual(rows.loc[index, "source_jpeg_quality_bin"], "unknown")
        self.assertEqual(rows.loc["5", "source_jpeg_quality_level"], "95")
        self.assertEqual(rows.loc["5", "source_jpeg_quality_bin"], "95")
        self.assertEqual(rows.loc["6", "source_jpeg_quality_level"], "unknown")
        self.assertEqual(rows.loc["6", "source_jpeg_quality_bin"], "unknown")
        self.assertEqual(rows.loc["6", "source_jpeg_subsampling"], "unknown")
        self.assertEqual(result["report"]["source_jpeg_balance"]["known_quality_rows"], 1)
        self.assertEqual(result["report"]["source_jpeg_balance"]["jpeg_rows"], 5)
        self.assertFalse(result["report"]["source_jpeg_balance"]["unknown_excluded"])
        self.assertEqual(len(result["excluded"]), 0)


if __name__ == "__main__":
    unittest.main()
