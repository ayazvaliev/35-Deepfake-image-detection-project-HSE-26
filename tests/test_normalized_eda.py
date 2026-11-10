import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.notebook_src import df40_eda as E
from src.preprocessing.verification_views import save_crop_verification_view, view_pixel_sha256


def frame_fixture(directory):
    rows = []
    for dataset in E.DATASETS:
        for split in ("train", "validate", "test"):
            for target in (0, 1):
                index = len(rows)
                raw = np.full((80, 100, 3), 50 + index * 4, np.uint8)
                cv2.circle(raw, (50, 40), 24, (160, 110, 80), -1)
                source = directory / f"raw-{index}.png"
                crop_path = directory / f"crop-{index}.png"
                cv2.imwrite(str(source), raw)
                crop = cv2.resize(raw, (256, 256), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(str(crop_path), crop)
                rows.append({
                    "dataset": dataset, "sample_id": str(index), "image_sample_id": str(index),
                    "input_path": str(source), "crop_path": str(crop_path),
                    "file_size": source.stat().st_size, "target": target, "split": split,
                    "eligible": True, "generator": "real" if not target else "native",
                    "native_method": "native", "native_family": "EFS",
                    "manipulation": "real" if not target else "EFS", "domain": "native",
                    "author_split": "test", "source_format": "JPEG" if not target else "PNG",
                    "image_width": 100 if not target else 512, "image_height": 80 if not target else 1024,
                    "padding_fraction": 0.0, "bbox": [20, 10, 40, 55], "yaw_bin": "frontal",
                    "light_bin": "medium", "flags": [], "exclusion_flags": [],
                    "crop_sha256": view_pixel_sha256(crop),
                    **save_crop_verification_view(crop, directory / "verified"),
                })
    return pd.DataFrame(rows)


def density_areas(axis):
    return [sum(patch.get_height() * patch.get_width() for patch in group.patches)
            for group in axis.containers if hasattr(group, "patches")]


class NormalizedPlotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.frame = frame_fixture(self.directory)
        E.style()

    def tearDown(self):
        plt.close("all")
        self.temporary.cleanup()

    def test_image_hists_have_unit_area_and_shared_edges_and_categories_sum_to_one(self):
        quality = E.quality_sample(self.frame)
        figure = E.plot_image_properties(self.frame, quality)
        for index in (0, 1, 3, 4, 5):
            np.testing.assert_allclose(density_areas(figure.axes[index]), [1, 1], atol=1e-12)
            groups = figure.axes[index].containers
            left = [(patch.get_x(), patch.get_width()) for patch in groups[0].patches]
            right = [(patch.get_x(), patch.get_width()) for patch in groups[1].patches]
            np.testing.assert_allclose(left, right)
        for group in figure.axes[2].containers:
            self.assertAlmostEqual(sum(patch.get_height() for patch in group.patches), 1)
        source_files = E.plot_source_files(quality)
        np.testing.assert_allclose(density_areas(source_files.axes[0]), [1, 1], atol=1e-12)
        composition = E.plot_composition(self.frame)
        for axis in composition.axes:
            for group in axis.containers:
                self.assertAlmostEqual(sum(patch.get_height() for patch in group.patches), 1)

    def test_joint_split_probabilities_and_multilabel_prevalence_have_correct_denominators(self):
        parts = {name: self.frame.loc[self.frame["split"].eq(name)] for name in ("train", "validate", "test")}
        parts["train"] = pd.concat([parts["train"], parts["train"]], ignore_index=True)
        parts["excluded"] = pd.DataFrame([
            {"dataset": "source", "flags": ["a", "a", "b"]},
            {"dataset": "source", "flags": ["a"]},
        ])
        figure = E.plot_split_counts(parts)
        values = np.array([[patch.get_height() for patch in group.patches]
                           for group in figure.axes[0].containers])
        np.testing.assert_allclose(values.sum(axis=0), [1, 1, 1])
        flags = parts["excluded"].assign(exclusion_flags=parts["excluded"]["flags"])
        prevalence = E.plot_flags(flags)
        self.assertEqual(sorted(patch.get_width() for patch in prevalence.axes[0].patches), [0.5, 1])
        table = E.exclusion_table(parts)
        self.assertEqual(sorted(table["row_prevalence"]), [0.5, 1])
        self.assertEqual(sorted(table["records"]), [1, 2])
        exclusions = E.plot_exclusions(table)
        self.assertEqual(sorted(patch.get_width() for patch in exclusions.axes[0].patches), [0.5, 1])
        outcomes = pd.DataFrame({"outcome": ["verified", "rejected", "unreadable"],
                                 "pairs": [2, 7, 1], "share": [0.2, 0.7, 0.1]})
        pairs = E.plot_near_duplicate_audit(outcomes)
        self.assertAlmostEqual(sum(patch.get_width() for patch in pairs.axes[0].patches), 1)

    def test_distribution_refresh_uses_exact_crops_and_metadata_sizes_without_rerendering_photos(self):
        metadata = self.directory / "data/processed/full/df40/metadata.parquet"
        metadata.parent.mkdir(parents=True)
        self.frame.to_parquet(metadata, index=False)
        original_quality = E.quality_sample(self.frame)
        assets = E.pixel_illustrations(self.directory, self.frame, metadata)
        example_path = assets["figure_paths"]["examples"]
        example_sha = E._sha256(example_path)
        for column in ("input_path", "crop_path"):
            for path in self.frame[column].unique():
                Path(path).unlink()
        refreshed_quality = E.quality_sample(self.frame)
        pd.testing.assert_frame_equal(original_quality, refreshed_quality)
        refresh = E.refresh_pixel_distribution_cache(self.directory, self.frame, metadata)
        self.assertFalse(refresh["original_photos_rerendered"])
        self.assertEqual(refresh["original_raw_files_present_in_selection"], 0)
        self.assertEqual(E._sha256(example_path), example_sha)
        reused = E.pixel_illustrations(self.directory, self.frame, metadata)
        self.assertEqual(reused["mode"], "historical_pixel_artifacts")
        self.assertEqual(reused["manifest"]["distribution_normalization"]["version"], 1)
        self.assertIn(".normalized.", reused["figure_paths"]["image_properties"].name)
        self.assertEqual(refresh["examples_sha256_unchanged"], example_sha)
        expected_sizes = refreshed_quality["sample_id"].map(self.frame.set_index("sample_id")["file_size"]) / 1024
        self.assertEqual(refreshed_quality["source_kib"].tolist(), expected_sizes.tolist())


if __name__ == "__main__":
    unittest.main()
