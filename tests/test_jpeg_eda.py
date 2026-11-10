import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.notebook_src import df40_eda as E
from src.notebook_src import jpeg_eda as J


def fixture():
    rows = []
    for dataset in E.DATASETS:
        for split in ("train", "validate", "test"):
            for target in (0, 1):
                for quality, level, quality_bin in (
                        (95, "95", "95"), (100, "100", "100"), (80, "80", "75-89"),
                        (None, "unknown", "unknown"), (None, "not_jpeg", "not_jpeg")):
                    identifier = str(len(rows))
                    rows.append({
                        "dataset": dataset, "sample_id": identifier, "image_sample_id": identifier,
                        "input_path": "/not-read/" + identifier, "target": target, "split": split,
                        "generator": "g" if target else "real", "manipulation": "face_synthesis" if target else "real",
                        "source_format": "PNG" if level == "not_jpeg" else "JPEG",
                        "source_jpeg_quality": quality, "source_jpeg_quality_level": level,
                        "source_jpeg_quality_bin": quality_bin, "source_jpeg_subsampling": "not_jpeg" if level == "not_jpeg" else "4:2:0",
                        "source_jpeg_quality_method": "exact_ijg_qtables" if quality else level,
                        "source_jpeg_quality_status": "exact" if quality else level,
                        "source_jpeg_qtable_hash": "table:" + level if level != "not_jpeg" else None,
                        "source_jpeg_progressive": False if level != "not_jpeg" else None,
                    })
    data = pd.DataFrame(rows)
    parts = {name: data.loc[data["split"].eq(name)].copy() for name in ("train", "validate", "test")}
    parts["excluded"] = data.iloc[:0].copy()
    return data, parts


class JpegEdaTests(unittest.TestCase):
    def setUp(self):
        E.style()
        self.data, self.parts = fixture()

    def tearDown(self):
        plt.close("all")

    def render(self, figure):
        figure.canvas.draw()
        self.assertGreater(len(figure.axes), 0)
        plt.close(figure)

    def test_exact_95_and_100_unknown_and_non_jpeg_are_distinct(self):
        summary = J.jpeg_quality_summary(self.data, "images")
        self.assertEqual(summary["quality_95"].sum(), 24)
        self.assertEqual(summary["quality_100"].sum(), 24)
        self.assertEqual(summary["unknown_quality"].sum(), 24)
        self.assertEqual(summary["not_jpeg_records"].sum(), 24)
        self.assertTrue(summary["known_share_of_jpeg"].eq(0.75).all())
        values = J.jpeg_quality_values(self.data)
        self.assertEqual(set(values["source_jpeg_quality_level"]), {"95", "100", "80", "unknown", "not_jpeg"})
        self.assertEqual(values["records"].sum(), len(self.data))
        self.assertIn("distinct_qtables", J.jpeg_header_audit(self.data))

    def test_image_unit_keeps_mixed_faces_without_changing_face_targets(self):
        frame = self.data.iloc[[0, 0]].copy()
        frame.iloc[1, frame.columns.get_loc("sample_id")] = "other-face"
        frame.iloc[1, frame.columns.get_loc("target")] = 1
        image_summary = J.jpeg_quality_summary(frame, "images")
        self.assertEqual(image_summary["records"].sum(), 1)
        self.assertEqual(image_summary.iloc[0]["class"], "mixed")
        face_summary = J.jpeg_quality_summary(frame, "faces")
        self.assertEqual(face_summary["records"].sum(), 2)
        self.assertEqual(set(face_summary["class"]), {"real", "fake"})
        frame.iloc[1, frame.columns.get_loc("source_jpeg_quality_level")] = "100"
        with self.assertRaisesRegex(ValueError, "различаются"):
            J.jpeg_quality_summary(frame, "images")

    def test_quality_panels_and_before_after_metrics_use_exact_levels(self):
        for figure in (J.plot_source_jpeg_quality(self.data),
                       J.plot_split_jpeg_quality(self.parts),
                       J.plot_conditional_split_jpeg_quality(self.parts)):
            self.render(figure)
        metrics = E.global_split_divergences(self.parts)
        quality = metrics.loc[metrics["feature"].isin(J.JPEG_FEATURES)]
        self.assertEqual(set(quality["feature"]), set(J.JPEG_FEATURES))
        self.assertTrue(quality["TV"].eq(0).all())
        conditional = E.split_divergences(self.parts)
        quality = conditional.loc[conditional["feature"].isin(J.JPEG_FEATURES)]
        self.assertTrue(quality["TV"].eq(0).all())
        self.assertIn("source_format", quality)
        baseline = {name: frame.copy() for name, frame in self.parts.items()}
        changed = baseline["train"]["source_jpeg_quality_level"].eq("95")
        baseline["train"].loc[changed, "source_jpeg_quality_level"] = "100"
        baseline["train"].loc[changed, "source_jpeg_quality_bin"] = "100"
        baseline["train"].loc[changed, "source_jpeg_quality"] = 100
        comparison = E.balance_before_after(baseline, self.parts)
        selected = comparison.loc[comparison["scope"].eq("global") &
                                  comparison["feature"].eq("source_jpeg_quality_level")].set_index("version")
        self.assertGreater(selected.loc["author_baseline", "TV_max"], 0)
        self.assertEqual(selected.loc["global_balanced", "TV_max"], 0)
        self.render(E.plot_balance_before_after(comparison))

    def test_quality_axes_focus_on_observed_levels_and_keep_95_100_separate(self):
        source = J.plot_source_jpeg_quality(self.data)
        self.assertEqual(source.axes[1].get_xlim(), (79.5, 100.8))
        self.assertTrue({80, 95, 100} <= set(source.axes[1].get_xticks()))
        self.render(source)
        split = J.plot_split_jpeg_quality(self.parts)
        self.assertEqual(split.axes[1].get_xlim(), (79.5, 100.8))
        self.assertTrue({80, 95, 100} <= set(split.axes[1].get_xticks()))
        self.render(split)

    def test_unknown_only_quality_is_renderable(self):
        frame = self.data.loc[self.data["source_jpeg_quality_level"].eq("unknown")].copy()
        self.render(J.plot_source_jpeg_quality(frame))
        with self.assertRaisesRegex(ValueError, "JPEG headers"):
            J.jpeg_quality_summary(frame.drop(columns="source_jpeg_quality"))

    def test_pixel_binding_migration_requires_all_old_columns_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "data/processed/df40/metadata.parquet"
            backup = root / "data/header_cache/jpeg/backups/processed_df40.parquet"
            cache = root / "data/eda_cache/df40/manifest.json"
            proof_path = root / "data/header_cache/jpeg/enrichment_provenance.json"
            for path in (current, backup, cache, proof_path):
                path.parent.mkdir(parents=True, exist_ok=True)
            old = pd.DataFrame({"sample_id": ["a"], "target": [1], "bbox": [np.array([0, 0, 32, 32])]})
            old.to_parquet(backup, index=False)
            new = old.assign(source_jpeg_quality=[95])
            new.to_parquet(current, index=False)
            record = {
                "path": str(current.relative_to(root)), "backup_path": str(backup.relative_to(root)),
                "old_sha256": E._sha256(backup), "new_sha256": E._sha256(current),
                "old_columns": list(old.columns), "all_old_columns_equal": True,
                "rows": 1, "columns_added": ["source_jpeg_quality"],
            }
            proof_path.write_text(json.dumps({"schema_version": 1, "files": [record]}))
            example = cache.parent / "examples.png"
            example.write_bytes(b"actual cached illustration")
            original = {"metadata_sha256": record["old_sha256"], "source_notebook_sha256": "preserved",
                        "artifacts": {"examples": {"path": "examples.png", "sha256": E._sha256(example)}}}
            cache.write_text(json.dumps(original))
            result = J.migrate_pixel_metadata_binding(root, proof_path)
            self.assertEqual(result["status"], "metadata_extension_migration")
            manifest = json.loads(cache.read_text())
            self.assertEqual(manifest["metadata_sha256"], record["new_sha256"])
            self.assertEqual(manifest["source_notebook_sha256"], original["source_notebook_sha256"])
            self.assertEqual(manifest["artifacts"], original["artifacts"])
            self.assertEqual(J.migrate_pixel_metadata_binding(root, proof_path)["status"], "already_bound")
            new["target"] = 0
            new.to_parquet(current, index=False)
            record["new_sha256"] = E._sha256(current)
            proof_path.write_text(json.dumps({"schema_version": 1, "files": [record]}))
            with self.assertRaises(AssertionError):
                J.migrate_pixel_metadata_binding(root, proof_path)


if __name__ == "__main__":
    unittest.main()
