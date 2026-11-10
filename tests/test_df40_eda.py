import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.notebook_src import df40_eda as E


def fixture(directory):
    records = []
    for dataset in E.DATASETS:
        for partition in ("train", "validate", "test"):
            for target in (0, 1):
                index = len(records)
                image = np.full((80, 100, 3), 35 + index * 7, dtype=np.uint8)
                cv2.rectangle(image, (20, 10), (60, 65), (150, 110, 90), -1)
                image_path = directory / f"image-{index}.png"
                cv2.imwrite(str(image_path), image)
                identifier = f"sample-{index}"
                value = hashlib.sha256(identifier.encode()).hexdigest()
                records.append({
                    "dataset": dataset, "sample_id": identifier, "image_sample_id": identifier,
                    "face_index": 0,
                    "group_id": identifier, "input_path": str(image_path),
                    "path": str(image_path), "crop_path": str(image_path),
                    "target": target, "partition": partition, "author_split": partition,
                    "generator": "native-generator" if target else "real",
                    "generator_raw": "Native Generator" if target else "Real",
                    "manipulation_raw": "EntireFaceSynthesis" if target else "Real",
                    "transformation_raw": "Smile" if target else None,
                    "transformation": "smile" if target else "unknown",
                    "transformation_tags": (["smile", "glasses"] if index % 3 == 0 else ["smile"]) if target else [],
                    "native_method": "native-generator", "native_family": "EFS",
                    "manipulation": "EFS" if target else "real",
                    "domain": "ffhq", "source_format": "PNG", "source_format_verified": True,
                    "image_width": 100, "image_height": 80, "bbox": [20, 10, 40, 55],
                    "padding_fraction": 0.0, "eligible": True, "flags": [],
                    "exclusion_flags": [], "image_sha256": value, "decoded_sha256": value,
                    "crop_sha256": value, "yaw": float(index % 20), "pitch": -4.0,
                    "roll": 2.0, "yaw_bin": "frontal" if index % 20 < 15 else "three_quarter",
                    "pitch_bin": "frontal", "roll_bin": "frontal",
                    "lightness_lstar": 35.0 + index, "light_bin": "medium",
                    "face_mask_fraction_of_crop": 0.15 + (index % 5) * 0.16,
                    "face_mask_fraction_of_crop_bin": ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1"][min(index % 5, 3)],
                    "mask_source": "annotation" if dataset == "openforensics" else "mediapipe_face_oval",
                    "face_size_bin": "32-64", "padding_bin": "none", "source_resolution_bin": "<256",
                })
    frame = pd.DataFrame(records)
    parts = {partition: frame.loc[frame["partition"].eq(partition)].copy()
             for partition in ("train", "validate", "test")}
    parts["excluded"] = frame.iloc[:0].copy()
    return frame, parts


class PlottingSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.frame, self.parts = fixture(self.directory)
        E.style()

    def tearDown(self):
        plt.close("all")
        self.temporary.cleanup()

    def render(self, figure):
        figure.canvas.draw()
        self.assertGreater(len(figure.axes), 0)
        plt.close(figure)

    def test_every_eda_and_composite_panel_renders(self):
        quality = E.quality_sample(self.frame, maximum=12)
        plots = [
            lambda: E.plot_sampling(self.frame, self.frame),
            lambda: E.plot_composition(self.frame),
            lambda: E.plot_examples(self.frame),
            lambda: E.plot_image_properties(self.frame, quality),
            lambda: E.plot_source_files(quality),
            lambda: E.plot_attributes(self.frame),
            lambda: E.plot_flags(self.frame),
            lambda: E.plot_face_mask_fraction(self.frame),
            lambda: E.plot_split_mask_fraction(self.parts),
            lambda: E.plot_split_counts(self.parts),
            lambda: E.plot_conditional_bins(self.parts, 0),
            lambda: E.plot_conditional_bins(self.parts, 1),
            lambda: E.plot_conditional_bins(self.parts, 0, features=("face_size_bin", "padding_bin", "source_resolution_bin")),
            lambda: E.plot_split_continuous(self.parts),
            lambda: E.plot_divergences(E.split_divergences(self.parts)),
            lambda: E.plot_exclusions(E.exclusion_table(self.parts)),
        ]
        for plot in plots:
            self.render(plot())
        self.assertTrue(E.sampling_fidelity(self.frame, self.frame)["TV"].eq(0).all())
        self.assertTrue(E.overlap_audit(self.parts)["между сплитами"].eq(0).all())
        self.assertTrue(E.exclusion_table(self.parts).empty)

    def test_unknown_pose_and_light_bins_are_renderable(self):
        missing = self.frame.copy()
        for column in ("yaw", "pitch", "roll", "lightness_lstar", "face_mask_fraction_of_crop"):
            missing[column] = np.nan
        for column in ("yaw_bin", "pitch_bin", "roll_bin", "light_bin"):
            missing[column] = "unknown"
        self.render(E.plot_attributes(missing))
        table = E.missing_table(missing)
        self.assertTrue(table.loc[table["признак"].ne("source_format"), "доля"].eq(1).all())
        parts = {key: value.copy() for key, value in self.parts.items()}
        parts["test"].loc[:, "group_id"] = parts["train"]["group_id"].to_numpy()
        self.assertGreater(E.overlap_audit(parts).loc["group_id", "между сплитами"], 0)

    def test_taxonomy_and_tag_presence_comparisons_keep_raw_auxiliary_labels(self):
        table = E.taxonomy_table(self.parts)
        self.assertEqual(set(table["field"]), {"generator", "manipulation", "transformation"})
        self.assertTrue(table.groupby("field")["records"].sum().eq(len(self.frame)).all())
        self.assertIn("Native Generator", set(table.loc[table["field"].eq("generator"), "raw"]))
        divergences = E.split_divergences(self.parts)
        tags = divergences.loc[divergences["feature"].eq("transformation_tags")]
        self.assertTrue(len(tags) > 0)
        self.assertTrue(tags["tag"].notna().all())
        self.assertTrue(tags["TV"].between(0, 1).all())
        self.assertTrue(tags["JS_bits"].between(0, 1).all())

    def test_stage_counts_do_not_call_error_placeholders_detected_faces(self):
        placeholder = self.frame.iloc[[0]].copy()
        placeholder["face_index"] = np.nan
        placeholder["crop_path"] = None
        placeholder["eligible"] = False
        records = pd.concat([self.frame, placeholder], ignore_index=True)
        table = E.stage_table(self.frame, self.frame, records)
        row = table.loc["Строки обработки (включая ошибки)"]
        self.assertEqual(row["строк"], len(self.frame) + 1)
        self.assertEqual(row["записей bbox (face_index)"], len(self.frame))
        self.assertEqual(row["сохранённых кропов"], len(self.frame))
        self.assertEqual(row["eligible"], len(self.frame))
        excluded_parts = {name: part.copy() for name, part in self.parts.items()}
        excluded_parts["excluded"] = placeholder.assign(flags=[["no_face"]])
        exclusion_counts = E.exclusion_table(excluded_parts)
        self.assertEqual(exclusion_counts["records"].sum(), 1)
        self.render(E.plot_exclusions(exclusion_counts))

    def test_mask_quality_summary_distinguishes_missing_small_and_unknown_model(self):
        data = pd.DataFrame([
            {"dataset": "d", "target": 1, "face_mask_fraction_of_crop": 0.25, "eligible": False,
             "generator": "known", "exclusion_flags": ["face_mask_too_small"]},
            {"dataset": "d", "target": 1, "face_mask_fraction_of_crop": None, "eligible": False,
             "generator": "known", "exclusion_flags": ["face_mask_missing"]},
            {"dataset": "d", "target": 1, "face_mask_fraction_of_crop": 0.6, "eligible": True,
             "generator": "unknown", "exclusion_flags": []},
        ])
        row = E.mask_quality_summary(data).iloc[0]
        self.assertEqual(row["processing_records"], 3)
        self.assertEqual(row["mask_fraction_le_threshold"], 1)
        self.assertEqual(row["missing_mask_fraction"], 1)
        self.assertEqual(row["eligible_current"], 1)
        self.assertEqual(row["eligible_unknown_generator"], 1)
        self.assertEqual(row["eligible_violating_mask_rule"], 0)

    def test_detector_coverage_uses_images_and_one_real_run(self):
        data = pd.DataFrame([
            {"image_sample_id": "a", "native_method": "large", "detector_native_faces": 0,
             "detector_downscaled_faces": 1, "detected_faces": 1},
            {"image_sample_id": "b", "native_method": "mixed", "detector_native_faces": 1,
             "detector_downscaled_faces": 1, "detected_faces": 2},
            {"image_sample_id": "b", "native_method": "mixed", "detector_native_faces": 1,
             "detector_downscaled_faces": 1, "detected_faces": 2},
            {"image_sample_id": "c", "native_method": "none", "detector_native_faces": 0,
             "detector_downscaled_faces": 0, "detected_faces": 0},
            {"image_sample_id": "d", "native_method": "broken", "detector_native_faces": None,
             "detector_downscaled_faces": None, "detected_faces": None},
        ])
        result = E.detector_coverage_summary(data)
        row = result["overall"].iloc[0]
        self.assertEqual(row["input_images"], 4)
        self.assertEqual(row["native_face_images"], 1)
        self.assertEqual(row["merged_face_images"], 2)
        self.assertEqual(row["rescued_images"], 1)
        self.assertEqual(row["measured_images"], 3)
        self.assertEqual(row["native_face_lost_images"], 0)
        self.render(E.plot_detector_coverage(result))

    def test_source_format_target_finding_is_driven_by_the_supplied_labels(self):
        data = pd.DataFrame([
            {"dataset": "a", "source_format": "PNG", "target": 1},
            {"dataset": "a", "source_format": "JPEG", "target": 0},
            {"dataset": "a", "source_format": "JPEG", "target": 1},
            {"dataset": "b", "source_format": "JPEG", "target": 0},
            {"dataset": "b", "source_format": "JPEG", "target": 1},
        ])
        table = E.format_target_table(data)
        self.assertEqual(table.loc[table["source_format"].eq("PNG"), "only_target"].tolist(), ["fake"])
        self.assertIn("PNG только у fake (лиц: 1)", E.format_target_finding(table))
        self.assertTrue(table.loc[table["source_format"].eq("JPEG"), "only_target"].eq("mixed").all())

    def test_removed_png_is_replaced_by_the_identical_verified_crop(self):
        from src.preprocessing.verification_views import save_crop_verification_view, view_pixel_sha256

        row = self.frame.iloc[0].to_dict()
        original = cv2.imread(row["input_path"])
        crop = cv2.resize(original, (256, 256), interpolation=cv2.INTER_CUBIC)
        row.update(save_crop_verification_view(crop, self.directory / "verified-crops"))
        row["crop_path"] = str(self.directory / "removed-training-crop.png")
        row["crop_sha256"] = view_pixel_sha256(crop)
        np.testing.assert_array_equal(E._read_crop(row), crop)
        frame = pd.DataFrame([row])
        self.assertEqual(len(E.quality_sample(frame, maximum=1)), 1)
        self.render(E.plot_examples(frame))
        row["crop_verification_pixel_sha256"] = "different-pixels"
        with self.assertRaisesRegex(ValueError, "not identical"):
            E._read_crop(row)

    def test_pixel_panels_are_fresh_or_verified_historical_artifacts(self):
        root = self.directory
        metadata = root / "data/processed/df40/metadata.parquet"
        metadata.parent.mkdir(parents=True)
        self.frame.to_parquet(metadata, index=False)
        fresh = E.pixel_illustrations(root, self.frame, metadata)
        self.assertEqual(fresh["mode"], "fresh_pixels")
        self.assertEqual(fresh["manifest"]["quality_sample_size"], len(self.frame))
        self.assertIn("чёткость_медиана", fresh["quality_summary_html"])
        for filename in self.frame["crop_path"].unique():
            Path(filename).unlink()
        historical = E.pixel_illustrations(root, self.frame, metadata)
        self.assertEqual(historical["mode"], "historical_pixel_artifacts")
        self.assertEqual(historical["quality_summary_html"], fresh["quality_summary_html"])
        source_snapshot = root / "data/splits/author_baseline/eda_df40_executed.ipynb"
        source_snapshot.parent.mkdir(parents=True)
        source_snapshot.write_bytes(b"verified original executed notebook")
        manifest_path = root / "data/eda_cache/df40/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["source_commit"] = "historical-source"
        manifest["source_notebook_snapshot"] = str(source_snapshot.relative_to(root))
        manifest["source_notebook_sha256"] = hashlib.sha256(source_snapshot.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        self.assertEqual(E.pixel_illustrations(root, self.frame, metadata)["mode"], "historical_pixel_artifacts")
        original_snapshot = source_snapshot.read_bytes()
        source_snapshot.write_bytes(b"changed original notebook")
        with self.assertRaisesRegex(ValueError, "исходного выполненного notebook"):
            E.pixel_illustrations(root, self.frame, metadata)
        source_snapshot.write_bytes(original_snapshot)
        saved = metadata.read_bytes()
        metadata.write_bytes(b"changed metadata")
        with self.assertRaisesRegex(ValueError, "другой версии metadata"):
            E.pixel_illustrations(root, self.frame, metadata)
        metadata.write_bytes(saved)
        historical["figure_paths"]["examples"].write_bytes(b"changed figure")
        with self.assertRaisesRegex(ValueError, "Повреждён"):
            E.pixel_illustrations(root, self.frame, metadata)

    def test_baseline_comparison_requires_the_same_input_cohort(self):
        E.assert_same_cohort(self.parts, self.parts)
        different = {name: frame.copy() for name, frame in self.parts.items()}
        different["train"].iloc[0, different["train"].columns.get_loc("target")] = 1
        with self.assertRaisesRegex(ValueError, "разным input records"):
            E.assert_same_cohort(self.parts, different)

    def test_global_and_conditional_balance_comparison_uses_actual_manifests(self):
        global_metrics = E.global_split_divergences(self.parts)
        for feature in ("dataset", "target", "source_format"):
            values = global_metrics.loc[global_metrics["feature"].eq(feature), "TV"]
            self.assertTrue(values.eq(0).all())
        baseline = {name: frame.copy() for name, frame in self.parts.items()}
        baseline["train"] = baseline["train"].loc[baseline["train"]["target"].eq(1)]
        comparison = E.balance_before_after(baseline, self.parts)
        target = comparison.loc[comparison["scope"].eq("global") & comparison["feature"].eq("target")]
        target = target.set_index("version")
        self.assertEqual(target.loc["author_baseline", "TV_max"], 0.5)
        self.assertEqual(target.loc["global_balanced", "TV_max"], 0)
        self.render(E.plot_global_marginals(self.parts))
        self.render(E.plot_balance_before_after(comparison))

    def test_df40_loader_uses_retained_inventory_and_image_metadata_after_cleanup(self):
        root = self.directory / "retained"
        inventory = root / "data/manifests"
        processed = root / "data/processed/df40"
        snapshot = root / "reports/generated/acquisition/df40"
        for directory in (inventory, processed, snapshot):
            directory.mkdir(parents=True)
        self.frame.to_parquet(inventory / "df40_inventory.parquet", index=False)
        self.frame.to_parquet(processed / "metadata.parquet", index=False)
        (processed / "preprocessing_audit.json").write_text("{}")
        (snapshot / "sampling.json").write_text(json.dumps({"seed": 42}))
        with patch.dict(os.environ, {"DEEPFAKE_DATA_ROOT": str(self.directory / "removed-raw")}):
            _, sample, faces, sampling, _ = E.load_df40(root)
        self.assertEqual(set(sample["sample_id"]), set(faces["image_sample_id"]))
        self.assertEqual(sample.attrs["artifact_source"], "reconstructed_from_processed_image_metadata")
        self.assertEqual(sampling["seed"], 42)

    def test_pixel_verification_audit_keeps_pairs_separate_from_rows(self):
        audit = {"near_duplicate_verification": {
            "mode": "pixel_verification", "candidate_pairs": 11, "verified_pairs": 3,
            "rejected_pairs": 6, "unreadable_pairs": 2, "exact_metadata_pairs": 1,
            "verified_original_pairs": 1, "verified_crop_pairs": 1, "low_texture_rejections": 2,
            "unreadable_rows": 3, "excluded_rows": 3,
            "settings": {"ssim_min": 0.95, "rgb_mae_max": 5 / 255, "min_gray_std": 8 / 255},
        }}
        result = E.near_duplicate_audit(audit)
        self.assertEqual(result["outcomes"]["pairs"].sum(), 11)
        self.assertAlmostEqual(result["outcomes"]["share"].sum(), 1)
        detail = result["details"].set_index("metric")
        self.assertEqual(detail.loc["все пары-кандидаты pHash", "unit"], "пар")
        self.assertEqual(detail.loc["строки без доступного сравнения", "unit"], "строк")
        self.assertAlmostEqual(result["settings"].loc["rgb_mae_max_0_255", "value"], 5)
        self.assertEqual(result["settings"].loc["ssim_min", "value"], 0.95)
        self.render(E.plot_near_duplicate_audit(result["outcomes"]))
        with self.assertRaisesRegex(ValueError, "pixel_verification"):
            E.near_duplicate_audit({"near_duplicate_candidate_pairs": 11})
        audit["near_duplicate_verification"]["rejected_pairs"] = 7
        with self.assertRaisesRegex(ValueError, "числом пар"):
            E.near_duplicate_audit(audit)
        for key in ("candidate_pairs", "verified_pairs", "rejected_pairs", "unreadable_pairs",
                    "exact_metadata_pairs", "verified_original_pairs", "verified_crop_pairs",
                    "low_texture_rejections", "unreadable_rows", "excluded_rows"):
            audit["near_duplicate_verification"][key] = 0
        empty = E.near_duplicate_audit(audit)
        self.assertTrue(empty["outcomes"]["share"].isna().all())
        self.render(E.plot_near_duplicate_audit(empty["outcomes"]))

    def test_unknown_lineage_units_do_not_turn_batches_into_identities(self):
        data = pd.DataFrame([
            {"dataset": "DF40", "source_group_basis": "generated_image_unknown_identity",
             "lineage_status": "unknown", "source_group": "image:a", "identity_group": "",
             "generation_batch": "batch:one"},
            {"dataset": "df40", "source_group_basis": "generated_image_unknown_identity",
             "lineage_status": "unknown", "source_group": "image:b", "identity_group": None,
             "generation_batch": "batch:one"},
            {"dataset": "df40", "source_group_basis": "explicit_source_identifier",
             "lineage_status": "known_source", "source_group": "source:c", "identity_group": "person:c",
             "generation_batch": ""},
            {"dataset": "140k", "source_group": "different-dataset"},
        ])
        table = E.lineage_table(data).set_index("source_group_basis")
        generated = table.loc["generated_image_unknown_identity"]
        self.assertEqual(generated["processing_records"], 2)
        self.assertEqual(generated["source_groups"], 2)
        self.assertEqual(generated["known_identity_records"], 0)
        self.assertEqual(generated["generation_batches_audit_only"], 1)
        self.assertEqual(table["processing_records"].sum(), 3)
        missing = E.lineage_table(pd.DataFrame([{"dataset": "df40"}])).iloc[0]
        self.assertEqual(missing["source_group_basis"], "unknown")
        self.assertEqual(missing["known_identity_records"], 0)
        self.assertTrue(E.lineage_table(pd.DataFrame([{"dataset": "140k"}])).empty)

    def test_loaders_honor_external_raw_root_and_composite_schema(self):
        root = self.directory / "repo"
        external = self.directory / "external-raw"
        raw = external / "df40"
        processed = root / "data/processed/df40"
        raw.mkdir(parents=True)
        processed.mkdir(parents=True)
        self.frame.to_parquet(raw / "inventory.parquet", index=False)
        self.frame.to_parquet(raw / "sample.parquet", index=False)
        self.frame.to_parquet(processed / "metadata.parquet", index=False)
        (raw / "sampling.json").write_text(json.dumps({"seed": 42}))
        (processed / "preprocessing_audit.json").write_text("{}")
        with patch.dict(os.environ, {"DEEPFAKE_DATA_ROOT": str(external)}):
            loaded = E.load_df40(root)
        self.assertEqual(len(loaded[0]), len(self.frame))
        self.assertEqual(loaded[3]["seed"], 42)
        composite = root / "data/splits/composite"
        composite.mkdir(parents=True)
        for name, frame in self.parts.items():
            frame.to_parquet(composite / f"{name}.parquet", index=False)
        (composite / "audit.json").write_text(json.dumps({"train_rows": len(self.parts["train"])}))
        parts, audit = E.load_composite(root)
        self.assertEqual(set(parts), {"train", "validate", "test", "excluded"})
        self.assertEqual(audit["train_rows"], len(parts["train"]))
        self.assertTrue(E.exclusion_table(parts).empty)


class CropBoundaryTests(unittest.TestCase):
    def test_image_denominator_primary_face_and_strict_padding_thresholds(self):
        frame = pd.DataFrame([
            {"image_sample_id": "a", "sample_id": "a:0", "bbox": [0, 0, 100, 100],
             "face_index": 0, "padding_fraction": 0.1, "eligible": True, "flags": [],
             "image_width": 200, "image_height": 200, "native_method": "e4e", "target": 1, "source_format": "JPEG"},
            {"image_sample_id": "a", "sample_id": "a:1", "bbox": [0, 0, 10, 10],
             "face_index": 1, "padding_fraction": 0.9, "eligible": False, "flags": ["face_too_small"],
             "image_width": 200, "image_height": 200, "native_method": "e4e", "target": 1, "source_format": "JPEG"},
            {"image_sample_id": "b", "sample_id": "b:0", "bbox": [0, 0, 30, 30],
             "face_index": 0, "padding_fraction": 0.0, "eligible": True, "flags": [],
             "image_width": 100, "image_height": 200, "native_method": "StyleGAN2", "target": 1, "source_format": "PNG"},
            {"image_sample_id": "c", "sample_id": "c:0", "bbox": None,
             "padding_fraction": np.nan, "eligible": False, "flags": ["no_face"],
             "image_width": 512, "image_height": 512, "native_method": "real", "target": 0, "source_format": "PNG"},
            {"image_sample_id": "d", "sample_id": "d:0", "bbox": None,
             "padding_fraction": np.nan, "eligible": False, "flags": ["decoding_failed"],
             "image_width": np.nan, "image_height": np.nan, "native_method": "real", "target": 0, "source_format": "unknown"},
        ])
        result = E.crop_boundary_summary(frame)
        summary = result["overall"].iloc[0]
        self.assertEqual(summary["input_images"], 4)
        self.assertEqual(summary["measured_primary_images"], 2)
        self.assertEqual(summary["unmeasured_primary_images"], 2)
        self.assertEqual(summary["images_padded"], 1)
        self.assertEqual(summary["share_images_padded_all"], 0.25)
        self.assertEqual(summary["share_images_padded_measured"], 0.5)
        self.assertEqual(summary["images_padding_gt05"], 1)
        self.assertEqual(summary["images_padding_gt10"], 0)
        self.assertAlmostEqual(summary["padding_area_median"], 0.05)
        self.assertAlmostEqual(summary["padding_area_p95"], 0.095)
        self.assertEqual(summary["no_face_images"], 1)
        self.assertEqual(summary["decoding_failed_images"], 1)
        self.assertEqual(summary["primary_bad_images"], 2)
        self.assertEqual(summary["images_with_any_bad_face"], 3)
        self.assertEqual(summary["raw_square_images"], 2)
        self.assertAlmostEqual(summary["share_square_inputs_known_dimensions"], 2 / 3)
        self.assertEqual(result["by_native_method"]["input_images"].sum(), 4)
        self.assertEqual(result["by_target"]["input_images"].sum(), 4)
        adaptive = frame.copy()
        adaptive["crop_policy"] = "adaptive"
        adaptive["proposed_padding_fraction"] = adaptive["padding_fraction"]
        adaptive.loc[adaptive["padding_fraction"].notna(), "padding_fraction"] = 0
        adaptive_summary = E.crop_boundary_summary(adaptive)["overall"].iloc[0]
        self.assertEqual(adaptive_summary["images_padded"], 1)
        self.assertEqual(adaptive_summary["actual_images_padded"], 0)
        self.assertEqual(adaptive_summary["actual_share_images_padded_all"], 0)
        figure = E.plot_crop_boundaries(result)
        figure.canvas.draw()
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
