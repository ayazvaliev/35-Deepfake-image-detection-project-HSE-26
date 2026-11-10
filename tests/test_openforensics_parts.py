import hashlib
from argparse import Namespace
from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
import threading

import cv2
import numpy as np
import pandas as pd

from scripts.download_openforensics_parts import cleanup_owned, manifest_chunk, merge_parts, verify_views, preprocessing_arguments, preprocessing_settings, reusable_part, needs_raw_restoration, current_part_provenance, ArchivePrefetch, next_archive_to_prefetch, run
from src.preprocessing.verification_views import (
    normalized_original_view, read_verification_view, save_crop_verification_view,
    save_original_verification_view, view_pixel_sha256,
)


class OpenForensicsPartsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw/openforensics"
        (self.raw / "images/Train").mkdir(parents=True)

    def owned(self, names):
        result = []
        for name in names:
            path = self.raw / "images/Train" / name
            path.write_bytes(name.encode())
            result.append({"path": str(path), "sha256": hashlib.sha256(name.encode()).hexdigest()})
        return {"owned_files": result}

    def test_manifest_is_limited_to_annotated_owned_images(self):
        paths = [str(self.raw / "images/Train" / name) for name in ["a.jpg", "b.jpg"]]
        inventory = pd.DataFrame({"path": paths, "sample_id": ["a", "b"], "target": [0, 1]})
        chunk = manifest_chunk(inventory, [paths[1]])
        self.assertEqual(chunk.sample_id.tolist(), ["b"])
        with self.assertRaises(ValueError):
            manifest_chunk(inventory, [str(self.raw / "images/Train/unannotated.jpg")])

    def test_raw_cleanup_requires_verified_completed_views(self):
        ready = self.owned(["a.jpg"])
        path = Path(ready["owned_files"][0]["path"])
        self.assertFalse(cleanup_owned(self.raw, ready, {"metadata_complete": True}))
        self.assertTrue(path.exists())
        self.assertFalse(cleanup_owned(self.raw, ready, {"original_views_verified": True}))
        self.assertTrue(path.exists())
        self.assertTrue(cleanup_owned(self.raw, ready, {
            "metadata_complete": True, "original_views_verified": True,
        }))
        self.assertFalse(path.exists())

    def test_cleanup_checks_all_hashes_before_removing_only_owned_files(self):
        ready = self.owned(["a.jpg", "b.jpg"])
        untouched = self.raw / "images/Train/other.jpg"
        untouched.write_bytes(b"unowned")
        second = Path(ready["owned_files"][1]["path"])
        second.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            cleanup_owned(self.raw, ready, {"metadata_complete": True, "original_views_verified": True})
        self.assertTrue(Path(ready["owned_files"][0]["path"]).exists())
        self.assertTrue(untouched.exists())

    def test_cleanup_rejects_outside_files_and_symlinks(self):
        outside = self.root / "outside.jpg"
        outside.write_bytes(b"outside")
        link = self.raw / "images/Train/link.jpg"
        link.symlink_to(outside)
        ready = {"owned_files": [{"path": str(link), "sha256": hashlib.sha256(b"outside").hexdigest()}]}
        with self.assertRaises(ValueError):
            cleanup_owned(self.raw, ready, {"metadata_complete": True, "original_views_verified": True})
        self.assertTrue(outside.exists())

    def test_exact_proxy_validation_and_corruption(self):
        image = np.random.default_rng(42).integers(0, 256, (64, 128, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        original = save_original_verification_view(decoded, encoded.tobytes(), self.root / "views")
        crop = normalized_original_view(decoded)
        record = {**original, **save_crop_verification_view(crop, self.root / "views", original)}
        record.update(decoded_sha256=view_pixel_sha256(decoded), crop_sha256=view_pixel_sha256(crop))
        frame = pd.DataFrame([record, record])
        with patch("src.preprocessing.verification_views.read_verification_view", wraps=read_verification_view) as reader:
            counts = verify_views(frame, require_crops=True)
        self.assertEqual(counts, {"original": 1, "crop": 1})
        self.assertEqual(reader.call_count, 1)
        path = Path(original["original_verification_path"])
        path.write_bytes(path.read_bytes() + b"corrupt")
        with self.assertRaises(ValueError):
            verify_views(frame, require_crops=True)

    def test_keep_raw_advances_completed_part_without_deleting_sources(self):
        ready = self.owned(["a.jpg"])
        ready["owned_paths"] = [record["path"] for record in ready["owned_files"]]
        ready["image_count"] = 1
        parts = self.root / "data/processed/full/manifests/of_parts"
        parts.mkdir(parents=True)
        (parts / "Train_part_1.ready.json").write_text(json.dumps(ready))
        marker = parts / "Train_part_1.processed.json"
        marker.write_text(json.dumps({"metadata_complete": True, "original_views_verified": True}))
        inventory = pd.DataFrame({
            "path": ready["owned_paths"], "sample_id": ["frame-a"],
            "dataset": ["openforensics"], "target": [0], "author_split": ["train"],
            "n_faces": [1], "frame_composition": ["all_real"],
        })
        files = [{"key": split + "_poly.json", "checksum": "md5:" + "0" * 32, "size": 1}
                 for split in ["Train", "Val", "Test-Dev"]]
        response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"files": files})
        args = Namespace(
            root=self.root, raw_root=self.raw.parent, preprocess_output=None,
            expected_images=1, limit=1, seed=42, parts=["Train_part_1.zip"],
            stage_only=False, workers=1, compact_storage=False, config=None, keep_raw=True,
        )
        with patch("scripts.download_openforensics_parts.requests.get", return_value=response), \
             patch("scripts.download_openforensics_parts.download"), \
             patch("scripts.download_openforensics_parts.index_openforensics", return_value=inventory), \
             patch("scripts.download_openforensics_parts.sample_manifest", return_value=inventory):
            self.assertEqual(run(args), 0)
        self.assertTrue(Path(ready["owned_paths"][0]).is_file())
        self.assertTrue(json.loads(marker.read_text())["raw_retained"])
        self.assertFalse(json.loads(marker.read_text())["raw_deleted"])

    def completed_fixture(self, compact=True):
        output = self.root / "complete"
        folder = output / "parts/Train_part_1"
        folder.mkdir(parents=True)
        crop = folder / "crop.png"
        crop.write_bytes(b"fixture-crop")
        frame = pd.DataFrame({
            "decoded_sha256": ["decoded"], "crop_sha256": ["pixels"],
            "crop_path": [str(crop)],
            "crop_file_sha256": [hashlib.sha256(crop.read_bytes()).hexdigest()],
        })
        metadata = folder / "metadata.parquet"
        frame.to_parquet(metadata, index=False)
        settings = preprocessing_settings(None)
        for relative in [
            "src/preprocessing/faces.py", "scripts/preprocess.py",
            "src/preprocessing/jpeg.py", "src/preprocessing/verification_views.py",
            "models/face_detection_yunet_2023mar.onnx", "models/face_landmarker.task",
        ]:
            artifact = self.root / relative
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(relative.encode())
        manifest = self.root / "part.jsonl"
        manifest.write_text('{"sample_id":"frame-a"}\\n')
        (folder / "preprocessing_audit.json").write_text(json.dumps({
            "provenance": {**current_part_provenance(self.root),
                           "settings": {**settings, "compact_verification_crops": compact}},
            "quality_policy": {"min_mask_fraction": settings["min_mask_fraction"]},
            "manifest_path": str(manifest),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        }))
        state = {
            "metadata_complete": True, "original_views_verified": True,
            "metadata_path": str(metadata), "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
            "compact_storage": compact, "preprocessing_settings": settings,
        }
        return output, state, crop

    def test_processed_part_reuse_checks_storage_and_shared_parameters(self):
        output, state, _ = self.completed_fixture()
        config = self.root / "changed.yml"
        config.write_text("preprocessing:\n  min_mask_fraction: 0.4\n")
        with patch("scripts.download_openforensics_parts.verify_views", return_value={"original": 1, "crop": 1}):
            self.assertTrue(reusable_part(state, output, "Train_part_1", True, None, root=self.root))
            self.assertFalse(reusable_part(state, output, "Train_part_1", False, None, root=self.root))
            self.assertFalse(reusable_part(state, output, "Train_part_1", True, config, root=self.root))
        missing = {"owned_paths": [str(self.root / "deleted-original.jpg")]}
        self.assertTrue(needs_raw_restoration(missing, keep_raw=False, needs_processing=True))
        self.assertFalse(needs_raw_restoration(missing, keep_raw=False, needs_processing=False))

    def test_normal_mode_repairs_missing_training_pngs(self):
        output, state, crop = self.completed_fixture(compact=False)
        with patch("scripts.download_openforensics_parts.verify_views", return_value={"original": 1}):
            self.assertTrue(reusable_part(state, output, "Train_part_1", False, None, root=self.root))
            crop.unlink()
            self.assertFalse(reusable_part(state, output, "Train_part_1", False, None, root=self.root))

    def test_reuse_fails_closed_on_changed_metadata_and_repairs_missing_views(self):
        output, state, _ = self.completed_fixture()
        with patch("scripts.download_openforensics_parts.verify_views", side_effect=FileNotFoundError("missing proxy")):
            self.assertFalse(reusable_part(state, output, "Train_part_1", True, None, root=self.root))
        state["metadata_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "metadata SHA256"):
            reusable_part(state, output, "Train_part_1", True, None, root=self.root)

    def test_reuse_binds_current_models_code_and_manifest(self):
        output, state, _ = self.completed_fixture()
        with patch("scripts.download_openforensics_parts.verify_views", return_value={"original": 1, "crop": 1}):
            self.assertTrue(reusable_part(state, output, "Train_part_1", True, None, root=self.root))
            for relative in [
                "src/preprocessing/faces.py", "scripts/preprocess.py",
                "src/preprocessing/jpeg.py", "src/preprocessing/verification_views.py",
                "models/face_detection_yunet_2023mar.onnx", "models/face_landmarker.task",
                "part.jsonl",
            ]:
                with self.subTest(artifact=relative):
                    artifact = self.root / relative
                    original = artifact.read_bytes()
                    artifact.write_bytes(original + b"changed")
                    self.assertFalse(reusable_part(state, output, "Train_part_1", True, None, root=self.root))
                    artifact.write_bytes(original)

    def test_shared_preprocessing_configuration_is_forwarded(self):
        config = self.root / "config.yml"
        config.write_text("preprocessing:\n  size: 256\n  expansion: 1.3\n  crop_policy: adaptive\n  min_face_size: 32\n  min_mask_fraction: 0.25\n  detection_threshold: 0.9\n")
        arguments = preprocessing_arguments(config)
        self.assertEqual(arguments, [
            "--size", "256", "--expansion", "1.3", "--crop-policy", "adaptive",
            "--min-face-size", "32", "--min-mask-fraction", "0.25",
            "--detection-threshold", "0.9",
        ])
        config.write_text("preprocessing:\n  output_format: JPEG\n")
        with self.assertRaises(ValueError):
            preprocessing_arguments(config)

    def merged_fixture(self):
        parts = self.root / "manifests"
        parts.mkdir()
        output = self.root / "processed"
        settings = preprocessing_settings(None)
        provenance = {
            "settings": settings, "model_sha256": {"yunet_model": "u" * 64, "landmarker_model": "l" * 64},
            "preprocessor_source_sha256": "p" * 64, "runner_source_sha256": "r" * 64,
            "jpeg_source_sha256": "j" * 64, "verification_views_source_sha256": "v" * 64,
        }
        for index, name in enumerate(["Train_part_1", "Val"]):
            folder = output / "parts" / name
            folder.mkdir(parents=True)
            frame = pd.DataFrame({
                "dataset": ["openforensics"], "sample_id": [f"face-{index}"],
                "image_sample_id": [f"image-{index}"], "target": [index],
                "eligible": [True], "face_index": [0], "crop_path": [f"crop-{index}.png"],
            })
            frame.to_parquet(folder / "metadata.parquet", index=False)
            (folder / "preprocessing_audit.json").write_text(json.dumps({
                "images": 1, "computed_images": 1, "reused_images": 0,
                "fingerprint": "same-settings", "provenance": provenance,
                "quality_policy": {
                    "schema_version": 1, "min_mask_fraction": 0.25,
                    "processing_records": 1, "known_fraction_records": 1,
                    "missing_fraction_records": 0, "too_small_fraction_records": 0,
                    "eligible_before": 1, "eligible_after": 1, "newly_excluded_records": 0,
                },
                "quality_provenance": {
                    "application_mode": "post_inference", "policy_source_sha256": "q" * 64,
                    "runner_source_sha256": "r" * 64,
                    "applied_at_utc": f"2026-11-10T00:00:0{index}+00:00",
                    "inference_rerun": True, "inference_fingerprint": "same-settings",
                    "metadata_input_sha256": None,
                },
            }))
            (parts / f"{name}.processed.json").write_text(json.dumps({
                "metadata_complete": True, "metadata_path": str(folder / "metadata.parquet"),
            }))
        return parts, output, provenance

    def test_completed_parts_merge_without_changing_face_records(self):
        parts, output, provenance = self.merged_fixture()
        audit = merge_parts(parts, ["Train_part_1.zip", "Val.zip"], output, 2)
        result = pd.read_parquet(output / "metadata.parquet")
        self.assertEqual(result.target.tolist(), [0, 1])
        self.assertEqual(result.sample_id.tolist(), ["face-0", "face-1"])
        self.assertEqual(audit["images"], 2)
        self.assertEqual(audit["eligible_faces"], 2)
        self.assertEqual(audit["provenance"]["settings"], provenance["settings"])
        self.assertEqual(audit["provenance"]["model_sha256"], provenance["model_sha256"])
        self.assertEqual(audit["quality_policy"]["processing_records"], 2)
        self.assertEqual(audit["quality_policy"]["eligible_after"], 2)
        self.assertEqual(audit["quality_provenance"]["application_mode"], "aggregated_parts")

    def test_merge_rejects_mixed_models_or_policy_before_publishing(self):
        parts, output, _ = self.merged_fixture()
        metadata = output / "metadata.parquet"
        metadata.write_bytes(b"previous-valid-artifact")
        audit_path = output / "parts/Val/preprocessing_audit.json"
        original = json.loads(audit_path.read_text())
        changed = json.loads(json.dumps(original))
        changed["provenance"]["model_sha256"]["yunet_model"] = "different"
        audit_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "settings, model weights"):
            merge_parts(parts, ["Train_part_1.zip", "Val.zip"], output, 2)
        self.assertEqual(metadata.read_bytes(), b"previous-valid-artifact")
        changed = json.loads(json.dumps(original))
        changed["quality_policy"]["min_mask_fraction"] = 0.4
        audit_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "quality policies"):
            merge_parts(parts, ["Train_part_1.zip", "Val.zip"], output, 2)
        self.assertEqual(metadata.read_bytes(), b"previous-valid-artifact")


class OpenForensicsPrefetchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.names = ["Train_part_1.zip", "Train_part_2.zip", "Train_part_3.zip"]
        self.payloads = {name: name.encode() for name in self.names}
        self.catalog = {
            name: {"size": len(content), "checksum": "md5:" + hashlib.md5(content).hexdigest()}
            for name, content in self.payloads.items()
        }

    def test_future_guard_never_duplicates_filename_and_holds_one_lookahead(self):
        release = threading.Event()
        lock = threading.Lock()
        calls = []
        active = 0
        maximum = 0

        def fake_download(url, destination, expected, algorithm, expected_size):
            nonlocal active, maximum
            name = destination.name
            with lock:
                calls.append(name)
                active += 1
                maximum = max(maximum, active)
            if not release.wait(5):
                raise RuntimeError("Test release timeout")
            content = self.payloads[name]
            self.assertEqual(len(content), expected_size)
            self.assertEqual(hashlib.md5(content).hexdigest(), expected)
            destination.write_bytes(content)
            with lock:
                active -= 1
            return destination

        with patch("scripts.download_openforensics_parts.download", side_effect=fake_download), \
             patch("scripts.download_openforensics_parts.shutil.disk_usage", return_value=SimpleNamespace(free=10000)):
            manager = ArchivePrefetch(self.root, self.catalog, reserve_bytes=6)
            try:
                first = manager.current(self.names[0])
                self.assertTrue(manager.prefetch(self.names[1]))
                self.assertFalse(manager.prefetch(self.names[2]))
                with ThreadPoolExecutor(max_workers=8) as callers:
                    repeated = list(callers.map(lambda _: manager.current(self.names[0]), range(30)))
                self.assertTrue(all(item is first for item in repeated))
                self.assertTrue(manager.prefetch(self.names[1]))
                with self.assertRaises(RuntimeError):
                    manager.current(self.names[1])
                release.set()
                first.result()
                manager.current(self.names[1]).result()
                self.assertTrue(manager.prefetch(self.names[2]))
                manager.current(self.names[2]).result()
            finally:
                release.set()
                manager.close()
        self.assertEqual(sorted(calls), sorted(self.names))
        self.assertLessEqual(maximum, 2)

    def test_low_space_defers_lookahead_and_counts_current_remaining(self):
        release = threading.Event()
        archive_size = self.catalog[self.names[0]]["size"]
        free = archive_size + 6

        def fake_download(url, destination, *args):
            release.wait(5)
            destination.write_bytes(self.payloads[destination.name])
            return destination

        with patch("scripts.download_openforensics_parts.download", side_effect=fake_download) as downloader, \
             patch("scripts.download_openforensics_parts.shutil.disk_usage", return_value=SimpleNamespace(free=free)):
            manager = ArchivePrefetch(self.root, self.catalog, reserve_bytes=6)
            try:
                future = manager.current(self.names[0])
                self.assertFalse(manager.prefetch(self.names[1]))
                self.assertEqual(downloader.call_count, 1)
                release.set()
                future.result()
            finally:
                release.set()
                manager.close()

    def test_checksum_failure_is_propagated_without_retrying_or_submitting_more(self):
        with patch("scripts.download_openforensics_parts.download", side_effect=ValueError("MD5 mismatch")) as downloader, \
             patch("scripts.download_openforensics_parts.shutil.disk_usage", return_value=SimpleNamespace(free=10000)):
            manager = ArchivePrefetch(self.root, self.catalog, reserve_bytes=6)
            try:
                future = manager.current(self.names[0])
                with self.assertRaisesRegex(ValueError, "MD5 mismatch"):
                    future.result()
                with self.assertRaisesRegex(ValueError, "MD5 mismatch"):
                    manager.prefetch(self.names[1])
                self.assertEqual(downloader.call_count, 1)
            finally:
                manager.close()

    def test_prefetch_skips_processed_and_already_staged_archives(self):
        parts = self.root / "parts"
        parts.mkdir()
        (parts / "Train_part_2.processed.json").write_text(json.dumps({"metadata_complete": True}))
        (parts / "Train_part_2.ready.json").write_text(json.dumps({
            "owned_paths": [str(self.root / "deleted-raw.jpg")],
        }))
        self.assertEqual(next_archive_to_prefetch(self.names, 0, parts, False), self.names[2])
        self.assertEqual(next_archive_to_prefetch(self.names, 0, parts, True), self.names[1])
        raw = self.root / "staged.jpg"
        raw.write_bytes(b"source")
        (parts / "Train_part_3.ready.json").write_text(json.dumps({"owned_paths": [str(raw)]}))
        self.assertIsNone(next_archive_to_prefetch(self.names, 0, parts, False))

    def test_resume_prefix_reduces_reserved_remaining_bytes(self):
        name = self.names[0]
        size = self.catalog[name]["size"]
        (self.root / (name + ".part")).write_bytes(self.payloads[name][:-2])
        with patch("scripts.download_openforensics_parts.shutil.disk_usage", return_value=SimpleNamespace(free=8)):
            manager = ArchivePrefetch(self.root, self.catalog, reserve_bytes=6)
            try:
                self.assertEqual(manager._remaining(name), 2)
                self.assertTrue(manager._has_space(name))
            finally:
                manager.close()


if __name__ == "__main__":
    unittest.main()
