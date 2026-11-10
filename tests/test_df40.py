import io
import hashlib
import json
import zlib
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import zipfile
from unittest.mock import patch

import pandas as pd

from src.preprocessing.df40 import (
    METHODS, _metadata_lookup, _source_fields, archive_specs, inventory_archive,
    safe_image_member, _coalesced_blocks, _extract_selected_block, match_mirror_records,
    adopt_stream_records, acquire_selected, apply_source_group_policy,
)


class SizedBytesIO(io.BytesIO):
    @property
    def size(self):
        return len(self.getvalue())


class DF40InventoryTests(unittest.TestCase):
    def archive(self, names):
        stream = SizedBytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in names:
                archive.writestr(name, b"image")
        stream.seek(0)
        return stream

    def test_only_native_generation_families_are_registered(self):
        self.assertEqual(len(METHODS), 17)
        self.assertEqual(len(archive_specs()), 27)
        self.assertEqual({value[0] for value in METHODS.values()}, {"EFS", "FE"})

    def test_author_test_overrides_duplicate_validation_paths(self):
        path = "deepfakes_detection_datasets/DF40/CollabDiff/fake/0/000000_x_0.png"
        item = {"label": "CollabDiff_Fake", "frames": [path]}
        document = {"DF40_all": {"DF40All_Fake": {
            "val": {"0": item}, "test": {"0": item}, "train": {},
        }}}
        lookup = _metadata_lookup(document)
        self.assertEqual(len(lookup), 1)
        self.assertEqual(next(iter(lookup.values()))["author_split"], "test")
        document["DF40_all"]["DF40All_Real"] = {
            "train": {"071": {"label": "EFSAll_Real", "frames": [
                "deepfakes_detection_datasets/FaceForensics++/original_sequences/youtube/c23/frames/071/000.png"
            ]}}
        }
        self.assertEqual(len(_metadata_lookup(document)), 1)

    def test_inventory_uses_actual_members_and_native_real_fake_labels(self):
        spec = next(row for row in archive_specs() if row["generator"] == "whichfaceisreal")
        stream = self.archive([
            "whichfaceisreal/real/00414.jpeg", "whichfaceisreal/fake/seed12.png",
            "__MACOSX/whichfaceisreal/._image.jpg", "whichfaceisreal/.DS_Store",
        ])
        with patch("src.preprocessing.df40.HTTPRangeFile", return_value=stream):
            rows = inventory_archive(spec)
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["target"] for row in rows}, {0, 1})
        real = next(row for row in rows if row["target"] == 0)
        self.assertEqual(real["source_group"], "df40:native_real:whichfaceisreal:ffhq:test:real/00414.jpeg")
        self.assertEqual(real["manipulation"], "real")
        self.assertEqual(real["source_format"], "JPEG")
        self.assertFalse(real["source_format_verified"])

    def test_training_directory_infers_ff_domain_and_individual_output(self):
        spec = next(row for row in archive_specs()
                    if row["generator"] == "StyleGAN2" and row["archive_partition"] == "train")
        stream = self.archive(["StyleGAN2/071/seed1.png"])
        with patch("src.preprocessing.df40.HTTPRangeFile", return_value=stream):
            row = inventory_archive(spec)[0]
        self.assertEqual(row["domain"], "ff")
        self.assertEqual(row["source_group"], "df40:efs:StyleGAN2:ff:train:image:071/seed1.png")
        self.assertEqual(row["generation_batch"], "df40:StyleGAN2:ff:train:batch:071")
        self.assertEqual(row["identity_group"], "")
        self.assertEqual(row["source_group_basis"], "generated_image_unknown_identity")

    def test_edit_identifiers_do_not_invent_celeba_identity(self):
        source, identity, basis = _source_fields(
            "CollabDiff", "celeba", "fake/446/000000_x_0.png", 1
        )
        self.assertEqual(source, "df40:fe:CollabDiff:celeba:unknown:image:fake/446/000000_x_0.png")
        self.assertEqual(identity, "")
        self.assertEqual(basis, "edited_image_unknown_identity")

    def test_selected_zip_ranges_decode_and_verify_crc(self):
        spec = next(row for row in archive_specs() if row["generator"] == "whichfaceisreal")
        stream = self.archive([
            "whichfaceisreal/real/00414.jpeg", "whichfaceisreal/fake/seed12.png",
        ])
        archive_bytes = stream.getvalue()
        with patch("src.preprocessing.df40.HTTPRangeFile", return_value=stream):
            rows = inventory_archive(spec)
        reader = SimpleNamespace(
            _request=lambda first, last: (archive_bytes[first:last + 1], "")
        )
        with tempfile.TemporaryDirectory() as temporary:
            for index, row in enumerate(rows):
                row["path"] = str(Path(temporary) / f"{index}.png")
                row["relative_path"] = f"{index}.png"
            results = []
            for block in _coalesced_blocks(rows):
                results.extend(_extract_selected_block(reader, block))
            self.assertEqual(len(results), 2)
            self.assertTrue(all(Path(row["path"]).read_bytes() == b"image" for row in rows))
            rows[0]["zip_crc32"] = "00000000"
            with self.assertRaises(RuntimeError):
                _extract_selected_block(reader, _coalesced_blocks(rows[:1])[0])

    def test_mirror_requires_primary_path_crc_and_length(self):
        primary = [{
            "native_method": "VQGAN", "archive_partition": "test",
            "archive_member": "VQGAN/ff/071/seed1.png",
            "zip_crc32": "12345678", "uncompressed_bytes": 123,
            "target": 1, "author_split": "test",
        }]
        mirror = [{
            "archive_member": "repacked/VQGAN/ff/071/seed1.png",
            "zip_crc32": "12345678", "uncompressed_bytes": 123,
        }]
        self.assertEqual(len(match_mirror_records(primary, mirror)), 1)
        mirror[0]["uncompressed_bytes"] = 124
        with self.assertRaises(RuntimeError):
            match_mirror_records(primary, mirror)
        mirror[0]["uncompressed_bytes"] = 123
        mirror[0]["zip_crc32"] = "00000000"
        with self.assertRaises(RuntimeError):
            match_mirror_records(primary, mirror)

    def test_external_stream_bytes_are_independently_verified(self):
        image = b"native image"
        crc = f"{zlib.crc32(image) & 0xffffffff:08x}"
        row = {"sample_id": "sample", "archive_partition": "test",
               "archive_member": "e4e/image.jpg", "zip_crc32": crc,
               "uncompressed_bytes": len(image)}
        relative = "images/test/e4e/image.jpg"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / relative
            destination.parent.mkdir(parents=True)
            destination.write_bytes(image)
            (root / "metadata").mkdir()
            checkpoint = {"archive": {"download_source": {"kind": "hf_stream_crc_verified"},
                                     "files": {"sample": {
                                         "relative_path": relative, "zip_crc32": crc,
                                         "bytes": len(image), "sha256": hashlib.sha256(image).hexdigest()
                                     }}}}
            (root / "metadata/stream_downloads.json").write_text(json.dumps(checkpoint))
            files = {}
            adopt_stream_records(root, "archive", pd.DataFrame([row]), files)
            self.assertIn("sample", files)
            destination.write_bytes(b"corrupted")
            with self.assertRaises(RuntimeError):
                adopt_stream_records(root, "archive", pd.DataFrame([row]), {})

    def test_missing_source_cannot_create_complete_manifest(self):
        spec = next(row for row in archive_specs() if row["generator"] == "whichfaceisreal")
        stream = self.archive(["whichfaceisreal/fake/seed12.png"])
        with patch("src.preprocessing.df40.HTTPRangeFile", return_value=stream):
            row = inventory_archive(spec)[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("src.preprocessing.df40.resolve_download_source",
                       side_effect=RuntimeError("source unavailable")):
                with self.assertRaises(RuntimeError):
                    acquire_selected(root, pd.DataFrame([row]), backend="hf-verified")
            self.assertFalse((root / "sample.parquet").exists())
            self.assertTrue((root / "sample.partial.parquet").exists())
            status = json.loads((root / "metadata/acquisition_status.json").read_text())
            self.assertFalse(status["complete"])
            self.assertEqual(status["requested_images"], 1)

    def test_unconditional_batches_do_not_create_input_identity(self):
        first = _source_fields("StyleGAN2", "ff", "ff/000/seed1.png", 1,
                               archive_partition="train")
        same_batch = _source_fields("StyleGAN2", "ff", "ff/000/seed2.png", 1,
                                    archive_partition="train")
        other_release = _source_fields("StyleGAN2", "ff", "ff/000/seed1.png", 1,
                                       archive_partition="test")
        conditional = _source_fields("e4e", "ff", "ff/000/00001.jpg", 1,
                                     archive_partition="test")
        self.assertNotEqual(first[0], same_batch[0])
        self.assertNotEqual(first[0], other_release[0])
        self.assertNotEqual(first[0], conditional[0])
        self.assertEqual(first[1], "")
        self.assertEqual(conditional[0], "global:ff:000")

    def test_real_serial_names_are_not_global_image_identity(self):
        first = _source_fields("whichfaceisreal", "ffhq", "real/00414.jpg", 0,
                               archive_partition="test")
        second = _source_fields("MidJourney", "ffhq", "real/00414.jpg", 0,
                                archive_partition="test")
        self.assertNotEqual(first[0], second[0])
        self.assertFalse(first[0].startswith("global:"))
        self.assertEqual(first[2], "native_real_serial_unknown_identity")

    def test_generic_e4e_inversions_are_individual_unknown_outputs(self):
        first = _source_fields("e4e", "cdf", "cdf/Celeb-real/inversions/0001.jpg", 1,
                               archive_partition="test")
        second = _source_fields("e4e", "cdf", "cdf/Celeb-real/inversions/0002.jpg", 1,
                                archive_partition="test")
        self.assertNotEqual(first[0], second[0])
        self.assertEqual(first[1], "")
        self.assertEqual(first[2], "edited_image_unknown_identity")
        known = _source_fields("e4e", "cdf", "cdf/id13_0012/00001.jpg", 1,
                               archive_partition="test")
        self.assertEqual(known[0], "global:cdf:id13_0012")

    def test_lineage_migration_preserves_pixels_targets_and_explicit_groups(self):
        records = [
            {"dataset": "DF40", "native_method": "StyleGAN2", "domain": "ff",
             "archive_partition": "train", "archive_member": "StyleGAN2/ff/000/seed1.png",
             "sample_id": "first:face-0", "target": 1, "source_group": "old_batch",
             "identity_group": "wrong", "source_group_basis": "synthetic_output_batch_unknown_identity"},
            {"dataset": "DF40", "native_method": "StyleGAN2", "domain": "ff",
             "archive_partition": "train", "archive_member": "StyleGAN2/ff/000/seed2.png",
             "sample_id": "second:face-0", "target": 1, "source_group": "old_batch",
             "identity_group": "", "source_group_basis": "synthetic_output_batch_unknown_identity"},
            {"dataset": "DF40", "native_method": "e4e", "domain": "cdf",
             "archive_partition": "test", "archive_member": "e4e/cdf/Celeb-real/inversions/1.jpg",
             "sample_id": "third:face-0", "target": 1, "source_group": "all_inversions",
             "identity_group": "", "source_group_basis": "published_edit_directory"},
            {"dataset": "DF40", "native_method": "e4e", "domain": "ff",
             "archive_partition": "test", "archive_member": "e4e/ff/071/1.jpg",
             "sample_id": "fourth:face-0", "target": 1, "source_group": "global:ff:071",
             "identity_group": "", "source_group_basis": "published_source_identifier"},
            {"dataset": "DF40", "native_method": "CollabDiff", "domain": "celeba",
             "archive_partition": "test", "archive_member": "CollabDiff/fake/446/1.png",
             "sample_id": "fifth:face-0", "target": 1, "source_group": "paired:446",
             "identity_group": "", "source_group_basis": "published_edit_directory"},
            {"dataset": "mffi", "native_method": None, "domain": "unknown",
             "archive_partition": "test", "archive_member": None,
             "sample_id": "other", "target": 0, "source_group": "other-source",
             "identity_group": "", "source_group_basis": "unknown"},
        ]
        frame = pd.DataFrame(records)
        frame["crop_sha256"] = ["pixels"] * len(frame)
        frame["eligible"] = True
        result = apply_source_group_policy(frame)
        self.assertNotEqual(result.loc[0, "source_group"], result.loc[1, "source_group"])
        self.assertEqual(result.loc[0, "generation_batch"], result.loc[1, "generation_batch"])
        self.assertEqual(result.loc[0, "identity_group"], "")
        self.assertEqual(result.loc[2, "source_group_basis"], "edited_image_unknown_identity")
        self.assertEqual(result.loc[3, "source_group"], "global:ff:071")
        self.assertEqual(result.loc[4, "source_group_basis"], "edited_image_unknown_identity")
        self.assertEqual(result.loc[4, "lineage_status"], "unknown")
        self.assertEqual(result.loc[5, "source_group"], "other-source")
        mutable = {"source_group", "identity_group", "source_group_basis"}
        columns = [column for column in frame if column not in mutable]
        pd.testing.assert_frame_equal(frame[columns], result[columns])
        repeated = apply_source_group_policy(result)
        pd.testing.assert_frame_equal(result, repeated)

    def test_multiple_faces_from_same_generated_image_remain_grouped(self):
        frame = pd.DataFrame({
            "dataset": ["DF40", "DF40"], "native_method": ["pixart", "pixart"],
            "domain": ["ff", "ff"], "archive_partition": ["test", "test"],
            "image_sample_id": ["image-a", "image-a"],
            "sample_id": ["image-a:face-0", "image-a:face-1"], "target": [1, 1],
        })
        result = apply_source_group_policy(frame)
        self.assertEqual(result.loc[0, "source_group"], result.loc[1, "source_group"])
        self.assertEqual(result.loc[0, "source_group_basis"], "generated_image_unknown_identity")

    def test_generic_stargan_folder_never_combines_distinct_images(self):
        frame = pd.DataFrame({
            "dataset": ["DF40", "DF40"], "native_method": ["stargan", "stargan"],
            "domain": ["celeba", "celeba"], "archive_partition": ["test", "test"],
            "archive_member": ["stargan/fake/fake/0.jpg", "stargan/fake/fake/1.jpg"],
            "sample_id": ["stargan-0", "stargan-1"], "target": [1, 1],
            "source_group": ["df40:stargan:fake/fake"] * 2,
            "source_group_basis": ["published_edit_directory"] * 2,
        })
        result = apply_source_group_policy(frame)
        self.assertNotEqual(result.loc[0, "source_group"], result.loc[1, "source_group"])
        self.assertEqual(result.loc[0, "generation_batch"], result.loc[1, "generation_batch"])
        self.assertEqual(result["source_group_basis"].tolist(), ["edited_image_unknown_identity"] * 2)
        self.assertEqual(result["lineage_status"].tolist(), ["unknown"] * 2)
        first = _source_fields("stargan", "celeba", "fake/fake/0.jpg", 1,
                               archive_partition="test")
        self.assertEqual(first[0], result.loc[0, "source_group"])

    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError):
            safe_image_member("../outside.png")
        with self.assertRaises(ValueError):
            safe_image_member("/outside.png")


if __name__ == "__main__":
    unittest.main()
