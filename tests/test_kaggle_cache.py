import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import download_assets as assets


class KaggleCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.raw = self.directory / "raw"
        self.slug = "publisher/tiny-dataset"
        self.source = self.directory / "source.zip"
        self.payloads = {
            "train/real/one.jpg": b"real source bytes",
            "train/fake/two.jpg": b"fake source bytes",
            "metadata/labels.txt": b"real,0\nfake,1\n",
        }
        with zipfile.ZipFile(self.source, "w") as archive:
            for name, data in self.payloads.items():
                archive.writestr(name, data)
        self.raw_patch = patch.object(assets, "RAW", self.raw)
        self.raw_patch.start()
        self.download_patch = patch.object(assets, "download", side_effect=self.copy_download)
        self.download = self.download_patch.start()
        self.root = self.raw / "140k"
        self.marker = self.root / ".download_complete.json"

    def tearDown(self):
        self.download_patch.stop()
        self.raw_patch.stop()
        self.temporary.cleanup()

    def copy_download(self, url, destination, *args, **kwargs):
        self.assertEqual(url, "https://www.kaggle.com/api/v1/datasets/download/" + self.slug)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.source, destination)
        return destination

    def run_dataset(self):
        assets.kaggle_dataset(self.slug, "140k")

    def test_complete_marker_skips_only_while_every_saved_file_is_intact(self):
        self.run_dataset()
        self.assertEqual(self.download.call_count, 1)
        marker = json.loads(self.marker.read_text())
        self.assertEqual(marker["source"], self.slug)
        self.assertEqual(marker["archive_sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual({row["path"] for row in marker["files"]}, set(self.payloads))
        self.assertFalse(self.marker.with_name(self.marker.name + ".tmp").exists())
        self.run_dataset()
        self.assertEqual(self.download.call_count, 1)
        image = self.root / "train/real/one.jpg"
        image.unlink()
        self.run_dataset()
        self.assertEqual(self.download.call_count, 2)
        self.assertEqual(image.read_bytes(), self.payloads["train/real/one.jpg"])
        original = image.read_bytes()
        image.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        self.assertEqual(image.stat().st_size, len(original))
        self.run_dataset()
        self.assertEqual(self.download.call_count, 3)
        self.assertEqual(image.read_bytes(), original)
        self.assertFalse((self.raw / ".archives/140k.zip").exists())
        marker = json.loads(self.marker.read_text())
        for row in marker["files"]:
            path = self.root / row["path"]
            self.assertEqual(row["bytes"], path.stat().st_size)
            self.assertEqual(row["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_unsafe_saved_path_is_rejected_even_after_an_earlier_missing_file(self):
        self.run_dataset()
        outside = self.directory / "outside.txt"
        outside.write_bytes(b"do not change")
        marker = json.loads(self.marker.read_text())
        (self.root / marker["files"][0]["path"]).unlink()
        marker["files"].append({
            "path": "../../outside.txt", "bytes": outside.stat().st_size,
            "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
        })
        self.marker.write_text(json.dumps(marker))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            self.run_dataset()
        self.assertEqual(self.download.call_count, 1)
        self.assertEqual(outside.read_bytes(), b"do not change")
        self.assertTrue(self.marker.is_file())

    def test_unsafe_zip_member_cannot_overwrite_or_remove_outside_file(self):
        outside = self.directory / "outside.txt"
        outside.write_bytes(b"outside unchanged")
        with zipfile.ZipFile(self.source, "w") as archive:
            archive.writestr("../../outside.txt", b"replacement")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            self.run_dataset()
        self.assertEqual(outside.read_bytes(), b"outside unchanged")
        self.assertFalse(self.marker.exists())

    def test_marker_for_another_source_is_not_treated_as_the_requested_dataset(self):
        self.run_dataset()
        marker = json.loads(self.marker.read_text())
        marker["source"] = "another/dataset"
        self.marker.write_text(json.dumps(marker))
        with self.assertRaisesRegex(ValueError, "different dataset source"):
            self.run_dataset()
        self.assertEqual(self.download.call_count, 1)

    def test_marker_symlink_does_not_modify_an_outside_file(self):
        self.root.mkdir(parents=True)
        outside = self.directory / "outside.json"
        outside.write_text('{"keep": true}')
        self.marker.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            self.run_dataset()
        self.assertEqual(outside.read_text(), '{"keep": true}')
        self.download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
