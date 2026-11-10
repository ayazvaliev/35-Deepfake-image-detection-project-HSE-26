import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import preprocess


class PreprocessFingerprintTests(unittest.TestCase):
    def test_source_crop_dependency_change_invalidates_inference_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "src/preprocessing"
            sources.mkdir(parents=True)
            for name in ("faces.py", "jpeg.py", "verification_views.py", "quality.py"):
                (sources / name).write_text("unchanged dependency")
            dependency = sources / "source_crops.py"
            dependency.write_bytes(b"source crop implementation version one")
            manifest = root / "images.jsonl"
            manifest.write_text("")
            models = []
            for name in ("yunet", "landmarker"):
                model = root / name
                model.write_bytes(name.encode())
                models.append(model)
            args = argparse.Namespace(
                quality_only=False, manifest=manifest, workers=1,
                output_dir=root / "processed", data_root=root, max_records=None,
                yunet_model=models[0], landmarker_model=models[1], size=256,
                expansion=1.3, crop_policy="adaptive", min_face_size=32,
                detection_threshold=0.9, compact_verification_crops=True,
                resume=True, min_mask_fraction=0.25,
            )
            with patch.object(preprocess, "ROOT", root), redirect_stdout(io.StringIO()):
                self.assertEqual(preprocess.run(args), 0)
                audit_path = args.output_dir / "preprocessing_audit.json"
                first = json.loads(audit_path.read_text())
                dependency.write_bytes(b"source crop implementation version two")
                self.assertEqual(preprocess.run(args), 0)
                second = json.loads(audit_path.read_text())
            self.assertEqual(first["computed_images"], 0)
            self.assertEqual(second["computed_images"], 0)
            self.assertNotEqual(first["fingerprint"], second["fingerprint"])
            key = "source_crops_source_sha256"
            self.assertNotEqual(first["provenance"][key], second["provenance"][key])
            self.assertEqual(second["provenance"][key], hashlib.sha256(dependency.read_bytes()).hexdigest())
            self.assertEqual(first["provenance"]["model_sha256"], second["provenance"]["model_sha256"])
            self.assertEqual(first["provenance"]["settings"], second["provenance"]["settings"])


if __name__ == "__main__":
    unittest.main()
