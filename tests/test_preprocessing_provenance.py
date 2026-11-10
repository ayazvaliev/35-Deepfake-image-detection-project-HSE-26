import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from src.notebook_src import df40_eda as E


def audit():
    return {"provenance": {
        "settings": {"size": 256, "expansion": 1.3, "crop_policy": "adaptive",
                     "min_face_size": 32, "detection_threshold": 0.9,
                     "compact_verification_crops": True},
        "model_sha256": {"yunet_model": "a" * 64, "landmarker_model": "b" * 64},
        "preprocessor_source_sha256": "c" * 64, "runner_source_sha256": "d" * 64,
        "verification_views_source_sha256": "e" * 64, "jpeg_source_sha256": "f" * 64,
    }}


class PreprocessingProvenanceTests(unittest.TestCase):
    def test_storage_code_can_differ_while_weights_and_measurement_settings_match(self):
        inputs = {dataset: audit() for dataset in E.DATASETS}
        inputs["140k"]["provenance"]["preprocessor_source_sha256"] = "1" * 64
        inputs["140k"]["provenance"]["runner_source_sha256"] = "2" * 64
        inputs["140k"]["provenance"]["verification_views_source_sha256"] = "3" * 64
        table = E.validate_preprocessing_provenance(inputs)
        self.assertEqual(len(table), 4)
        self.assertEqual(table["preprocessor_source_sha256"].nunique(), 2)
        self.assertTrue(table["size"].eq(256).all())
        self.assertTrue(table["model_sha256:yunet_model"].eq("a" * 64).all())

    def test_changed_weights_parameters_or_missing_canonical_provenance_fail(self):
        inputs = {dataset: audit() for dataset in E.DATASETS}
        changed = copy.deepcopy(inputs)
        changed["mffi"]["provenance"]["model_sha256"]["yunet_model"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "weights differ"):
            E.validate_preprocessing_provenance(changed)
        for key, value in (("size", 224), ("expansion", 1.0), ("crop_policy", "padded"),
                           ("min_face_size", 16), ("detection_threshold", 0.8)):
            with self.subTest(key=key):
                changed = copy.deepcopy(inputs)
                changed["df40"]["provenance"]["settings"][key] = value
                with self.assertRaisesRegex(ValueError, "preprocessing setting"):
                    E.validate_preprocessing_provenance(changed)
        changed = copy.deepcopy(inputs)
        changed["openforensics"] = {"parts": []}
        with self.assertRaisesRegex(ValueError, "preprocessing setting"):
            E.validate_preprocessing_provenance(changed)
        changed = copy.deepcopy(inputs)
        changed["openforensics"]["provenance"]["model_sha256"] = {}
        with self.assertRaisesRegex(ValueError, "model_sha256"):
            E.validate_preprocessing_provenance(changed)

    def test_storage_audit_is_bound_to_actual_metadata_and_true_equality_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "data/processed/full/140k"
            output.mkdir(parents=True)
            metadata = output / "metadata.parquet"
            pd.DataFrame({"sample_id": ["a", "b", "c"]}).to_parquet(metadata, index=False)
            proof = {
                "schema_version": 1, "kind": "lossless_source_crop_extension",
                "converted_crop_records": 3, "all_crop_pixels_equal": True,
                "files": [{"path": str(metadata.relative_to(root)), "new_sha256": E._sha256(metadata),
                           "all_nonstorage_columns_equal": True}],
            }
            path = output / "source_crop_audit.json"
            path.write_text(json.dumps(proof))
            table = E.storage_conversion_audit(root)
            self.assertEqual(table.iloc[0]["converted_crop_records"], 3)
            self.assertEqual(table.iloc[0]["dataset"], "140k")
            proof["files"][0]["all_nonstorage_columns_equal"] = False
            path.write_text(json.dumps(proof))
            with self.assertRaisesRegex(ValueError, "nonstorage measurements"):
                E.storage_conversion_audit(root)
            proof["files"][0]["all_nonstorage_columns_equal"] = True
            proof["files"][0]["new_sha256"] = "0" * 64
            path.write_text(json.dumps(proof))
            with self.assertRaisesRegex(ValueError, "current metadata"):
                E.storage_conversion_audit(root)


if __name__ == "__main__":
    unittest.main()
