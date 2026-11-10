import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.preprocessing.splits import _BALANCE_COLUMNS, _BALANCE_LIST_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("split_export_cli", ROOT / "scripts/build_composite_splits.py")
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def fixture():
    rows = []
    for dataset in ("source_a", "source_b"):
        for group in range(40):
            for face in range(2):
                identifier = f"{dataset}-{group}-{face}"
                quality = (95, 100, None)[group % 3]
                source_format = "PNG" if group % 5 == 0 else "JPEG"
                quality = quality if source_format == "JPEG" else None
                level = str(quality) if quality is not None else "unknown" if source_format == "JPEG" else "not_jpeg"
                rows.append({
                    "dataset": dataset, "sample_id": identifier, "image_sample_id": f"{dataset}-image-{group}",
                    "target": (group + face) % 2, "source_group": f"{dataset}-group-{group}",
                    "identity_group": "", "author_split": "test",
                    "crop_path": f'/crop/{identifier}\t"quoted"\n.png',
                    "input_path": f'/source/{identifier}\t"quoted"\n.jpg',
                    "generator": "New\tGenerator\nVersion1" if face else "Real",
                    "manipulation": "EFS" if face else "real",
                    "transformation_tags": ["Smile", 'custom"tag\tline\n'] if group % 2 else [],
                    "native_method": "Publisher\nMethod", "native_family": "EFS", "domain": "native",
                    "source_format": source_format, "source_jpeg_quality": quality,
                    "source_jpeg_quality_level": level, "source_jpeg_quality_bin": level,
                    "source_jpeg_subsampling": "4:2:0" if source_format == "JPEG" else "not_jpeg",
                    "source_jpeg_quality_status": "exact" if quality is not None else level,
                    "yaw": 7.5 + group / 10, "pitch": -4.25, "roll": 1.125,
                    "lightness_lstar": 47.5, "face_mask_fraction_of_crop": 0.6,
                    "image_width": 256, "image_height": 256, "bbox": [5.0, 5.0, 64.0, 64.0],
                    "crop_bounds": [0, 0, 84, 84], "padding_fraction": 0.0,
                    "eligible": True, "flags": [], "training_crop_removed": True,
                })
    rows.append({
        "dataset": "source_a", "sample_id": "decode-error", "image_sample_id": "decode-error",
        "target": None, "eligible": False, "flags": ["decoding_failed"], "training_crop_removed": True,
    })
    return pd.DataFrame(rows)


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or value is pd.NA or isinstance(value, float) and np.isnan(value):
        return None
    return value


class TsvExportTests(unittest.TestCase):
    def test_cli_tsv_roundtrips_all_balancing_features_and_compact_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest = directory / "metadata.parquet"
            fixture().to_parquet(manifest, index=False)
            output = directory / "splits"
            process = subprocess.run(
                [sys.executable, str(ROOT / "scripts/build_composite_splits.py"),
                 "--manifest", str(manifest), "--output", str(output)],
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            audit = json.loads((output / "audit.json").read_text())
            self.assertEqual(audit["input_manifest_sha256"],
                             {str(manifest.resolve()): hashlib.sha256(manifest.read_bytes()).hexdigest()})
            table = pd.read_csv(output / "splits.tsv", sep="\t", keep_default_na=False)
            parquet = pd.concat(
                [pd.read_parquet(output / f"{name}.parquet") for name in ("train", "validate", "test")],
                ignore_index=True,
            )
            self.assertEqual(len(table), 160)
            self.assertEqual(len(table), len(parquet))
            self.assertFalse(table.duplicated(["dataset", "sample_id"]).any())
            self.assertEqual(set(table["target"]), {0, 1})
            self.assertEqual(set(table["partition"]), {"train", "validate", "test"})
            self.assertEqual(table.groupby("group_id")["partition"].nunique().max(), 1)
            self.assertFalse(any(name.startswith("Unnamed:") for name in table))
            balanced = [name for name in (*_BALANCE_COLUMNS, *_BALANCE_LIST_COLUMNS) if name in parquet]
            self.assertTrue(set(balanced) <= set(table))
            self.assertEqual(audit["tsv_exports"]["splits"]["balance_feature_columns"], balanced)
            self.assertEqual(audit["tsv_exports"]["splits"]["columns"], list(table))
            self.assertEqual(audit["tsv_exports"]["splits"]["filename"], "splits.tsv")
            self.assertEqual(audit["tsv_exports"]["splits"]["rows"], len(parquet))
            self.assertTrue(audit["ready_for_split_analysis"])
            self.assertFalse(audit["ready_for_training"])
            self.assertEqual(audit["training_crops_removed_for_audit"], 160)
            self.assertTrue({"95", "100", "unknown"} <= set(table["source_jpeg_quality_level"]))
            self.assertTrue({"95", "100", "unknown"} <= set(table["source_jpeg_quality_bin"]))
            left = table.set_index(["dataset", "sample_id"]).sort_index()
            right = parquet.set_index(["dataset", "sample_id"]).sort_index()
            for column in _BALANCE_COLUMNS:
                if column not in right:
                    continue
                if column == "target":
                    self.assertEqual(left[column].astype(int).tolist(), right[column].astype(int).tolist())
                else:
                    self.assertEqual(left[column].map(lambda value: str(plain(value) or "")).tolist(),
                                     right[column].map(lambda value: str(plain(value) or "")).tolist())
            for column in ("partition", "crop_path", "input_path", "generator_raw", "native_method",
                           "source_jpeg_quality_level", "source_jpeg_quality_bin", "source_jpeg_subsampling"):
                self.assertEqual(left[column].tolist(), right[column].tolist())
            for column in ("yaw", "pitch", "roll", "lightness_lstar", "face_mask_fraction_of_crop",
                           "image_width", "image_height"):
                np.testing.assert_allclose(left[column].to_numpy(float), right[column].to_numpy(float))
            for column in audit["tsv_exports"]["splits"]["json_columns"]:
                self.assertEqual(left[column].map(json.loads).tolist(), right[column].map(plain).tolist())
            for name in ("train", "validate", "test", "excluded"):
                self.assertTrue((output / f"{name}.csv").is_file())
                self.assertTrue((output / f"{name}.parquet").is_file())
            excluded = pd.read_csv(output / "excluded.tsv", sep="\t", keep_default_na=False)
            self.assertEqual(excluded["sample_id"].tolist(), ["decode-error"])
            self.assertEqual(excluded["partition"].tolist(), ["excluded"])
            self.assertEqual(excluded["target"].tolist(), [""])
            self.assertEqual(json.loads(excluded.iloc[0]["flags"]), ["decoding_failed", "face_mask_missing"])
            self.assertEqual(audit["tsv_exports"]["excluded"]["rows"], 1)
            self.assertEqual(table["partition"].value_counts().to_dict(),
                             {name: audit[name + "_rows"] for name in ("train", "validate", "test")})

    def test_nested_missing_and_nonfinite_values_are_strict_json_null(self):
        nested = {
            "array": np.array([np.nan, np.inf, -np.inf, 1.5], dtype=np.float32),
            "items": (np.int64(7), np.bool_(True), pd.NA, None),
            "deeper": {"values": [np.float64(np.nan), {"missing": pd.NA, "finite": np.float32(2.5)}]},
            "set": {np.int64(2), np.int64(1)},
            "numpy_key": {np.int64(3): np.float64(np.inf)},
        }
        frame = pd.DataFrame([
            {"dataset": "source", "sample_id": "invalid-box", "target": 0,
             "bbox": nested, "transformation_tags": [np.str_('quoted"tag\tline\n'), pd.NA]},
            {"dataset": "source", "sample_id": "missing-box", "target": 1,
             "bbox": None, "transformation_tags": None},
        ])

        def reject_constant(value):
            raise ValueError("Nonstandard JSON constant: " + value)

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            audit = CLI._write_tsv(frame, directory, "strict.tsv")
            table = pd.read_csv(directory / "strict.tsv", sep="\t", keep_default_na=False)
            self.assertIn("bbox", audit["json_columns"])
            for column in ("bbox", "transformation_tags"):
                for value in table[column]:
                    json.loads(value, parse_constant=reject_constant)
            decoded = json.loads(table.iloc[0]["bbox"], parse_constant=reject_constant)
            self.assertEqual(decoded["array"], [None, None, None, 1.5])
            self.assertEqual(decoded["items"], [7, True, None, None])
            self.assertEqual(decoded["deeper"], {"values": [None, {"missing": None, "finite": 2.5}]})
            self.assertEqual(decoded["set"], [1, 2])
            self.assertEqual(decoded["numpy_key"], {"3": None})
            self.assertIsNone(json.loads(table.iloc[1]["bbox"], parse_constant=reject_constant))
            self.assertIsNone(json.loads(table.iloc[1]["transformation_tags"], parse_constant=reject_constant))
            tags = json.loads(table.iloc[0]["transformation_tags"], parse_constant=reject_constant)
            self.assertEqual(tags, ['quoted"tag\tline\n', None])

    def test_tsv_writer_rejects_duplicate_kept_ids_and_nonbinary_targets(self):
        columns = ["dataset", "sample_id", "target"]
        empty = pd.DataFrame(columns=columns)
        row = pd.DataFrame([{"dataset": "a", "sample_id": "same", "target": 0}])
        with tempfile.TemporaryDirectory() as directory:
            result = {"train": row, "validate": row.copy(), "test": empty, "excluded": empty}
            with self.assertRaisesRegex(ValueError, "identifiers repeat"):
                CLI._write_tsv_exports(result, Path(directory))
            result["validate"] = empty
            result["train"] = row.assign(target=2)
            with self.assertRaisesRegex(ValueError, "binary targets"):
                CLI._write_tsv_exports(result, Path(directory))

    def test_empty_outputs_still_have_stable_tsv_headers(self):
        frame = pd.DataFrame(columns=["dataset", "sample_id", "target", "transformation_tags"])
        result = {name: frame.copy() for name in ("train", "validate", "test", "excluded")}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            audit = CLI._write_tsv_exports(result, directory)
            table = pd.read_csv(directory / "splits.tsv", sep="\t")
            self.assertTrue(table.empty)
            self.assertEqual(list(table)[:10], list(CLI._PRIMARY_TSV_COLUMNS))
            self.assertIn("transformation_tags", audit["splits"]["json_columns"])
            self.assertEqual(audit["splits"]["rows"], 0)


if __name__ == "__main__":
    unittest.main()
