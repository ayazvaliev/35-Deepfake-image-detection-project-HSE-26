from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.splits import _BALANCE_COLUMNS, _BALANCE_LIST_COLUMNS, build_composite_splits


def _default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def _json_native(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.ndarray):
        return _json_native(value.tolist())
    if isinstance(value, np.floating):
        return _json_native(float(value))
    if isinstance(value, np.generic):
        native = value.item()
        if isinstance(native, np.generic):
            raise TypeError(f"Cannot convert numpy scalar {value.dtype} to JSON")
        return _json_native(native)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {_json_native(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, set):
        normalized = [_json_native(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, ensure_ascii=False, allow_nan=False))
    if isinstance(value, Path):
        return str(value)
    return value


def _json_dumps(value, indent=None):
    return json.dumps(_json_native(value), ensure_ascii=False, default=_default,
                      allow_nan=False, indent=indent)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(frame, output, partition):
    parquet = output / f"{partition}.parquet"
    temporary = parquet.with_name(parquet.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(parquet)
    csv = frame.copy()
    for column in csv:
        if csv[column].dtype == object:
            csv[column] = csv[column].map(
                lambda value: _json_dumps(value)
                if isinstance(value, (list, tuple, dict, np.ndarray)) else value
            )
    csv.to_csv(output / f"{partition}.csv", index=False)


_PRIMARY_TSV_COLUMNS = (
    "sample_id", "image_sample_id", "dataset", "partition", "target", "crop_path", "input_path",
    "group_id", "source_group", "identity_group",
)
_TSV_METADATA_COLUMNS = (
    "path", "eligible", "flags", "exclusion_flags", "training_crop_removed",
    "native_method", "native_family", "generator_raw", "manipulation_raw", "transformation",
    "transformation_raw", "transformation_tags_raw", "author_split", "author_split_inventory",
    "source_group_basis", "lineage_status", "generation_batch", "source_extension",
    "source_jpeg_quality", "source_jpeg_quality_method", "source_jpeg_quality_status",
    "source_jpeg_qtable_hash", "source_jpeg_progressive",
    "yaw", "pitch", "roll", "lightness_lstar", "face_mask_fraction_of_crop", "mask_source",
    "image_width", "image_height", "face_min_side", "visible_face_min_side", "bbox",
    "crop_bounds", "proposed_crop_bounds", "padding_fraction", "proposed_padding_fraction",
    "crop_policy", "expansion_applied", "crop_truncates_bbox", "face_coverage_fraction",
    "image_sha256", "decoded_sha256", "crop_sha256", "phash",
    "crop_verification_path", "crop_verification_sha256", "crop_verification_pixel_sha256",
    "crop_verification_size", "crop_verification_codec",
)


def _tsv_frame(frame):
    balanced = [name for name in (*_BALANCE_COLUMNS, *_BALANCE_LIST_COLUMNS) if name in frame]
    extra = [name for name in _TSV_METADATA_COLUMNS if name in frame]
    columns = list(dict.fromkeys([*_PRIMARY_TSV_COLUMNS, *balanced, *extra]))
    result = frame.reindex(columns=columns).copy()
    json_columns = []
    for column in result:
        nested = result[column].map(lambda value: isinstance(value, (list, tuple, dict, set, np.ndarray)))
        if nested.any() or column in _BALANCE_LIST_COLUMNS:
            json_columns.append(column)

            result[column] = result[column].map(_json_dumps)
    return result, balanced, json_columns


def _write_tsv(frame, output, filename):
    table, balanced, json_columns = _tsv_frame(frame)
    path = output / filename
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, sep="\t", index=False, encoding="utf-8",
                 quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    temporary.replace(path)
    return {
        "filename": filename, "rows": len(table), "columns": list(table.columns),
        "balance_feature_columns": balanced,
        "balance_list_columns": [name for name in _BALANCE_LIST_COLUMNS if name in table],
        "json_columns": json_columns,
    }


def _write_tsv_exports(result, output):
    kept = pd.concat(
        [result[name].assign(partition=name) for name in ("train", "validate", "test")],
        ignore_index=True,
    )
    if kept.duplicated(["dataset", "sample_id"]).any():
        raise ValueError("Face identifiers repeat across train, validation or test")
    target = pd.to_numeric(kept["target"], errors="raise")
    if not target.isin([0, 1]).all():
        raise ValueError("Training TSV requires binary targets 0/1")
    kept["target"] = target.astype(np.int8)
    excluded = result["excluded"].assign(partition="excluded")
    return {
        "delimiter": "\t", "encoding": "utf-8", "id_columns": ["dataset", "sample_id"],
        "unique_kept_ids": True, "binary_kept_target": True,
        "splits": _write_tsv(kept, output, "splits.tsv"),
        "excluded": _write_tsv(excluded, output, "excluded.tsv"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--group-cache", type=Path)
    parser.add_argument("--group-cache-policy", choices=("strict", "versioned"), default="strict")
    parser.add_argument("--split-policy", choices=("global_balanced", "author_preserved"))
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--test-fraction", type=float)
    parser.add_argument("--author-validation-fraction", type=float)
    parser.add_argument("--near-duplicate-threshold", type=int)
    parser.add_argument("--verification-cache-size", type=int)
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()
    config = {}
    if args.config:
        import yaml

        config = yaml.safe_load(args.config.read_text()) or {}
        config = config.get("splits", config)
    split_policy = args.split_policy or config.get("policy", "global_balanced")
    validation_fraction = args.validation_fraction if args.validation_fraction is not None else config.get("validation_fraction", 0.15)
    test_fraction = args.test_fraction if args.test_fraction is not None else config.get("test_fraction", 0.15)
    author_validation_fraction = args.author_validation_fraction if args.author_validation_fraction is not None else config.get("author_validation_fraction", 0.2)
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    threshold = args.near_duplicate_threshold if args.near_duplicate_threshold is not None else config.get("near_duplicate_threshold", 4)
    verification_cache_size = args.verification_cache_size if args.verification_cache_size is not None else config.get("verification_cache_size", 512)
    input_manifest_sha256 = {str(path.resolve()): _file_sha256(path) for path in args.manifest}
    frames = [pd.read_parquet(path) for path in args.manifest]
    data = pd.concat(frames, ignore_index=True)
    cv2.setNumThreads(1)
    result = build_composite_splits(data, validation_fraction=validation_fraction, seed=seed,
                                    near_duplicate_threshold=threshold,
                                    test_fraction=test_fraction, author_validation_fraction=author_validation_fraction,
                                    split_policy=split_policy, verified_group_cache=args.group_cache,
                                    group_cache_policy=args.group_cache_policy,
                                    verification_cache_size=verification_cache_size)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for partition in ("train", "validate", "test", "excluded"):
        _write_manifest(result[partition], output, partition)
    report = result["report"]
    report["input_manifests"] = [str(path.resolve()) for path in args.manifest]
    report["input_manifest_sha256"] = input_manifest_sha256
    report["tsv_exports"] = _write_tsv_exports(result, output)
    report["ready_for_split_analysis"] = bool(len(result["train"]) and len(result["validate"]) and len(result["test"]))
    removed_crops = sum(
        int(result[part].get("training_crop_removed", pd.Series(False, index=result[part].index)).astype("boolean").fillna(False).sum())
        for part in ("train", "validate", "test")
    )
    report["training_crops_removed_for_audit"] = removed_crops
    report["ready_for_training"] = report["ready_for_split_analysis"] and removed_crops == 0
    (output / "audit.json").write_text(_json_dumps(report, indent=2) + "\n")
    pd.DataFrame(report["balance"]).to_json(output / "balance.json", orient="records", force_ascii=False, indent=2)
    print(json.dumps({key: report[key] for key in ("train_rows", "validate_rows", "test_rows", "excluded_rows", "ready_for_training")}))
    if not report["ready_for_split_analysis"] and not args.allow_empty:
        print("Empty train, validation or test partition. Supply enough independent eligible groups or explicitly use --allow-empty for an audit-only run.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
