from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.verification_views import (
    read_verification_view, save_crop_verification_view, save_original_verification_view,
    view_pixel_sha256,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value):
    if value is None or value is pd.NA or isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def _path(value):
    path = Path(value).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def compact(metadata_path, delete_crops=False, audit_path=None):
    cv2.setNumThreads(1)
    metadata_path = Path(metadata_path).resolve()
    before = pd.read_parquet(metadata_path)
    frame = before.copy(deep=True)
    old_sha = _sha256(metadata_path)
    backup = metadata_path.with_name(metadata_path.stem + ".precompact." + old_sha[:12] + metadata_path.suffix)
    if not backup.exists():
        shutil.copy2(metadata_path, backup)
    elif _sha256(backup) != old_sha:
        raise ValueError("Existing compact metadata backup has different bytes")
    storage = metadata_path.parent / "verification_compact"
    removals = {}
    original_paths = set()
    crop_paths = set()
    crop_records = 0

    @lru_cache(maxsize=256)
    def checked(path, file_sha, pixel_sha, size, codec):
        return read_verification_view(path, file_sha, pixel_sha, size, codec)

    for index in frame.index:
        row = frame.loc[index].to_dict()
        crop_path = _text(row.get("crop_path"))
        if not crop_path:
            continue
        crop_records += 1
        original = {key: value for key, value in row.items() if key.startswith("original_")}
        if not _text(original.get("original_verification_path")):
            source_path = Path(_text(row.get("input_path")))
            content = source_path.read_bytes()
            if hashlib.sha256(content).hexdigest() != _text(row.get("image_sha256")):
                raise ValueError("Raw input differs from its preprocessing SHA256")
            image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Raw input cannot establish an original verification view")
            original = save_original_verification_view(image, content, storage / "originals")
        checked(original["original_verification_path"], original["original_verification_sha256"],
                original["original_verification_pixel_sha256"], int(original["original_verification_size"]),
                original["original_verification_codec"])
        original_paths.add(original["original_verification_path"])
        png = Path(crop_path)
        if png.is_file():
            if _text(row.get("crop_file_sha256")) and _sha256(png) != _text(row["crop_file_sha256"]):
                raise ValueError("Training PNG differs from its recorded file SHA256")
            crop = cv2.imdecode(np.frombuffer(png.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if crop is None or view_pixel_sha256(crop) != _text(row.get("crop_sha256")):
                raise ValueError("Training PNG differs from its recorded pixel SHA256")
            cached = save_crop_verification_view(crop, storage / "crops", original)
            if delete_crops:
                resolved = png.resolve()
                if resolved.suffix.lower() != ".png" or not resolved.is_relative_to(metadata_path.parent):
                    raise ValueError("Refusing to remove a crop outside this metadata output directory")
                removals[str(resolved)] = _sha256(resolved)
        else:
            cached = {key: value for key, value in row.items() if key.startswith("crop_verification_")}
        crop_pixels = checked(cached["crop_verification_path"], cached["crop_verification_sha256"],
                              cached["crop_verification_pixel_sha256"], int(cached["crop_verification_size"]),
                              cached["crop_verification_codec"])
        if view_pixel_sha256(crop_pixels) != _text(row.get("crop_sha256")):
            raise ValueError("Compact crop differs from the original preprocessing pixels")
        crop_paths.add(cached["crop_verification_path"])
        updates = {**original, **cached, "training_crop_removed": delete_crops or not png.is_file()}
        for column, value in updates.items():
            if column not in frame:
                frame[column] = None
            frame.at[index, column] = value
    changed = []
    for column in before:
        try:
            pd.testing.assert_series_equal(before[column], frame[column], check_dtype=False)
        except AssertionError:
            changed.append(column)
    if set(changed) - {"training_crop_removed"}:
        raise ValueError("Compaction unexpectedly changed existing preprocessing metadata")
    temporary = metadata_path.with_name(metadata_path.name + ".compact.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(metadata_path)
    removed_bytes = 0
    for filename, expected_sha in removals.items():
        path = Path(filename)
        if path.is_file():
            if _sha256(path) != expected_sha:
                raise ValueError("Crop changed after verification; refusing to remove it")
            removed_bytes += path.stat().st_size
            path.unlink()
    new_sha = _sha256(metadata_path)
    evidence = {
        "path": _path(metadata_path), "backup_path": _path(backup),
        "old_sha256": old_sha, "new_sha256": new_sha, "old_columns": list(before),
        "all_old_columns_equal": not changed, "changed_storage_columns": changed,
        "rows": len(frame), "columns_added": [column for column in frame if column not in before],
    }
    audit = {
        "schema_version": 1, "kind": "lossless_verification_extension", "files": [evidence],
        "original_views_verified": True, "crop_views_verified": True,
        "crop_records_verified": crop_records, "unique_original_views": len(original_paths),
        "unique_crop_views": len(crop_paths), "removed_png_crops": len(removals),
        "removed_png_bytes": removed_bytes, "raw_originals_removed": False,
    }
    destination = Path(audit_path) if audit_path is not None else metadata_path.parent / "compact_audit.json"
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(audit, indent=2) + "\n")
    temporary.replace(destination)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--delete-crops", action="store_true")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    print(json.dumps(compact(args.metadata, args.delete_crops, args.audit)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
