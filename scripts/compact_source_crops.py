from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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

from src.preprocessing.source_crops import read_source_crop, save_source_crop
from src.preprocessing.verification_views import read_verification_view, view_pixel_sha256


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _convert(index, row, directory):
    if row.get("crop_verification_codec") == "source_crop" or row.get("crop_verification_aliases_original"):
        return index, None
    filename = row.get("crop_verification_path")
    if not isinstance(filename, str) or row.get("original_verification_codec") != "source_bytes":
        return index, None
    crop = read_verification_view(filename, row["crop_verification_sha256"],
                                  row["crop_verification_pixel_sha256"], row["crop_verification_size"],
                                  row["crop_verification_codec"])
    if view_pixel_sha256(crop) != row["crop_sha256"]:
        raise ValueError("Existing verification crop differs from preprocessing pixels")
    metadata = save_source_crop(crop, row["crop_bounds"], row, directory)
    reconstructed = read_source_crop(metadata["crop_verification_path"], metadata["crop_verification_sha256"],
                                     metadata["crop_verification_pixel_sha256"], metadata["crop_verification_size"])
    if not np.array_equal(reconstructed, crop):
        raise ValueError("Source crop differs from the retained lossless crop")
    return index, metadata


def compact_source_crops(metadata_path, delete_old_caches=False, workers=8, audit_path=None):
    if workers < 1:
        raise ValueError("workers must be positive")
    cv2.setNumThreads(1)
    metadata_path = Path(metadata_path).resolve()
    before = pd.read_parquet(metadata_path)
    frame = before.copy(deep=True)
    old_sha = _sha256(metadata_path)
    backup = metadata_path.with_name(metadata_path.stem + ".presourcecrop." + old_sha[:12] + metadata_path.suffix)
    if not backup.exists():
        shutil.copy2(metadata_path, backup)
    elif _sha256(backup) != old_sha:
        raise ValueError("Source crop metadata backup differs from the input")
    directory = metadata_path.parent / "verification_source_crops"
    converted = 0
    old_paths = {}

    def accept(result):
        nonlocal converted
        index, updates = result
        if updates is None:
            return
        old_paths[str(frame.at[index, "crop_verification_path"])] = str(frame.at[index, "crop_verification_sha256"])
        for column, value in updates.items():
            if column not in frame:
                frame[column] = None
            frame.at[index, column] = value
        converted += 1
        if converted % 1000 == 0:
            print(f"Verified {converted:,} source crops", flush=True)

    columns = list(before)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = set()
        for values in before.itertuples(index=True, name=None):
            index, *items = values
            pending.add(executor.submit(_convert, index, dict(zip(columns, items)), directory))
            if len(pending) >= workers * 3:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    accept(future.result())
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                accept(future.result())
    changed = []
    for column in before:
        try:
            pd.testing.assert_series_equal(before[column], frame[column], check_dtype=False)
        except AssertionError:
            changed.append(column)
    if any(not column.startswith("crop_verification_") for column in changed):
        raise ValueError("Source crop compaction changed non-storage metadata")
    temporary = metadata_path.with_name(metadata_path.name + ".sourcecrop.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(metadata_path)
    retained = set(frame["crop_verification_path"].dropna().astype(str)) | set(frame["original_verification_path"].dropna().astype(str))
    removed_bytes = 0
    removed_files = 0
    if delete_old_caches:
        for filename, expected_sha in old_paths.items():
            path = Path(filename).resolve()
            if str(path) in retained or not path.is_file():
                continue
            if not path.is_relative_to(metadata_path.parent) or path.suffix.lower() not in {".webp", ".png"}:
                raise ValueError("Refusing to remove a verification file outside this output directory")
            if _sha256(path) != expected_sha:
                raise ValueError("Old lossless crop changed; refusing to remove it")
            removed_bytes += path.stat().st_size
            removed_files += 1
            path.unlink()
    audit = {
        "schema_version": 1, "kind": "lossless_source_crop_extension",
        "files": [{
            "path": _path(metadata_path), "backup_path": _path(backup),
            "old_sha256": old_sha, "new_sha256": _sha256(metadata_path), "old_columns": list(before),
            "all_old_columns_equal": not changed, "all_nonstorage_columns_equal": True,
            "changed_storage_columns": changed, "rows": len(frame),
            "columns_added": [column for column in frame if column not in before],
        }],
        "converted_crop_records": converted, "all_crop_pixels_equal": True,
        "removed_old_cache_files": removed_files, "removed_old_cache_bytes": removed_bytes,
        "raw_originals_removed": False,
    }
    destination = Path(audit_path) if audit_path is not None else metadata_path.parent / "source_crop_audit.json"
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(audit, indent=2) + "\n")
    temporary.replace(destination)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--delete-old-caches", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    print(json.dumps(compact_source_crops(args.metadata, args.delete_old_caches, args.workers, args.audit)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
