from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.download_assets import file_hash
from scripts.download_openforensics_parts import verify_views


def release_sources(metadata, raw_root, delete_checkpoint=False):
    metadata, raw_root = Path(metadata).resolve(), Path(raw_root).resolve()
    frame = pd.read_parquet(metadata)
    if frame["dataset"].nunique() != 1 or frame["sample_id"].duplicated().any():
        raise ValueError("A complete single-dataset preprocessing manifest is required")
    verified = verify_views(frame, require_crops=True)
    source_rows = frame[["input_path", "image_sha256"]].drop_duplicates()
    if source_rows["input_path"].duplicated().any():
        raise ValueError("Conflicting original SHA256 for one input path")
    owned = []
    for filename, digest in source_rows.itertuples(index=False, name=None):
        original = Path(filename)
        path = original.resolve()
        if original.is_symlink() or raw_root not in path.parents:
            raise ValueError("Refusing to remove a source outside the selected dataset root")
        if path.exists():
            if file_hash(path) != digest:
                raise ValueError("Original image SHA256 changed after preprocessing")
            owned.append(path)
    removed_bytes = sum(path.stat().st_size for path in owned)
    for path in owned:
        path.unlink()
    checkpoint = metadata.parent / "checkpoint.sqlite"
    if delete_checkpoint:
        checkpoint.unlink(missing_ok=True)
    audit = {
        "metadata": str(metadata), "metadata_sha256": file_hash(metadata), "raw_root": str(raw_root),
        "original_views_verified": True, "crop_views_verified": True,
        "verified_view_counts": verified, "removed_originals": len(owned),
        "removed_original_bytes": removed_bytes, "checkpoint_removed": delete_checkpoint,
        "training_pngs_removed": bool(frame.get("training_crop_removed", pd.Series(False, index=frame.index)).astype("boolean").fillna(False).any()),
    }
    (metadata.parent / "storage_cleanup.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--delete-checkpoint", action="store_true")
    args = parser.parse_args()
    print(json.dumps(release_sources(args.metadata, args.raw_root, args.delete_checkpoint)))


if __name__ == "__main__":
    main()
