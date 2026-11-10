from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.datasets import sample_manifest


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_processed(frame, limit=20_000, seed=42):
    required = {"dataset", "sample_id", "image_sample_id", "target"}
    if not required <= set(frame):
        raise ValueError(f"Missing processed image columns: {sorted(required - set(frame))}")
    if frame["dataset"].nunique() != 1 or frame["image_sample_id"].isna().any():
        raise ValueError("One complete dataset with explicit image identifiers is required")
    if frame["sample_id"].duplicated().any():
        raise ValueError("Face sample_id must be unique")
    images = frame.drop_duplicates("image_sample_id").copy()
    composition = frame.groupby("image_sample_id")["target"].agg(
        lambda values: ";".join(sorted({str(int(value)) for value in values if pd.notna(value)})) or "unknown"
    )
    images["target_composition"] = images["image_sample_id"].map(composition)
    strata = [name for name in ("target_composition", "generator", "manipulation", "domain",
                                "author_split", "source_format", "source_jpeg_quality_level",
                                "source_jpeg_subsampling") if name in images]
    selected = sample_manifest(images, limit, strata, seed)
    if len(selected) != limit:
        raise ValueError(f"Expected {limit:,} input images; only {len(images):,} available")
    result = frame.loc[frame["image_sample_id"].isin(selected["image_sample_id"])].copy()
    if result["image_sample_id"].nunique() != limit:
        raise RuntimeError("Image selection did not preserve every selected input image")
    result = result.sort_values(["image_sample_id", "sample_id"], kind="stable").reset_index(drop=True)
    result.attrs["sampling"] = selected.attrs["sampling"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/processed/demo")
    args = parser.parse_args()
    seen = set()
    for path in args.manifest:
        frame = pd.read_parquet(path)
        selected = sample_processed(frame, args.limit, args.seed)
        dataset = str(selected["dataset"].iloc[0]).casefold()
        if dataset in seen or dataset not in {"df40", "140k", "openforensics", "mffi"}:
            raise ValueError(f"Duplicate or unsupported dataset: {dataset}")
        seen.add(dataset)
        output = args.output_root / dataset
        output.mkdir(parents=True, exist_ok=True)
        temporary = output / "metadata.parquet.tmp"
        selected.to_parquet(temporary, index=False)
        temporary.replace(output / "metadata.parquet")
        audit = {
            "mode": "reuse_full_preprocessing", "source_manifest": str(path.resolve()),
            "source_sha256": file_sha256(path), "seed": args.seed,
            "images": int(selected["image_sample_id"].nunique()), "faces": len(selected),
            "source_images": int(frame["image_sample_id"].nunique()),
            "source_faces": len(frame), "sampling": selected.attrs["sampling"],
            "all_faces_of_selected_images_preserved": True,
            "inference_rerun": False,
        }
        (output / "preprocessing_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"dataset": dataset, "images": audit["images"], "faces": audit["faces"]}))


if __name__ == "__main__":
    main()
