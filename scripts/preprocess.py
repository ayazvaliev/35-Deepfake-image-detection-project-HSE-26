from __future__ import annotations

import argparse
import atexit
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_PROCESSOR = None
_CROP_DIR = None

from src.preprocessing.quality import apply_quality_policy


def _native(value):
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [_native(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA:
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _json(value):
    return json.dumps(_native(value), sort_keys=True, ensure_ascii=False, allow_nan=False)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_records(path, data_root):
    if path.suffix.lower() in (".parquet", ".pq"):
        records = pd.read_parquet(path).to_dict("records")
    elif path.suffix.lower() in (".jsonl", ".ndjson"):
        with path.open() as stream:
            records = [json.loads(line) for line in stream if line.strip()]
    else:
        raise ValueError("Image manifest must be JSONL or Parquet")
    records = [_native(record) for record in records]
    seen = set()
    for record in records:
        for column in ("dataset", "sample_id"):
            if not record.get(column):
                raise ValueError(f"Each image record requires {column}")
        sample_id = str(record["sample_id"])
        if sample_id in seen:
            raise ValueError(f"Duplicate image sample_id: {sample_id}")
        seen.add(sample_id)
        value = record.get("input_path") or record.get("path")
        if not value:
            raise ValueError(f"Missing input image path: {sample_id}")
        image_path = Path(value)
        if not image_path.is_absolute():
            image_path = data_root / image_path
        record["input_path"] = str(image_path.resolve())
        record["path"] = str(image_path.resolve())
    return records


def _initialize_worker(settings, crop_dir):
    global _PROCESSOR, _CROP_DIR
    import cv2
    from src.preprocessing.faces import YuNetFacePreprocessor

    cv2.setNumThreads(1)
    _PROCESSOR = YuNetFacePreprocessor(**settings)
    _CROP_DIR = crop_dir
    atexit.register(_PROCESSOR.close)


def _process_task(key, record):
    rows = _PROCESSOR.process_record(record, output_dir=_CROP_DIR)
    for index, row in enumerate(rows):
        row["image_sample_id"] = str(record["sample_id"])
        row["sample_id"] = f"{record['sample_id']}:face-{row.get('face_index', index)}"
        if row.get("crop_path"):
            row["crop_file_sha256"] = _sha256(row["crop_path"])
    return key, _native(rows)


def _cached_rows(connection, key):
    stored = connection.execute("SELECT payload FROM results WHERE cache_key = ?", (key,)).fetchone()
    if stored is None:
        return None
    rows = json.loads(stored[0])
    for row in rows:
        path = row.get("crop_path")
        if path:
            if not Path(path).is_file() or _sha256(path) != row.get("crop_file_sha256"):
                return None
    return rows


def _write_parquet(frame, path):
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)



def _write_audit(audit, path):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(_json(audit) + "\n")
    temporary.replace(path)


def _finalize_quality(frame, audit, min_mask_fraction, mode, metadata_input_sha256=None):
    frame, policy = apply_quality_policy(frame, min_mask_fraction=min_mask_fraction)
    flags = Counter(flag for values in frame["flags"] for flag in values)
    face_records = int(frame["face_index"].notna().sum()) if "face_index" in frame else 0
    crop_records = int(frame["crop_path"].fillna("").astype(str).ne("").sum()) if "crop_path" in frame else 0
    audit.update({
        "faces": len(frame),
        "processing_records": len(frame),
        "face_records": face_records,
        "crop_records": crop_records,
        "eligible_faces": int(frame["eligible"].sum()),
        "flags": dict(flags),
        "quality_policy": policy,
        "quality_provenance": {
            "application_mode": mode,
            "applied_at_utc": datetime.now(timezone.utc).isoformat(),
            "policy_source_sha256": _sha256(ROOT / "src/preprocessing/quality.py"),
            "runner_source_sha256": _sha256(Path(__file__)),
            "inference_fingerprint": audit.get("fingerprint"),
            "inference_rerun": mode != "quality_only" and bool(audit.get("computed_images", 0)),
            "metadata_input_sha256": metadata_input_sha256,
        },
    })
    return frame, audit


def run_quality_only(args):
    output = args.output_dir.resolve()
    metadata_path = output / "metadata.parquet"
    audit_path = output / "preprocessing_audit.json"
    frame = pd.read_parquet(metadata_path)
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    input_sha256 = _sha256(metadata_path)
    frame, audit = _finalize_quality(
        frame, audit, args.min_mask_fraction, "quality_only",
        metadata_input_sha256=input_sha256,
    )
    _write_parquet(frame, metadata_path)
    _write_audit(audit, audit_path)
    print(_json({
        "quality_only": True,
        "processing_records": audit["processing_records"],
        "face_records": audit["face_records"],
        "crop_records": audit["crop_records"],
        "eligible_faces": audit["eligible_faces"],
        "quality_policy": audit["quality_policy"],
    }), flush=True)
    return 0


def run(args):
    if getattr(args, "quality_only", False):
        return run_quality_only(args)
    if args.manifest is None:
        raise ValueError("--manifest is required unless --quality-only is used")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = _load_records(args.manifest.resolve(), args.data_root.resolve())
    if args.max_records is not None:
        records = records[:args.max_records]
    settings = {
        "yunet_model": str(args.yunet_model.resolve()),
        "landmarker_model": str(args.landmarker_model.resolve()),
        "size": args.size, "expansion": args.expansion, "crop_policy": args.crop_policy,
        "min_face_size": args.min_face_size, "detection_threshold": args.detection_threshold,
    }
    model_hashes = {key: _sha256(settings[key]) for key in ("yunet_model", "landmarker_model")}
    provenance = {
        "settings": settings, "model_sha256": model_hashes,
        "preprocessor_source_sha256": _sha256(ROOT / "src/preprocessing/faces.py"),
        "runner_source_sha256": _sha256(Path(__file__)),
    }
    fingerprint = hashlib.sha256(_json(provenance).encode()).hexdigest()
    crop_dir = output / "crops" / fingerprint[:16]
    crop_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(output / "checkpoint.sqlite")
    connection.execute("CREATE TABLE IF NOT EXISTS results (cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    connection.commit()
    all_rows = []
    cached = 0
    completed = 0
    started = time.monotonic()

    def accept(result):
        nonlocal completed
        key, rows = result
        connection.execute("INSERT OR REPLACE INTO results(cache_key, payload) VALUES (?, ?)", (key, _json(rows)))
        all_rows.extend(rows)
        completed += 1
        if completed % 25 == 0:
            connection.commit()
        if completed % 100 == 0:
            print(f"Processed {completed:,} images; reused {cached:,} checkpoints", flush=True)

    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_initialize_worker,
                                 initargs=(settings, str(crop_dir))) as executor:
            pending = set()
            for record in records:
                input_hash = _sha256(record["input_path"])
                key = hashlib.sha256((fingerprint + input_hash + _json(record)).encode()).hexdigest()
                reused = _cached_rows(connection, key) if args.resume else None
                if reused is not None:
                    all_rows.extend(reused)
                    cached += 1
                    continue
                pending.add(executor.submit(_process_task, key, record))
                if len(pending) >= args.workers * 3:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        accept(future.result())
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    accept(future.result())
        connection.commit()
    finally:
        connection.commit()
        connection.close()
    frame = pd.DataFrame(all_rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["dataset", "sample_id", "target", "eligible", "flags"])
    if frame["sample_id"].duplicated().any():
        raise ValueError("Preprocessing produced duplicate face sample_id values")
    frame = frame.sort_values(["dataset", "sample_id"], kind="stable").reset_index(drop=True)
    audit = {
        "images": len(records), "faces": len(frame),
        "eligible_faces": int(frame.get("eligible", pd.Series(dtype=bool)).sum()),
        "computed_images": completed, "reused_images": cached,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "fingerprint": fingerprint, "provenance": provenance,
        "manifest_path": str(args.manifest.resolve()), "manifest_sha256": _sha256(args.manifest),
    }
    frame, audit = _finalize_quality(frame, audit, args.min_mask_fraction, "post_inference")
    _write_parquet(frame, output / "metadata.parquet")
    _write_audit(audit, output / "preprocessing_audit.json")
    print(_json({key: audit[key] for key in ("images", "faces", "eligible_faces", "computed_images", "reused_images", "elapsed_seconds")}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quality-only", action="store_true")
    parser.add_argument("--min-mask-fraction", type=float, default=0.25)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--yunet-model", type=Path, default=ROOT / "models/face_detection_yunet_2023mar.onnx")
    parser.add_argument("--landmarker-model", type=Path, default=ROOT / "models/face_landmarker.task")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--expansion", type=float, default=1.3)
    parser.add_argument("--crop-policy", choices=("adaptive", "padded"), default="adaptive")
    parser.add_argument("--min-face-size", type=float, default=32)
    parser.add_argument("--detection-threshold", type=float, default=0.9)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.manifest is None and not args.quality_only:
        parser.error("--manifest is required unless --quality-only is used")
    if not math.isfinite(args.min_mask_fraction) or not 0 <= args.min_mask_fraction <= 1:
        parser.error("--min-mask-fraction must be finite and between 0 and 1")
    if args.max_records is not None and args.max_records < 0:
        parser.error("--max-records must be nonnegative")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
