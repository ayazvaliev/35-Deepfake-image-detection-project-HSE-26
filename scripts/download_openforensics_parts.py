from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import hashlib
import os
import shutil
import subprocess
from pathlib import Path
import sys
import time
import threading

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.download_assets import download, extract_zip, file_hash, safe_target
from src.preprocessing.datasets import index_openforensics, sample_manifest


PARTS = ("Train_part_1.zip", "Train_part_2.zip", "Train_part_3.zip",
         "Train_part_4.zip", "Train_part_5.zip", "Val.zip",
         "Test-Dev_part_1.zip", "Test-Dev_part_2.zip")
ZENODO = "https://zenodo.org/records/5528418/files/"


class ArchivePrefetch:
    def __init__(self, directory, catalog, reserve_bytes=6 * 1024 ** 3):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.catalog = catalog
        self.reserve_bytes = int(reserve_bytes)
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.futures = {}
        self.lookahead = None
        self.current_name = None
        self.lock = threading.RLock()

    def _remaining(self, name):
        entry = self.catalog[name]
        destination = self.directory / name
        partial = destination.with_name(destination.name + ".part")
        present = destination.stat().st_size if destination.exists() else partial.stat().st_size if partial.exists() else 0
        return max(0, int(entry["size"]) - present)

    def _has_space(self, name):
        outstanding = {key for key, future in self.futures.items() if not future.done()}
        outstanding.add(name)
        remaining = sum(self._remaining(key) for key in outstanding)
        return shutil.disk_usage(self.directory).free >= remaining + self.reserve_bytes

    def _submit(self, name):
        if name in self.futures:
            return self.futures[name]
        if Path(name).name != name or name not in self.catalog:
            raise ValueError("Unexpected archive prefetch filename")
        entry = self.catalog[name]
        algorithm, expected = entry["checksum"].split(":")
        future = self.executor.submit(
            download, ZENODO + name + "?download=1", self.directory / name,
            expected, algorithm, entry["size"],
        )
        self.futures[name] = future
        return future

    def current(self, name):
        with self.lock:
            self.raise_failures()
            if self.current_name is not None and self.current_name != name:
                previous = self.futures.get(self.current_name)
                if previous is not None and not previous.done():
                    raise RuntimeError("Cannot advance before the current archive download completes")
            self.current_name = name
            if self.lookahead == name:
                self.lookahead = None
            if name not in self.futures and not self._has_space(name):
                raise OSError("Insufficient space for current archive plus reserved raw/metadata workspace")
            return self._submit(name)

    def prefetch(self, name):
        with self.lock:
            self.raise_failures()
            if name in self.futures:
                return True
            if self.lookahead is not None:
                return False
            if not self._has_space(name):
                print(json.dumps({"archive_prefetch": name, "deferred": "disk_reserve"}), flush=True)
                return False
            self.lookahead = name
            self._submit(name)
            print(json.dumps({"archive_prefetch": name, "scheduled": True}), flush=True)
            return True

    def raise_failures(self):
        with self.lock:
            for future in self.futures.values():
                if future.done() and not future.cancelled():
                    error = future.exception()
                    if error is not None:
                        raise error

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)



def next_archive_to_prefetch(names, position, parts_root, keep_raw):
    for name in names[position + 1:]:
        part = Path(name).stem
        ready_path = parts_root / (part + ".ready.json")
        processed_path = parts_root / (part + ".processed.json")
        if processed_path.exists() and not keep_raw:
            processed = json.loads(processed_path.read_text())
            if processed.get("metadata_complete"):
                continue
        if ready_path.exists():
            ready = json.loads(ready_path.read_text())
            if all(Path(path).is_file() for path in ready["owned_paths"]):
                continue
        return name
    return None



def native(value):
    if value is None or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [native(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(native(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def manifest_chunk(inventory, owned_paths):
    owned = {str(Path(path).resolve()) for path in owned_paths}
    selected = inventory[inventory["path"].map(lambda path: str(Path(path).resolve()) in owned)].copy()
    actual = set(selected["path"].map(lambda path: str(Path(path).resolve())))
    if actual != owned:
        raise ValueError("Archive members do not match the annotated full inventory")
    if selected["sample_id"].duplicated().any():
        raise ValueError("Duplicate annotated frame IDs in part")
    return selected.sort_values("sample_id", kind="stable")


def cleanup_owned(raw_root, ready, processed):
    if not processed.get("metadata_complete"):
        return False
    if processed.get("raw_deleted"):
        return True
    if not (processed.get("original_views_verified") or processed.get("cleanup_allowed")):
        return False
    images = (Path(raw_root) / "images").resolve()
    validated = []
    for record in ready["owned_files"]:
        original = Path(record["path"])
        path = original.resolve()
        if images not in path.parents or original.is_symlink():
            raise ValueError("Refusing to remove a path outside owned raw image root")
        if path.exists():
            if file_hash(path) != record["sha256"]:
                raise ValueError("Owned raw image changed before cleanup")
            validated.append(path)
    for path in validated:
        path.unlink()
    return True



def verify_views(frame, require_crops=False, workers=8):
    import cv2

    if workers < 1:
        raise ValueError("Verification workers must be positive")
    cv2.setNumThreads(1)
    from src.preprocessing.verification_views import read_verification_view

    checks = []
    counts = {}
    for prefix, present_column in (("original", "decoded_sha256"), ("crop", "crop_sha256")):
        if prefix == "crop" and not require_crops:
            continue
        present = frame[present_column].fillna("").astype(str).ne("")
        needed = frame.loc[present]
        columns = [prefix + "_verification_" + suffix for suffix in
                   ("path", "sha256", "pixel_sha256", "size", "codec")]
        if len(needed) and not set(columns) <= set(needed):
            raise ValueError(f"Missing {prefix} verification views before raw cleanup")
        if not len(needed):
            counts[prefix] = 0
            continue
        if needed[columns[0]].isna().any():
            raise ValueError(f"A measured image has no {prefix} verification view")
        unique = needed[columns].drop_duplicates()
        counts[prefix] = len(unique)
        checks.extend(unique.itertuples(index=False, name=None))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = set()
        for values in dict.fromkeys(checks):
            pending.add(executor.submit(read_verification_view, *values))
            if len(pending) >= min(24, workers * 3):
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
        for future in pending:
            future.result()
    return counts


def preprocessing_arguments(config):
    if config is None:
        return []
    import yaml

    payload = yaml.safe_load(Path(config).read_text()) or {}
    settings = payload.get("preprocessing", payload)
    if settings.get("padding_value", 127) != 127:
        raise ValueError("The shared face crop contract requires padding_value=127")
    if str(settings.get("output_format", "PNG")).upper() != "PNG":
        raise ValueError("The shared face crop contract requires lossless PNG")
    names = ("size", "expansion", "crop_policy", "min_face_size",
             "min_mask_fraction", "detection_threshold")
    arguments = []
    for name in names:
        if name in settings:
            arguments.extend(["--" + name.replace("_", "-"), str(settings[name])])
    return arguments


def preprocessing_settings(config):
    defaults = {
        "size": 256, "expansion": 1.3, "crop_policy": "adaptive",
        "min_face_size": 32.0, "min_mask_fraction": 0.25, "detection_threshold": 0.9,
    }
    arguments = preprocessing_arguments(config)
    supplied = dict(zip(arguments[::2], arguments[1::2]))
    for name in defaults:
        value = supplied.get("--" + name.replace("_", "-"), defaults[name])
        if name == "size":
            defaults[name] = int(value)
        elif name == "crop_policy":
            defaults[name] = str(value)
        else:
            defaults[name] = float(value)
    return defaults


def current_part_provenance(root):
    root = Path(root)
    sources = {
        "preprocessor_source_sha256": "src/preprocessing/faces.py",
        "runner_source_sha256": "scripts/preprocess.py",
        "jpeg_source_sha256": "src/preprocessing/jpeg.py",
        "verification_views_source_sha256": "src/preprocessing/verification_views.py",
    }
    result = {key: file_hash(root / relative) for key, relative in sources.items()}
    result["model_sha256"] = {
        "yunet_model": file_hash(root / "models/face_detection_yunet_2023mar.onnx"),
        "landmarker_model": file_hash(root / "models/face_landmarker.task"),
    }
    return result


def reusable_part(processed, output, part, compact, config, root=ROOT, ready=None):
    if not processed or not processed.get("metadata_complete") or not processed.get("original_views_verified"):
        return False
    metadata = Path(processed.get("metadata_path", output / "parts" / part / "metadata.parquet"))
    audit_path = metadata.parent / "preprocessing_audit.json"
    if not metadata.is_file() or not audit_path.is_file():
        return False
    audit = json.loads(audit_path.read_text())
    provenance = audit.get("provenance", {})
    try:
        expected_provenance = current_part_provenance(root)
    except OSError:
        return False
    if any(provenance.get(key) != value for key, value in expected_provenance.items()):
        return False
    optional_sources = {
        "source_crops_source_sha256": "src/preprocessing/source_crops.py",
        "source_crop_source_sha256": "src/preprocessing/source_crops.py",
    }
    for key, relative in optional_sources.items():
        if key in provenance and provenance[key] != file_hash(Path(root) / relative):
            return False
    manifest_value = (ready or {}).get("manifest") or audit.get("manifest_path")
    if not manifest_value:
        return False
    manifest = Path(manifest_value)
    if not manifest.is_file() or file_hash(manifest) != audit.get("manifest_sha256"):
        return False
    model_settings = provenance.get("settings", {})
    actual_compact = processed.get("compact_storage", model_settings.get("compact_verification_crops", False))
    if bool(actual_compact) != bool(compact):
        return False
    desired = preprocessing_settings(config)
    actual = processed.get("preprocessing_settings")
    if actual is None:
        actual = {key: model_settings.get(key) for key in desired}
        actual["min_mask_fraction"] = audit.get("quality_policy", {}).get("min_mask_fraction")
    if any(actual.get(key) != value for key, value in desired.items()):
        return False
    expected_sha256 = processed.get("metadata_sha256")
    if expected_sha256 and file_hash(metadata) != expected_sha256:
        raise ValueError("Completed part metadata SHA256 mismatch")
    frame = pd.read_parquet(metadata)
    try:
        verify_views(frame, require_crops=compact)
    except (OSError, ValueError):
        return False
    if not compact:
        crop_rows = frame.loc[frame["crop_sha256"].fillna("").astype(str).ne("")]
        if "crop_file_sha256" not in crop_rows and len(crop_rows):
            return False
        for row in crop_rows.itertuples(index=False):
            filename = getattr(row, "crop_path", None)
            if not isinstance(filename, str) or not Path(filename).is_file():
                return False
            if file_hash(filename) != row.crop_file_sha256:
                return False
    return True


def needs_raw_restoration(ready, keep_raw, needs_processing):
    if ready is None or not (keep_raw or needs_processing):
        return False
    return any(not Path(path).is_file() for path in ready["owned_paths"])



def preprocess_part(root, output, part, ready, workers, compact, config=None):
    part_output = output / "parts" / part
    command = [sys.executable, str(root / "scripts/preprocess.py"),
               "--manifest", ready["manifest"], "--output-dir", str(part_output),
               "--workers", str(workers)]
    command.extend(preprocessing_arguments(config))
    if compact:
        command.append("--compact-verification-crops")
    subprocess.run(command, cwd=root, check=True)
    metadata = part_output / "metadata.parquet"
    audit_path = part_output / "preprocessing_audit.json"
    audit = json.loads(audit_path.read_text())
    if audit["images"] != ready["image_count"]:
        raise ValueError("Preprocessed part image count differs from staged manifest")
    verified = verify_views(pd.read_parquet(metadata), require_crops=compact)
    return {
        "metadata_complete": True, "original_views_verified": True,
        "crop_views_verified": compact, "verified_view_counts": verified,
        "image_count": ready["image_count"], "metadata_path": str(metadata),
        "metadata_sha256": file_hash(metadata), "compact_storage": compact,
        "preprocessing_settings": preprocessing_settings(config),
    }


def merge_parts(parts_root, names, output, expected_images):
    frames, audits = [], []
    for name in names:
        part = Path(name).stem
        processed = json.loads((parts_root / (part + ".processed.json")).read_text())
        if not processed.get("metadata_complete"):
            raise ValueError("Cannot merge an unfinished archive part")
        metadata = Path(processed.get("metadata_path", output / "parts" / part / "metadata.parquet"))
        frame = pd.read_parquet(metadata)
        frames.append(frame)
        audit = json.loads((metadata.parent / "preprocessing_audit.json").read_text())
        audits.append({"part": part, "metadata_path": str(metadata), "audit": audit})
    frame = pd.concat(frames, ignore_index=True)
    if frame["sample_id"].duplicated().any():
        raise ValueError("Duplicate face sample IDs across archive parts")
    if frame["image_sample_id"].nunique() != expected_images:
        raise ValueError("Merged image IDs do not cover the selected annotated inventory")
    frame = frame.sort_values(["dataset", "sample_id"], kind="stable").reset_index(drop=True)
    required_provenance = {
        "settings", "model_sha256", "preprocessor_source_sha256",
        "runner_source_sha256", "jpeg_source_sha256", "verification_views_source_sha256",
    }
    canonical = audits[0]["audit"].get("provenance", {})
    if not required_provenance <= set(canonical) or any(not canonical[key] for key in required_provenance):
        raise ValueError("Archive part has incomplete canonical inference provenance")
    if not all(canonical["model_sha256"].get(key) for key in ("yunet_model", "landmarker_model")):
        raise ValueError("Archive part has missing model weight hashes")
    if any(entry["audit"].get("provenance") != canonical for entry in audits):
        raise ValueError("Archive part settings, model weights or inference source hashes differ")
    fingerprints = {entry["audit"].get("fingerprint") for entry in audits}
    if len(fingerprints) != 1 or not next(iter(fingerprints)):
        raise ValueError("Archive part inference fingerprints differ")
    quality_counts = {
        "processing_records", "known_fraction_records", "missing_fraction_records",
        "too_small_fraction_records", "eligible_before", "eligible_after", "newly_excluded_records",
    }
    policies = [entry["audit"].get("quality_policy", {}) for entry in audits]
    if any(not quality_counts <= set(policy) for policy in policies):
        raise ValueError("Archive part has incomplete quality policy counts")
    quality_static = {key: value for key, value in policies[0].items() if key not in quality_counts}
    if any({key: value for key, value in policy.items() if key not in quality_counts} != quality_static
           for policy in policies):
        raise ValueError("Archive part quality policies differ")
    quality_policy = {
        **quality_static,
        **{key: sum(int(policy[key]) for policy in policies) for key in quality_counts},
    }
    if quality_policy["processing_records"] != len(frame) or quality_policy["eligible_after"] != int(frame["eligible"].sum()):
        raise ValueError("Aggregated quality counts differ from actual face metadata")
    quality_provenances = [entry["audit"].get("quality_provenance", {}) for entry in audits]
    variable_fields = {"applied_at_utc", "inference_rerun", "metadata_input_sha256"}
    quality_base = {key: value for key, value in quality_provenances[0].items() if key not in variable_fields}
    if not quality_base.get("policy_source_sha256") or any(
        {key: field for key, field in item.items() if key not in variable_fields} != quality_base
        for item in quality_provenances
    ):
        raise ValueError("Archive part quality provenance differs")
    quality_provenance = {
        **quality_base, "application_mode": "aggregated_parts",
        "applied_at_utc": max(value.get("applied_at_utc", "") for value in quality_provenances),
        "inference_rerun": any(value.get("inference_rerun", False) for value in quality_provenances),
        "metadata_input_sha256": None,
        "parts": [{"part": entry["part"], "quality_provenance": value}
                  for entry, value in zip(audits, quality_provenances)],
    }
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "metadata.parquet.tmp"
    frame.to_parquet(temporary, index=False)
    temporary.replace(output / "metadata.parquet")
    audit = {
        "images": expected_images, "faces": len(frame), "processing_records": len(frame),
        "face_records": int(frame["face_index"].notna().sum()),
        "crop_records": int(frame["crop_path"].fillna("").astype(str).ne("").sum()),
        "eligible_faces": int(frame["eligible"].sum()),
        "computed_images": sum(entry["audit"].get("computed_images", 0) for entry in audits),
        "reused_images": sum(entry["audit"].get("reused_images", 0) for entry in audits),
        "fingerprint": next(iter(fingerprints)),
        "provenance": {**canonical, "aggregation": "Concatenated actual independently completed archive parts",
                       "parts": audits, "only_annotated_frames": True},
        "quality_policy": quality_policy, "quality_provenance": quality_provenance,
        "original_views_verified": True,
    }
    write_json(output / "preprocessing_audit.json", audit)
    return audit


def run(args):
    root = args.root.resolve()
    raw_base = (args.raw_root or Path(os.environ.get("DEEPFAKE_DATA_ROOT", root / "data/raw"))).expanduser().resolve()
    raw = raw_base / "openforensics"
    annotation_root = raw / "annotations"
    output = args.preprocess_output.resolve() if args.preprocess_output else None
    profile = output.parent if output else root / "data/processed/full"
    parts_root = profile / "manifests/of_parts"
    parts_root.mkdir(parents=True, exist_ok=True)
    response = requests.get("https://zenodo.org/api/records/5528418", timeout=60)
    response.raise_for_status()
    files = {entry["key"]: entry for entry in response.json()["files"]}
    for split in ("Train", "Val", "Test-Dev"):
        name = split + "_poly.json"
        entry = files[name]
        algorithm, expected = entry["checksum"].split(":")
        download(ZENODO + name + "?download=1", annotation_root / name,
                 expected, algorithm, entry["size"])
    inventory = index_openforensics(raw, existing_only=False)
    print(json.dumps({"annotated_images": len(inventory),
                      "splits": inventory.author_split.value_counts().to_dict()}), flush=True)
    if len(inventory) != args.expected_images:
        raise ValueError(f"Expected {args.expected_images} annotated frames, received {len(inventory)}")
    if inventory["sample_id"].duplicated().any():
        raise ValueError("Duplicate frame IDs in full inventory")
    inventory.to_parquet(parts_root.parent / "openforensics_inventory.parquet", index=False)
    inventory = sample_manifest(inventory, min(args.limit, len(inventory)),
                                ["author_split", "frame_composition", "n_faces"], args.seed)
    selection_sha256 = hashlib.sha256("\n".join(sorted(inventory["sample_id"])).encode()).hexdigest()
    needed = {Path(path).name for path in inventory["path"]}
    old_path = root / "data/processed/openforensics/metadata.parquet"
    previous_hashes = {}
    if old_path.exists():
        old = pd.read_parquet(old_path, columns=["input_path", "image_sha256"]).drop_duplicates("input_path")
        previous_hashes = old.set_index("input_path")["image_sha256"].to_dict()
    names = list(args.parts or PARTS)
    if len(names) != len(set(names)):
        raise ValueError("Archive parts must not repeat")
    prefetch_count = getattr(args, "prefetch_archives", 0)
    archive_prefetch = ArchivePrefetch(raw_base / ".archives", files) if prefetch_count and not args.stage_only else None
    try:
        for position, name in enumerate(names):
            if archive_prefetch:
                archive_prefetch.raise_failures()
            if name not in PARTS:
                raise ValueError(f"Unknown official archive part: {name}")
            part = Path(name).stem
            ready_path = parts_root / (part + ".ready.json")
            processed_path = parts_root / (part + ".processed.json")
            cached_ready = json.loads(ready_path.read_text()) if ready_path.exists() else None
            processed_state = json.loads(processed_path.read_text()) if processed_path.exists() else None
            needs_processing = output is not None and not reusable_part(
                processed_state, output, part, args.compact_storage, args.config, root=root, ready=cached_ready,
            )
            restore_raw = needs_raw_restoration(cached_ready, args.keep_raw, needs_processing)
            if cached_ready is not None and not restore_raw:
                ready = cached_ready
                if ready.get("selection_sha256", selection_sha256) != selection_sha256:
                    raise ValueError("Staged part belongs to another image selection")
            else:
                entry = files[name]
                algorithm, expected = entry["checksum"].split(":")
                if archive_prefetch:
                    current_download = archive_prefetch.current(name)
                    next_name = next_archive_to_prefetch(names, position, parts_root, args.keep_raw)
                    if next_name:
                        archive_prefetch.prefetch(next_name)
                    archive = current_download.result()
                else:
                    archive = download(ZENODO + name + "?download=1", raw_base / ".archives" / name,
                                       expected, algorithm, entry["size"])
                archive_sha256 = file_hash(archive)
                extracted = extract_zip(archive, raw / "images", needed)
                owned_files = []
                for item in extracted:
                    path = safe_target(raw / "images", item["path"])
                    if str(path) in previous_hashes and previous_hashes[str(path)] != item["sha256"]:
                        raise ValueError("Reacquired image differs from retained original SHA256")
                    owned_files.append({"path": str(path), "sha256": item["sha256"], "bytes": item["bytes"]})
                selected = manifest_chunk(inventory, [item["path"] for item in owned_files])
                manifest = parts_root / (part + ".jsonl")
                temporary = manifest.with_name(manifest.name + ".tmp")
                with temporary.open("w") as stream:
                    for row in selected.to_dict("records"):
                        stream.write(json.dumps(native(row), ensure_ascii=False, allow_nan=False) + "\n")
                temporary.replace(manifest)
                ready = {
                    "archive": name, "archive_sha256": archive_sha256,
                    "archive_source_checksum": entry["checksum"], "source_url": ZENODO + name,
                    "manifest": str(manifest), "image_count": len(selected),
                    "owned_paths": [item["path"] for item in owned_files], "owned_files": owned_files,
                    "annotation_only": True, "original_views_required_before_cleanup": True,
                    "selection_sha256": selection_sha256, "limit": args.limit, "seed": args.seed,
                }
                write_json(ready_path, ready)
                archive.unlink()
                print(json.dumps({"part": part, "ready": str(ready_path),
                                  "image_count": len(selected)}), flush=True)
            if archive_prefetch:
                archive_prefetch.raise_failures()
                next_name = next_archive_to_prefetch(names, position, parts_root, args.keep_raw)
                if next_name:
                    archive_prefetch.prefetch(next_name)
            if args.stage_only:
                return 0
            if output is not None and (restore_raw or cached_ready is None):
                needs_processing = not reusable_part(
                    processed_state, output, part, args.compact_storage, args.config, root=root, ready=ready,
                )
            if needs_processing:
                processed = preprocess_part(root, output, part, ready, args.workers, args.compact_storage, args.config)
                write_json(processed_path, processed)
            if archive_prefetch:
                archive_prefetch.raise_failures()
            while True:
                if processed_path.exists():
                    processed = json.loads(processed_path.read_text())
                    if args.keep_raw and processed.get("metadata_complete") and processed.get("original_views_verified"):
                        for record in ready["owned_files"]:
                            path = Path(record["path"])
                            if not path.is_file() or file_hash(path) != record["sha256"]:
                                raise ValueError("Raw retention requires complete unchanged staged originals")
                        processed["raw_deleted"] = False
                        processed["raw_retained"] = True
                        write_json(processed_path, processed)
                        print(json.dumps({"part": part, "processed": True, "owned_raw_retained": True}), flush=True)
                        break
                    if cleanup_owned(raw, ready, processed):
                        processed["raw_deleted"] = True
                        write_json(processed_path, processed)
                        print(json.dumps({"part": part, "processed": True,
                                          "owned_raw_deleted": True}), flush=True)
                        break
                time.sleep(10)
    finally:
        if archive_prefetch:
            archive_prefetch.close()
    if output is not None:
        expected = sum(json.loads((parts_root / (Path(name).stem + ".ready.json")).read_text())["image_count"]
                       for name in args.parts or PARTS)
        if not args.parts and expected != len(inventory):
            raise ValueError("Archive part manifests do not cover the complete selected inventory")
        merge_parts(parts_root, args.parts or PARTS, output, expected)
    print(json.dumps({"complete": True, "annotated_inventory": len(inventory)}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--parts", nargs="+", choices=PARTS)
    parser.add_argument("--expected-images", type=int, default=70300)
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--preprocess-output", type=Path)
    parser.add_argument("--limit", type=int, default=70300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--compact-storage", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--prefetch-archives", type=int, choices=(0, 1), default=0)
    parser.add_argument("--config", "--preprocessing-config", type=Path, dest="config")
    args = parser.parse_args()
    if args.limit < 1 or args.workers < 1:
        parser.error("--limit and --workers must be positive")
    if args.compact_storage and not args.preprocess_output:
        parser.error("--compact-storage requires --preprocess-output")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
