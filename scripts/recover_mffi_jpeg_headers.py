from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.jpeg import extract_jpeg_metadata

REVISION = "fd9fce422da6ff231906dc06e984674f2332919a"
BASE_URL = "https://www.modelscope.cn/api/v1/datasets/DDLteam/MFFI/repo?Revision=" + REVISION + "&FilePath="
ARCHIVES = {
    "phase1": {"url": BASE_URL + "phase1.tar.gz", "size": 24509568862,
               "sha256": "9a5f9dd847d163a84e456872f6b608b53d957330148003d06f7a3386687f104b"},
    "phase2": {"url": BASE_URL + "phase2/test.tar.gz", "size": 7133438131,
               "sha256": "27f709ffa91755cbeb5258ec439d1722efdfa49b0e5c460c85429f139c446eb6"},
}


def jpeg_header_prefix(data):
    if data[:2] != b"\xff\xd8":
        raise ValueError("Selected MFFI source is not JPEG")
    offset = 2
    while offset < len(data):
        if data[offset] != 255:
            raise ValueError("Invalid JPEG marker before SOS")
        while offset < len(data) and data[offset] == 255:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in (1, 216, 217) or 208 <= marker <= 215:
            continue
        if offset + 2 > len(data):
            break
        length = int.from_bytes(data[offset:offset + 2], "big")
        if length < 2:
            raise ValueError("Invalid JPEG segment length")
        end = offset + length
        if end > len(data):
            break
        if marker == 218:
            return bytes(data[:end])
        offset = end
    raise ValueError("JPEG header exceeds --max-header-bytes or is truncated")


def selected_sources(metadata):
    frame = pd.read_parquet(metadata)
    frame = frame.loc[frame["source_format"].astype(str).str.upper().isin(["JPEG", "JPG"])]
    unique = frame.drop_duplicates("image_sample_id")
    wanted = {}
    for row in unique.to_dict("records"):
        source = str(row["input_path"]).replace("\\", "/")
        if "/mffi/" not in source:
            raise ValueError("Cannot identify MFFI archive member: " + source)
        member = source.rsplit("/mffi/", 1)[1]
        phase = member.split("/", 1)[0]
        if phase not in ARCHIVES:
            raise ValueError("Unknown MFFI phase: " + member)
        key = phase, member
        if key in wanted:
            raise ValueError("Duplicate selected MFFI archive member: " + member)
        wanted[key] = {
            "dataset": "mffi", "image_sample_id": str(row["image_sample_id"]),
            "image_sha256": str(row["image_sha256"]), "source_format": str(row["source_format"]),
            "member": member,
        }
    authority = json.dumps(sorted(wanted.values(), key=lambda row: row["image_sample_id"]),
                           sort_keys=True, separators=(",", ":")).encode()
    return wanted, hashlib.sha256(authority).hexdigest()


def add_phase2_eda(wanted, raw_root, limit=20000):
    from src.preprocessing.datasets import read_mffi_labels, sample_manifest, stable_id

    labels_path = raw_root / "phase2/testset_label.txt"
    expected_sha = "24f9be8e2942d066454a9d135465157dbf79e036b40bb4b3d195e40352285288"
    if labels_path.is_file():
        content = labels_path.read_bytes()
    else:
        with requests.get(BASE_URL + "phase2/testset_label.txt", timeout=(30, 180)) as response:
            response.raise_for_status()
            content = response.content
    if hashlib.sha256(content).hexdigest() != expected_sha:
        raise ValueError("MFFI phase2 label SHA mismatch")
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_bytes(content)
    labels = read_mffi_labels(labels_path)
    selected = labels if limit is None else sample_manifest(labels, limit, ["target"], 42)
    added = 0
    for row in selected.to_dict("records"):
        member = "phase2/test/" + Path(row["filename"]).name
        key = "phase2", member
        if key not in wanted:
            wanted[key] = {
                "dataset": "mffi", "image_sample_id": stable_id("mffi", member),
                "image_sha256": None, "source_format": "JPEG", "member": member,
            }
            added += 1
    return {"labels_sha256": expected_sha, "label_images": len(labels), "selected_test_images": len(selected),
            "additional_test_images": added, "target_counts": {str(key): int(value) for key, value in selected["target"].value_counts().items()}}


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def save_checkpoint(output, records, provenance):
    rows = list(records.values())
    if rows:
        temporary = output.with_suffix(".parquet.tmp")
        pd.DataFrame(rows).to_parquet(temporary, index=False)
        temporary.replace(output)
    _atomic_json(output.parent / "mffi_provenance.json", provenance)


def load_checkpoint(output, authority):
    records = {}
    provenance_path = output.parent / "mffi_provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
    if provenance and provenance.get("input_authority_sha256") != authority:
        raise ValueError("MFFI header checkpoint belongs to different source metadata")
    provenance = provenance or {
        "schema_version": 1, "input_authority_sha256": authority, "source_revision": REVISION,
        "archives": {}, "complete": False,
    }
    if output.is_file():
        for row in pd.read_parquet(output).to_dict("records"):
            identifier = row["image_sample_id"]
            if identifier in records:
                raise ValueError("Duplicate checkpoint image_sample_id")
            records[identifier] = row
    return records, provenance


class CountingReader:
    def __init__(self, source, progress):
        self.source = source
        self.digest = hashlib.sha256()
        self.count = 0
        self.progress = progress

    def read(self, size=-1):
        block = self.source.read(size)
        self.digest.update(block)
        self.count += len(block)
        self.progress(self.count)
        return block


def read_selected_member(stream, expected, max_header_bytes, output_path=None):
    digest = hashlib.sha256()
    prefix = bytearray()
    size = 0
    temporary = None
    outgoing = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".part")
        outgoing = temporary.open("wb")
    try:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
            if outgoing is not None:
                outgoing.write(block)
            if len(prefix) < max_header_bytes:
                prefix.extend(block[:max_header_bytes - len(prefix)])
        actual_sha = digest.hexdigest()
        if expected.get("image_sha256") and actual_sha != expected["image_sha256"]:
            raise ValueError("MFFI source SHA mismatch: " + expected["member"])
        header = jpeg_header_prefix(prefix)
        metadata = extract_jpeg_metadata(header)
        if metadata["source_jpeg_quality_status"] == "unreadable":
            raise ValueError("Cannot parse verified JPEG header: " + expected["member"])
        if outgoing is not None:
            outgoing.close()
            outgoing = None
            temporary.replace(output_path)
        return header, metadata, size, actual_sha
    finally:
        if outgoing is not None:
            outgoing.close()
        if temporary is not None and temporary.is_file():
            temporary.unlink()


def recover_archive(phase, wanted, records, provenance, output, max_header_bytes, raw_root=None, checkpoint_every=1000):
    spec = ARCHIVES[phase]
    selected = {member: row for (part, member), row in wanted.items() if part == phase}
    if not selected:
        return
    completed = provenance["archives"].get(phase, {}).get("archive_sha256_verified", False)
    def available(row):
        return row["image_sample_id"] in records and (raw_root is None or (raw_root / row["member"]).is_file())
    if completed and all(available(row) for row in selected.values()):
        return
    start = time.monotonic()
    last = start
    new_records = 0

    def progress(count):
        nonlocal last
        now = time.monotonic()
        if now - last >= 20:
            found = sum(row["image_sample_id"] in records for row in selected.values())
            print(json.dumps({"phase": phase, "compressed_bytes": count,
                              "selected_found": found, "selected_total": len(selected),
                              "MiB_s": round(count / max(now - start, 1) / 2 ** 20, 1)}), flush=True)
            last = now

    print(json.dumps({"phase": phase, "url": spec["url"], "selected": len(selected)}), flush=True)
    with requests.get(spec["url"], stream=True, timeout=(30, 180), headers={"Accept-Encoding": "identity"}) as response:
        response.raise_for_status()
        reader = CountingReader(response.raw, progress)
        with tarfile.open(fileobj=reader, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                parts = PurePosixPath(member.name).parts
                normalized = "/".join(part for part in parts if part not in (".", "/"))
                candidates = [key for key in (normalized, phase + "/" + normalized) if key in selected]
                if len(candidates) > 1:
                    raise ValueError("Ambiguous MFFI tar member: " + member.name)
                if not candidates:
                    continue
                expected = selected[candidates[0]]
                identifier = expected["image_sample_id"]
                if available(expected):
                    continue
                incoming = archive.extractfile(member)
                if incoming is None:
                    raise ValueError("Selected MFFI tar member cannot be read")
                with incoming:
                    target = raw_root / expected["member"] if raw_root is not None else None
                    header, metadata, size, actual_sha = read_selected_member(incoming, expected, max_header_bytes, target)
                if size != member.size:
                    raise ValueError("Truncated selected MFFI image")
                source = {
                    "public_url": spec["url"], "archive_member": member.name, "source_revision": REVISION,
                    "source_archive_sha256_expected": spec["sha256"], "archive_sha256_verified": False,
                    "source_member_sha256_verified": bool(expected.get("image_sha256")), "source_member_bytes": size,
                }
                records[identifier] = {
                    "dataset": "mffi", "image_sample_id": identifier,
                    "image_sha256": actual_sha, "source_format": "JPEG",
                    "header_bytes": header, **metadata,
                    "recovery_source_json": json.dumps(source, sort_keys=True),
                }
                new_records += 1
                if new_records % checkpoint_every == 0:
                    provenance["recovered_images"] = len(records)
                    save_checkpoint(output, records, provenance)
        while reader.read(1024 * 1024):
            pass
        actual_sha = reader.digest.hexdigest()
        if reader.count != spec["size"] or actual_sha != spec["sha256"]:
            raise ValueError(f"MFFI archive digest mismatch: {phase} {reader.count} {actual_sha}")
    missing = [row["image_sample_id"] for row in selected.values() if row["image_sample_id"] not in records]
    if missing:
        raise ValueError(f"MFFI selected members missing from {phase}: {len(missing)}")
    for expected in selected.values():
        row = records[expected["image_sample_id"]]
        source = json.loads(row["recovery_source_json"])
        source["archive_sha256_verified"] = True
        source["source_member_sha256_verified"] = True
        row["recovery_source_json"] = json.dumps(source, sort_keys=True)
    provenance["archives"][phase] = {
        "public_url": spec["url"], "bytes": reader.count, "sha256": actual_sha,
        "archive_sha256_verified": True, "selected_images": len(selected),
        "elapsed_seconds": round(time.monotonic() - start, 1),
    }
    provenance["recovered_images"] = len(records)
    save_checkpoint(output, records, provenance)
    print(json.dumps({"phase": phase, "complete": True, **provenance["archives"][phase]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/processed/mffi/metadata.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "data/header_cache/jpeg/mffi_headers.parquet")
    parser.add_argument("--max-header-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--retain-images", action="store_true")
    parser.add_argument("--include-phase2-eda", action="store_true")
    parser.add_argument("--full-test", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw/mffi")
    args = parser.parse_args()
    if args.max_header_bytes < 1024 or args.attempts < 1 or args.checkpoint_every < 1:
        parser.error("Invalid header limit or attempts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wanted, authority = selected_sources(args.metadata)
    if args.full_test:
        wanted = {key: value for key, value in wanted.items() if key[0] == "phase2"}
        args.retain_images = True
    eda = add_phase2_eda(wanted, args.raw_root, None if args.full_test else 20000) if (args.include_phase2_eda or args.full_test) else None
    if eda is not None:
        print(json.dumps({"phase2_selection": eda, "full_test": args.full_test}), flush=True)
    if eda is not None:
        payload = json.dumps(sorted(wanted.values(), key=lambda row: row["image_sample_id"]), sort_keys=True, separators=(",", ":")).encode()
        authority = hashlib.sha256(payload).hexdigest()
    records, provenance = load_checkpoint(args.output, authority)
    provenance["phase2_eda_selection"] = eda
    expected = {row["image_sample_id"]: row for row in wanted.values()}
    if not set(records) <= set(expected):
        raise ValueError("Checkpoint contains images outside the selected metadata")
    for identifier, row in records.items():
        if expected[identifier].get("image_sha256") and row["image_sha256"] != expected[identifier]["image_sha256"]:
            raise ValueError("Checkpoint source SHA differs from metadata")
    for phase in ARCHIVES:
        for attempt in range(1, args.attempts + 1):
            try:
                recover_archive(phase, wanted, records, provenance, args.output, args.max_header_bytes,
                                args.raw_root if args.retain_images else None, args.checkpoint_every)
                break
            except (requests.RequestException, OSError, tarfile.TarError, EOFError) as error:
                provenance["last_error"] = str(error)
                save_checkpoint(args.output, records, provenance)
                if attempt == args.attempts:
                    raise
                print(json.dumps({"phase": phase, "retry": attempt, "error": str(error)}), flush=True)
    provenance["complete"] = len(records) == len(expected)
    provenance["recovered_images"] = len(records)
    save_checkpoint(args.output, records, provenance)
    if not provenance["complete"]:
        raise RuntimeError("Incomplete MFFI JPEG header recovery")
    if args.full_test:
        files = []
        for row in records.values():
            source = json.loads(row["recovery_source_json"])
            member = expected[row["image_sample_id"]]["member"]
            files.append({
                "path": member.split("phase2/", 1)[1],
                "sha256": row["image_sha256"], "bytes": source["source_member_bytes"],
            })
        _atomic_json(args.raw_root / "phase2/.selection_complete.json", {
            "requested_limit": len(expected), "seed": 42, "files": files,
            "source_url": ARCHIVES["phase2"]["url"], "archive_sha256": ARCHIVES["phase2"]["sha256"],
            "target_counts": eda["target_counts"], "full_test": True,
        })
    print(json.dumps({"complete": True, "images": len(records), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
