from __future__ import annotations

import hashlib
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import html.parser
import io
import json
import re
import shutil
import struct
import threading
import zlib
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd



METADATA_ID = "1rsyQVdpAcgFsPbQdlSrpPbejV-8kBbbh"
METADATA_FOLDER = "19VhAL4aDJOKvhl9stEq_ymFeHiXo6_j-"
TEST_FOLDER = "1980LCMAutfWvV6zvdxhoeIa67TmzKLQ_"
TRAIN_FOLDER = "1U8meBbqVvmUkc5GD0jxct6xe6Gwk9wKD"
METHODS = {
    "VQGAN": ("EFS", "ff_cdf", "1rjCi-PvrtmSf4nR1tJm8i_MFOCjwcbSf", "1UBYXZm1ZgS6_lywxONQ-hNq-g41qbhYU"),
    "StyleGAN2": ("EFS", "ff_cdf", "1AF9Le3doLoWyaBXc26zR-FUZzugMuiVr", "12LQnIp9gTtem9Wo4GMr6Q7MNVgxfp5Rg"),
    "StyleGAN3": ("EFS", "ff_cdf", "1PJT1C5_ffl8FRW4WhbtVTjcQVxNKa6M_", "1D_1Rp2-K-IoxvleuahmBSNTd2BmdLPUM"),
    "StyleGANXL": ("EFS", "ff_cdf", "1A8ubPW2aLQ-DUdlLTkae8SqnDVUjRY6E", "1UKDPdXd_p1iF_qux6RSyk9GWecy8WMoD"),
    "sd2.1": ("EFS", "ff_cdf", "1S8v_mKDEO5uBZFAQkBxOGATYpiCTt_t6", "1rRbjGij6Zznkj5PV7vAL1c3r_pWIGQa6"),
    "ddim": ("EFS", "ff_cdf", "1SuvkPN4UVyBUzY3dwlYEQDhVdK_fM-B_", "1Ex0-7nZFC_MsOC30uQEfG5Ey2e6jwwK3"),
    "RDDM": ("EFS", "ff_cdf", "1H4fssCmNGyoIx8eKhM-93e5thyH5sP0Z", "1VygeiCcCcj2wSvdyEDnIdeqDIEBk0NcQ"),
    "pixart": ("EFS", "ff_cdf", "18utEqz5VA-ALZVBGMdsH2tAMlIA6kkEK", "1LY6XzMhh5sxRGwcyLIKgn8G1KrNuSbe8"),
    "DiT": ("EFS", "ff_cdf", "1Y15vpgltFD1amMEABCRkOiDEK-aQpRJ8", "1cDaupU8HhUtyc_UXPZf1UZac-NGPcV5K"),
    "SiT": ("EFS", "ff_cdf", "1WCLokp2LUQdQLmOrK8iEM1d9T6C4vtJT", "1bViVK7sYOvP7m46T5ZcXvdQqfF6-X-pn"),
    "MidJourney": ("EFS", "ffhq", "1K0_sUcVLY4V6g7MF_hCEzoPagIMxdAxh", None),
    "whichfaceisreal": ("EFS", "ffhq", "1zlmnZx9bbBKtTqpYJvkRR0NVSuxr4DuT", None),
    "CollabDiff": ("FE", "celeba", "1GiG38Hbw_gCgXGFLpkix5jvbIgB_hmJr", None),
    "e4e": ("FE", "ff_cdf", "1p3U8UolSJs5Mp7rd1NRVxjr3hrwHEx-r", None),
    "stargan": ("FE", "celeba", "1LfZaVSb95Lzuaaqee9AnqbSa3cAwunC4", None),
    "starganv2": ("FE", "celeba", "1NC8akz76RQneJIrLL8S6JObDn5VzjyC-", None),
    "styleclip": ("FE", "celeba", "1AodpPEnL7njJqFEbBchR9GbXtWAbR2fS", None),
}
FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP",
           ".bmp": "BMP", ".tif": "TIFF", ".tiff": "TIFF"}


def drive_url(file_id):
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


class _FolderParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.files = {}

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if "data-id" in attributes and "data-tooltip" in attributes:
            self.files[attributes["data-id"]] = attributes["data-tooltip"]


def list_drive_folder(folder_id):
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    with urllib.request.urlopen(url, timeout=90) as response:
        document = response.read().decode("utf-8")
    parser = _FolderParser()
    parser.feed(document)
    if not parser.files:
        raise RuntimeError(f"No public files listed in official folder {folder_id}")
    return {"url": url, "files": parser.files,
            "possibly_truncated": len(parser.files) >= 50}


def download_drive_file(file_id, destination):
    import gdown

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = drive_url(file_id)
    with HTTPRangeFile(url) as remote:
        expected_size = remote.size
        if not destination.exists():
            gdown.download(url=url, output=str(destination), quiet=False, resume=True)
        existing_size = destination.stat().st_size if destination.exists() else 0
        if existing_size == expected_size:
            return destination
        valid_prefix = False
        if 0 < existing_size < expected_size:
            with destination.open("rb") as stream:
                prefix = stream.read(min(existing_size, 64))
            remote.seek(0)
            valid_prefix = prefix == remote.read(len(prefix))
        chunk_size = 16 * 1024 * 1024
        offset = (existing_size // chunk_size) * chunk_size if valid_prefix else 0
        mode = "r+b" if offset else "wb"
        print(f"HTTP Range download: {file_id}, {offset}/{expected_size} bytes; workers=4", flush=True)
        with destination.open(mode) as stream, ThreadPoolExecutor(max_workers=4) as executor:
            stream.seek(offset)
            stream.truncate(offset)
            pending = deque()
            next_offset = offset
            while pending or next_offset < expected_size:
                while len(pending) < 8 and next_offset < expected_size:
                    last = min(next_offset + chunk_size, expected_size) - 1
                    future = executor.submit(remote._request, next_offset, last)
                    pending.append((next_offset, future))
                    next_offset = last + 1
                expected_offset, future = pending.popleft()
                block, _ = future.result()
                if expected_offset != offset:
                    raise RuntimeError("Noncontiguous download write")
                stream.write(block)
                offset += len(block)
        if destination.stat().st_size != expected_size:
            raise RuntimeError(f"Official Google Drive download incomplete: {file_id}")
    return destination


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url):
        super().__init__()
        self.url = url
        self.position = 0
        self.sessions = threading.local()
        _, content_range = self._request(0, 0)
        self.size = int(content_range.rsplit("/", 1)[1])

    def _request(self, first, last):
        import requests

        session = getattr(self.sessions, "value", None)
        if session is None:
            session = requests.Session()
            self.sessions.value = session
        error = None
        for attempt in range(8):
            try:
                with session.get(
                    self.url, headers={"Range": f"bytes={first}-{last}"},
                    timeout=(20, 90), stream=True,
                ) as response:
                    content_range = response.headers.get("Content-Range", "")
                    if response.status_code == 429:
                        error = RuntimeError("HTTP 429 rate limit")
                        retry_after = response.headers.get("Retry-After", "")
                        window = re.search(r"t=(\d+)", response.headers.get("RateLimit", ""))
                        delay = float(retry_after) if retry_after.isdigit() else float(window.group(1)) if window else 5 * (2 ** min(attempt, 3))
                        response.close()
                        time.sleep(min(delay, 30))
                        continue
                    if response.status_code != 206 or not content_range.startswith(f"bytes {first}-"):
                        if last - first + 1 > 1024 * 1024:
                            pieces = []
                            for offset in range(first, last + 1, 1024 * 1024):
                                piece, _ = self._request(offset, min(offset + 1024 * 1024 - 1, last))
                                pieces.append(piece)
                            return b"".join(pieces), f"bytes {first}-{last}/{self.size}"
                        raise RuntimeError(f"HTTP Range unavailable, status={response.status_code}, content_type={response.headers.get('Content-Type', '')}")
                    self.url = response.url
                    data = response.content
                if len(data) != last - first + 1:
                    raise OSError("Truncated HTTP Range response")
                return data, content_range
            except (OSError, RuntimeError, requests.RequestException) as caught:
                error = RuntimeError(type(caught).__name__) if isinstance(caught, requests.RequestException) else caught
                if attempt < 7:
                    time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Cannot read archive range: {error}") from error

    def close(self):
        session = getattr(self.sessions, "value", None)
        if session is not None:
            session.close()
        super().close()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if position < 0:
            raise ValueError("Negative seek position")
        self.position = position
        return position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        data, _ = self._request(self.position, self.position + size - 1)
        self.position += len(data)
        return data


def archive_specs():
    rows = []
    for method, (manipulation, domain, test_id, train_id) in METHODS.items():
        for partition, file_id in (("test", test_id), ("train", train_id)):
            if file_id is not None:
                rows.append({
                    "archive_id": file_id, "archive_name": method + ".zip",
                    "archive_partition": partition, "generator": method,
                    "manipulation": manipulation, "default_domain": domain,
                    "archive_url": drive_url(file_id),
                })
    return rows


def safe_image_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive member {name!r}")
    if any(part.startswith(("._", "__MACOSX")) for part in path.parts):
        return False
    return path.suffix.lower() in FORMATS


def _member_tail(member, method):
    parts = list(PurePosixPath(member.replace("\\", "/")).parts)
    matching = [index for index, part in enumerate(parts) if part.lower() == method.lower()]
    if matching:
        parts = parts[matching[0] + 1:]
    while parts and parts[0].lower() == method.lower():
        parts.pop(0)
    return "/".join(part for part in parts if part != ".")


def _canonical_tail(tail, partition):
    parts = [part for part in PurePosixPath(tail).parts if part not in {".", "frames"}]
    if partition == "train" and parts and parts[0].lower() == "ff":
        parts = parts[1:]
    return "/".join(parts)


def _source_fields(method, domain, tail, target, metadata_group=None, archive_partition="unknown"):
    parts = PurePosixPath(tail).parts
    if target == 0:
        source = f"df40:native_real:{method}:{domain}:{archive_partition}:{tail}"
        return source, "", "native_real_serial_unknown_identity"
    if METHODS[method][0] == "EFS":
        source = f"df40:efs:{method}:{domain}:{archive_partition}:image:{tail}"
        return source, "", "generated_image_unknown_identity"
    source = None
    for index, part in enumerate(parts[:-1]):
        if domain == "ff" and re.fullmatch(r"\d{3}", part):
            source = f"global:ff:{part}"
            break
        if domain == "cdf" and re.fullmatch(r"id\d+_\d+", part, re.I):
            source = f"global:cdf:{part.lower()}"
            break
    if source is None and metadata_group and domain in {"ff", "cdf"}:
        candidate = re.sub(r"_" + re.escape(method) + r"$", "", metadata_group, flags=re.I)
        if domain == "ff" and re.fullmatch(r"\d{3}", candidate):
            source = f"global:ff:{candidate}"
        elif domain == "cdf" and re.fullmatch(r"id\d+_\d+", candidate, re.I):
            source = f"global:cdf:{candidate.lower()}"
    identity = ""
    if source and domain == "cdf":
        identity = "global:cdf:" + source.rsplit(":", 1)[1].split("_", 1)[0]
    if source:
        return source, identity, "published_source_identifier"
    return f"df40:fe:{method}:{domain}:{archive_partition}:image:{tail}", "", "edited_image_unknown_identity"


def _metadata_lookup(document):
    lookup = {}
    priority = {"train": 1, "val": 2, "validation": 2, "test": 3}
    aliases = {method.lower(): method for method in METHODS}
    for datasets in document.values():
        for label, partitions in datasets.items():
            for partition, entries in partitions.items():
                if partition not in priority:
                    continue
                for group, item in entries.items():
                    if not isinstance(item, dict) or "frames" not in item:
                        continue
                    for raw_path in item["frames"]:
                        parts = PurePosixPath(raw_path.replace("\\", "/")).parts
                        roots = [i for i, part in enumerate(parts) if part in {"DF40", "DF40_train"}]
                        if not roots or roots[0] + 1 >= len(parts):
                            continue
                        root = roots[0]
                        method = aliases.get(parts[root + 1].lower())
                        if method is None:
                            continue
                        origin_partition = "train" if parts[root] == "DF40_train" else "test"
                        tail = _member_tail("/".join(parts[root + 1:]), method)
                        key = (origin_partition, method, _canonical_tail(tail, origin_partition))
                        existing = lookup.get(key)
                        value = {
                            "metadata_label": item.get("label", label),
                            "metadata_group": group,
                            "author_split": "validate" if partition in {"val", "validation"} else partition,
                            "priority": priority[partition],
                            "author_path": raw_path,
                        }
                        if existing is None or value["priority"] > existing["priority"]:
                            lookup[key] = value
    return lookup


def fetch_metadata(root):
    root = Path(root)
    folder_snapshots = {}
    for name, folder in (("train", TRAIN_FOLDER), ("test", TEST_FOLDER), ("metadata", METADATA_FOLDER)):
        folder_snapshots[name] = list_drive_folder(folder)
    for spec in archive_specs():
        listing = folder_snapshots[spec["archive_partition"]]["files"]
        if spec["archive_id"] not in listing:
            raise RuntimeError(f"Official archive not found in current primary listing: {spec}")
    write_json(root / "metadata" / "official_listing.json", folder_snapshots)
    metadata_path = root / "metadata" / "DF40_all.json"
    try:
        document = json.loads(metadata_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        if metadata_path.exists():
            metadata_path.unlink()
        download_drive_file(METADATA_ID, metadata_path)
        document = json.loads(metadata_path.read_text())
    write_json(root / "metadata" / "provenance.json", {
        "metadata_url": drive_url(METADATA_ID), "metadata_sha256": file_sha256(metadata_path),
        "metadata_bytes": metadata_path.stat().st_size,
        "official_repository": "https://github.com/YZY-stack/DF40",
        "metadata_listing_may_be_truncated": folder_snapshots["metadata"]["possibly_truncated"],
        "author_split_protocol": "Official combined DF40_all.json exact-path assignments override archive partitions; unmatched paths retain released archive train/test. Per-method held-out JSONs represent a different protocol and are not mixed into this inventory.",
    })
    return _metadata_lookup(document)


def inventory_archive(spec, metadata=None):
    metadata = metadata or {}
    records = []
    with HTTPRangeFile(spec["archive_url"]) as remote:
        archive_size = remote.size
        with zipfile.ZipFile(remote) as archive:
            all_members = sorted(archive.infolist(), key=lambda item: item.header_offset)
            ends = {
                item.header_offset: all_members[index + 1].header_offset
                if index + 1 < len(all_members) else archive.start_dir
                for index, item in enumerate(all_members)
            }
            for item in all_members:
                if item.is_dir() or not safe_image_member(item.filename):
                    continue
                tail = _member_tail(item.filename, spec["generator"])
                parts = PurePosixPath(tail).parts
                lowered = [part.lower() for part in parts]
                domain = "ff" if "ff" in lowered else "cdf" if "cdf" in lowered else spec["default_domain"]
                if spec["archive_partition"] == "train" and domain == "ff_cdf":
                    domain = "ff"
                explicit_real = "real" in lowered
                explicit_fake = "fake" in lowered
                if spec["default_domain"] in {"ffhq", "celeba"} and not (explicit_real or explicit_fake):
                    raise RuntimeError(f"Cannot infer native real/fake label: {item.filename}")
                target = 0 if explicit_real else 1
                key = (spec["archive_partition"], spec["generator"],
                       _canonical_tail(tail, spec["archive_partition"]))
                annotation = metadata.get(key, {})
                label = annotation.get("metadata_label", "")
                if label.lower().endswith("_real") and target != 0:
                    raise RuntimeError(f"Archive label conflicts with author metadata: {item.filename}")
                if label.lower().endswith("_fake") and target != 1:
                    raise RuntimeError(f"Archive label conflicts with author metadata: {item.filename}")
                source, identity, basis = _source_fields(
                    spec["generator"], domain, tail, target, annotation.get("metadata_group"),
                    spec["archive_partition"],
                )
                identifier = f"df40:{spec['archive_partition']}:{spec['generator']}:{tail}"
                extension = PurePosixPath(item.filename).suffix.lower()
                record = dict(spec)
                record.update(
                    dataset="DF40", sample_id="df40_" + hashlib.sha256(identifier.encode()).hexdigest()[:24],
                    target=target, generator=spec["generator"] if target else "real",
                    native_method=spec["generator"], native_family=spec["manipulation"],
                    manipulation=spec["manipulation"] if target else "real", domain=domain,
                    author_split=annotation.get("author_split", spec["archive_partition"]),
                    author_metadata_status="matched" if annotation else "archive_partition_only",
                    source_group=source, identity_group=identity, source_group_basis=basis,
                    generation_batch=_generation_batch(spec["generator"], domain, tail, spec["archive_partition"]) if target else "",
                    lineage_status="unknown" if "unknown" in basis else "source_identifier",
                    source_group_policy_version=3,
                    archive_member=item.filename, archive_bytes=archive_size,
                    uncompressed_bytes=item.file_size, zip_crc32=f"{item.CRC:08x}",
                    compressed_bytes=item.compress_size, zip_compression=item.compress_type,
                    zip_header_offset=item.header_offset, zip_entry_end=ends[item.header_offset],
                    source_extension=extension, source_format=FORMATS[extension],
                    source_format_verified=False, native_still_image=True,
                    media_provenance="generated_still" if target else "native_real_still",
                    author_path=annotation.get("author_path", ""),
                    metadata_label=label,
                )
                records.append(record)
    return records


def build_inventory(root, refresh=False):
    root = Path(root)
    snapshot = load_frozen_inventory(root) if not refresh else None
    if snapshot is not None:
        return snapshot
    metadata = fetch_metadata(root)
    pieces = []
    for index, spec in enumerate(archive_specs(), 1):
        cache = root / "metadata" / "archive_indexes" / f"{spec['archive_partition']}_{spec['generator']}.json"
        if cache.is_file():
            records = json.loads(cache.read_text())
        else:
            records = []
        if not records:
            print(f"Inventory {index}/27: {spec['archive_partition']}/{spec['generator']}", flush=True)
            records = inventory_archive(spec, metadata)
            write_json(cache, records)
        if not records:
            raise RuntimeError(f"No native images in official archive {spec['archive_id']}")
        pieces.extend(records)
    population = pd.DataFrame.from_records(pieces)
    population["native_family"] = population["native_method"].map(lambda method: METHODS[method][0])
    population.loc[population["target"].eq(0), "manipulation"] = "real"
    population = apply_source_group_policy(population)
    if population["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate native sample identifiers in DF40 inventory")
    root.mkdir(parents=True, exist_ok=True)
    author_inventory = sorted(population["author_split"].unique().tolist())
    population["author_split_inventory"] = [author_inventory for _ in range(len(population))]
    population.to_parquet(root / "inventory.parquet", index=False)
    provenance_path = root / "metadata" / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance.update(
        population_n=len(population),
        author_mapping_counts=population["author_metadata_status"].value_counts().to_dict(),
        author_partition_counts=population["author_split"].value_counts().to_dict(),
        author_split_inventory=author_inventory,
    )
    write_json(provenance_path, provenance)
    return population


def sample_inventory(population, max_images=100_000, seed=42):
    from .splits import proportional_sample

    return proportional_sample(
        population, n=min(max_images, len(population)), seed=seed,
        strata=["target", "native_method", "manipulation", "domain", "author_split", "source_format"],
    )


def _acquire_full(root, selected):
    root = Path(root).resolve()
    staging = root / "_downloads"
    staging.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "metadata" / "downloads.json"
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.is_file() else {}
    records = []
    for archive_id, rows in selected.groupby("archive_id", sort=False):
        first = rows.iloc[0]
        archive_path = staging / f"{archive_id}.zip"
        expected_size = int(first["archive_bytes"])
        existing = checkpoint.get(archive_id, {})
        local_records = []
        for row in rows.to_dict("records"):
            relative = Path("images") / row["archive_partition"] / row["archive_member"]
            destination = root / relative
            saved = existing.get("files", {}).get(row["sample_id"], {})
            good = (
                destination.is_file() and destination.stat().st_size == row["uncompressed_bytes"]
                and saved.get("sha256") == file_sha256(destination)
            )
            row.update(path=str(destination), input_path=str(destination), relative_path=str(relative))
            if good:
                row["download_sha256"] = saved["sha256"]
            local_records.append((row, good))
        pending = [row for row, good in local_records if not good]
        if pending:
            free = shutil.disk_usage(root).free
            needed = expected_size + sum(int(row["uncompressed_bytes"]) for row in pending) + 1024 ** 3
            if free < needed:
                raise RuntimeError(f"Not enough free disk for serial archive {first['archive_name']}: need {needed}, free {free}")
            if not (archive_path.is_file() and archive_path.stat().st_size == expected_size):
                print(f"Download {first['archive_partition']}/{first['archive_name']} for {len(pending)} selected images", flush=True)
                download_drive_file(archive_id, archive_path)
            if archive_path.stat().st_size != expected_size:
                raise RuntimeError(f"Archive size differs from HTTP inventory: {archive_path}")
            archive_digest = file_sha256(archive_path)
            existing = {
                "archive_url": drive_url(archive_id), "archive_sha256": archive_digest,
                "archive_bytes": expected_size, "complete": False,
                "files": existing.get("files", {}),
            }
            checkpoint[archive_id] = existing
            write_json(checkpoint_path, checkpoint)
            with zipfile.ZipFile(archive_path) as archive:
                available = set(archive.namelist())
                for row in pending:
                    member = row["archive_member"]
                    if not safe_image_member(member) or member not in available:
                        raise RuntimeError(f"Selected native image missing from archive: {member}")
                    destination = Path(row["path"])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_suffix(destination.suffix + ".partial")
                    digest = hashlib.sha256()
                    with archive.open(member) as source, temporary.open("wb") as target:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            target.write(block)
                            digest.update(block)
                    if temporary.stat().st_size != row["uncompressed_bytes"]:
                        raise RuntimeError(f"Extracted size mismatch: {member}")
                    temporary.replace(destination)
                    row["download_sha256"] = digest.hexdigest()
                    existing["files"][row["sample_id"]] = {
                        "relative_path": row["relative_path"], "sha256": digest.hexdigest(),
                        "bytes": row["uncompressed_bytes"],
                    }
            existing["complete"] = True
            write_json(checkpoint_path, checkpoint)
            archive_path.unlink()
        records.extend(row for row, _ in local_records)
    manifest = pd.DataFrame.from_records(records)
    if len(manifest) != len(selected) or manifest["path"].isna().any():
        raise RuntimeError("DF40 acquisition is incomplete")
    manifest.to_parquet(root / "sample.parquet", index=False)
    return manifest


def _coalesced_blocks(records, max_gap=64 * 1024, max_block=1024 * 1024):
    ordered = sorted(records, key=lambda row: int(row["zip_header_offset"]))
    blocks = []
    for row in ordered:
        first, last = int(row["zip_header_offset"]), int(row["zip_entry_end"]) - 1
        if blocks and first - blocks[-1][1] - 1 <= max_gap and last - blocks[-1][0] + 1 <= max_block:
            blocks[-1][1] = last
            blocks[-1][2].append(row)
        else:
            blocks.append([first, last, [row]])
    return blocks


def _extract_selected_block(remote, block):
    first, last, records = block
    content, _ = remote._request(first, last)
    results = []
    for row in records:
        offset = int(row["zip_header_offset"]) - first
        header = struct.unpack_from("<IHHHHHIIIHH", content, offset)
        if header[0] != 0x04034B50 or header[2] & 1:
            raise RuntimeError(f"Invalid or encrypted ZIP entry: {row['archive_member']}")
        if header[3] != int(row["zip_compression"]):
            raise RuntimeError(f"ZIP compression metadata mismatch: {row['archive_member']}")
        start = offset + 30 + header[9] + header[10]
        end = start + int(row["compressed_bytes"])
        if end > len(content):
            raise RuntimeError(f"Truncated ZIP entry: {row['archive_member']}")
        compressed = content[start:end]
        if header[3] == zipfile.ZIP_STORED:
            image = compressed
        elif header[3] == zipfile.ZIP_DEFLATED:
            image = zlib.decompress(compressed, -15)
        elif header[3] == zipfile.ZIP_BZIP2:
            import bz2

            image = bz2.decompress(compressed)
        elif header[3] == zipfile.ZIP_LZMA:
            image = zipfile.LZMADecompressor().decompress(compressed)
        else:
            raise RuntimeError(f"Unsupported ZIP compression: {header[3]}")
        if len(image) != int(row["uncompressed_bytes"]):
            raise RuntimeError(f"ZIP uncompressed size mismatch: {row['archive_member']}")
        if f"{zlib.crc32(image) & 0xffffffff:08x}" != row["zip_crc32"]:
            raise RuntimeError(f"ZIP CRC mismatch: {row['archive_member']}")
        destination = Path(row["path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        temporary.write_bytes(image)
        temporary.replace(destination)
        results.append({
            "sample_id": row["sample_id"], "relative_path": row["relative_path"],
            "sha256": hashlib.sha256(image).hexdigest(), "bytes": len(image),
            "zip_crc32": row["zip_crc32"], "download_source": row.get("_download_source", {}),
        })
    return results


def _acquire_selective(root, selected, workers, backend):
    if workers < 1 or workers > 32:
        raise ValueError("DF40 download workers must be between one and thirty-two")
    root = Path(root).resolve()
    checkpoint_path = root / "metadata" / "downloads.json"
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.is_file() else {}
    records = []
    deferred = []
    for archive_id, rows in selected.groupby("archive_id", sort=False):
        first = rows.iloc[0]
        existing = checkpoint.get(archive_id, {})
        file_records = existing.get("files", {})
        previous_source = existing.get("download_source", {})
        for saved in file_records.values():
            saved.setdefault("download_source", previous_source)
        adopt_stream_records(root, archive_id, rows, file_records)
        existing.update(
            archive_url=drive_url(archive_id), archive_bytes=int(first["archive_bytes"]),
            mode="selective_http_range", complete=False,
            integrity="HTTPS primary archive; per-entry ZIP CRC32 and uncompressed length; extracted SHA256. Full archive SHA256 was not computed.",
            files=file_records,
        )
        existing.pop("archive_sha256", None)
        checkpoint[archive_id] = existing
        local_records = []
        pending_records = []
        for row in rows.to_dict("records"):
            member = row["archive_member"]
            if not safe_image_member(member):
                raise RuntimeError(f"Unsafe selected native member: {member}")
            relative = Path("images") / row["archive_partition"] / member
            destination = root / relative
            row.update(path=str(destination), input_path=str(destination), relative_path=str(relative))
            saved = file_records.get(row["sample_id"], {})
            valid = (
                destination.is_file()
                and destination.stat().st_size == int(row["uncompressed_bytes"])
                and saved.get("sha256") == file_sha256(destination)
            )
            if valid:
                row["download_sha256"] = saved["sha256"]
            else:
                pending_records.append(row)
            local_records.append(row)
        if pending_records:
            needed = sum(int(row["uncompressed_bytes"]) for row in pending_records) + 1024 ** 3
            if shutil.disk_usage(root).free < needed:
                raise RuntimeError(f"Not enough disk for selected {first['archive_name']} images")
            try:
                source = resolve_download_source(root, pending_records, backend)
            except RuntimeError as error:
                message = f"{first['archive_partition']}/{first['native_method']}: {error}"
                deferred.append(message)
                existing["source_error"] = message
                write_json(checkpoint_path, checkpoint)
                print("Deferred unavailable source: " + message, flush=True)
                continue
            existing["download_source"] = source
            for row in pending_records:
                row["_download_source"] = source
            existing["integrity"] = (
                "Each downloaded image is verified against authoritative Google Drive inventory "
                "by canonical member path, ZIP CRC32 and uncompressed length; extracted SHA256 "
                "is checkpointed. Mirror archive LFS SHA256 is a declaration, not a full-file readback."
            )
            blocks = _coalesced_blocks(pending_records)
            print(
                f"Selective {first['archive_partition']}/{first['archive_name']}: "
                f"{len(pending_records)} images, {len(blocks)} HTTP ranges, workers={workers}",
                flush=True,
            )
            write_json(checkpoint_path, checkpoint)
            completed = 0
            last_checkpoint = 0
            last_report = 0
            with HTTPRangeFile(source["url"]) as remote:
                if remote.size != source["bytes"]:
                    raise RuntimeError("Download archive size changed after source verification")
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = set()
                    iterator = iter(blocks)
                    exhausted = False
                    while futures or not exhausted:
                        while not exhausted and len(futures) < workers * 2:
                            block = next(iterator, None)
                            if block is None:
                                exhausted = True
                            else:
                                futures.add(executor.submit(_extract_selected_block, remote, block))
                        done, futures = wait(futures, return_when=FIRST_COMPLETED)
                        for future in done:
                            for result in future.result():
                                file_records[result["sample_id"]] = result
                                completed += 1
                        if completed - last_checkpoint >= 100:
                            write_json(checkpoint_path, checkpoint)
                            last_checkpoint = completed
                        if completed - last_report >= 500:
                            print(f"  Verified {completed}/{len(pending_records)} native images", flush=True)
                            last_report = completed
            for row in local_records:
                row["download_sha256"] = file_records[row["sample_id"]]["sha256"]
        for row in local_records:
            source = file_records[row["sample_id"]].get("download_source", existing.get("download_source", {}))
            row.pop("_download_source", None)
            row.update(
                download_source_url=source.get("url", ""),
                download_source_kind=source.get("kind", "primary"),
                mirror_repository=source.get("repository", ""),
                mirror_revision=source.get("revision", ""),
                mirror_lfs_sha256=source.get("lfs_sha256", ""),
                mirror_archive_sha256_verified=False,
                primary_member_crc_verified=True,
                source_tar_member=file_records[row["sample_id"]].get("source_tar_member", ""),
            )
        existing.pop("source_error", None)
        existing["complete"] = True
        existing["selected_images"] = len(local_records)
        write_json(checkpoint_path, checkpoint)
        own_partial_archive = root / "_downloads" / f"{archive_id}.zip"
        if own_partial_archive.is_file():
            own_partial_archive.unlink()
        records.extend(local_records)
    manifest = pd.DataFrame.from_records(records)
    mechanical = ["zip_header_offset", "zip_entry_end", "compressed_bytes", "zip_compression", "download_member"]
    manifest = manifest.drop(columns=mechanical, errors="ignore")
    complete = len(manifest) == len(selected) and not deferred
    write_json(root / "metadata" / "acquisition_status.json", {
        "complete": complete, "requested_images": len(selected),
        "verified_images": len(manifest), "deferred_sources": deferred,
    })
    if not complete or ("path" in manifest and manifest["path"].isna().any()):
        manifest.to_parquet(root / "sample.partial.parquet", index=False)
        raise RuntimeError(f"DF40 acquisition incomplete: verified {len(manifest)}/{len(selected)}, deferred {len(deferred)} sources. Selection is unchanged.")
    manifest.to_parquet(root / "sample.parquet", index=False)
    return manifest


def acquire_selected(root, selected, archive_mode="selective", workers=16, backend="auto"):
    if archive_mode == "selective":
        return _acquire_selective(root, selected, workers, backend)
    if archive_mode == "full":
        return _acquire_full(root, selected)
    raise ValueError("archive_mode must be selective or full")


HF_MIRRORS = (
    ("ManhQuangAI/df-40-test-full", "b052e68b675c4c52b36439cfa1c88bdb49bb92b8", "test"),
    ("ManhQuangAI/DF40_train", "ef0577c67506afea6df7a8bb46dd4ed4a8e138a0", "train"),
    ("ej2/df40", "e0d28c7b8ca85aa5a8102e1850e2c271fb23f24e", None),
)


def hf_mirror_catalog(root):
    path = Path(root) / "metadata" / "hf_mirror_catalog.json"
    if path.is_file():
        return json.loads(path.read_text())
    catalog = []
    for repository, revision, partition in HF_MIRRORS:
        url = f"https://huggingface.co/api/datasets/{repository}/revision/{revision}?blobs=true"
        with urllib.request.urlopen(url, timeout=90) as response:
            metadata = json.load(response)
        if metadata.get("private") or metadata.get("gated"):
            continue
        if metadata.get("sha") != revision:
            raise RuntimeError(f"Mirror revision mismatch: {repository}")
        for item in metadata.get("siblings", []):
            filename = item["rfilename"]
            if not filename.lower().endswith(".zip"):
                continue
            catalog.append({
                "kind": "hf_crc_verified", "repository": repository,
                "revision": revision, "filename": filename, "partition": partition,
                "url": f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{filename}",
                "bytes": int(item["size"]),
                "lfs_sha256": item.get("lfs", {}).get("sha256", ""),
                "lfs_sha256_verified_by_full_download": False,
            })
    write_json(path, catalog)
    return catalog


def match_mirror_records(primary_records, mirror_records):
    if not primary_records:
        return []
    method = primary_records[0]["native_method"]
    partition = primary_records[0]["archive_partition"]
    lookup = {}
    for row in mirror_records:
        tail = _member_tail(row["archive_member"], method)
        key = _canonical_tail(tail, partition)
        lookup.setdefault(key, []).append(row)
    matched = []
    errors = []
    for row in primary_records:
        tail = _member_tail(row["archive_member"], method)
        key = _canonical_tail(tail, partition)
        candidates = [
            item for item in lookup.get(key, [])
            if item["zip_crc32"] == row["zip_crc32"]
            and int(item["uncompressed_bytes"]) == int(row["uncompressed_bytes"])
        ]
        if len(candidates) != 1:
            errors.append(row["archive_member"])
            continue
        match = candidates[0]
        matched.append((row, match))
    if errors:
        raise RuntimeError(
            f"Mirror does not reproduce {len(errors)} selected authoritative members; "
            f"examples={errors[:3]}"
        )
    return matched


def resolve_download_source(root, records, backend):
    if backend not in {"auto", "primary", "hf-verified"}:
        raise ValueError("backend must be auto, primary or hf-verified")
    first = records[0]
    candidates = []
    availability_path = Path(root) / "metadata" / "availability.json"
    unavailable = set()
    if availability_path.is_file():
        unavailable = {
            item["id"] for item in json.loads(availability_path.read_text())
            if item.get("status") != 206
        }
    if backend == "primary" or (backend == "auto" and first["archive_id"] not in unavailable):
        candidates.append({
            "kind": "primary", "url": first["archive_url"], "bytes": int(first["archive_bytes"]),
            "repository": "", "revision": "", "lfs_sha256": "",
        })
    if backend != "primary":
        for candidate in hf_mirror_catalog(root):
            if candidate["partition"] not in (None, first["archive_partition"]):
                continue
            if Path(candidate["filename"]).name.lower() != first["native_method"].lower() + ".zip":
                continue
            candidates.append(candidate)
    if backend == "auto":
        candidates.sort(key=lambda candidate: candidate["kind"] == "primary")
        if first["archive_partition"] == "test" and first["native_method"] in {"e4e", "pixart"}:
            candidates = [candidate for candidate in candidates if candidate["kind"] != "primary"]
    failures = []
    for candidate in candidates:
        token = hashlib.sha256(candidate["url"].encode()).hexdigest()[:20]
        cache = Path(root) / "metadata" / "mirror_indexes" / f"{token}.json"
        try:
            if cache.is_file():
                mirror_records = json.loads(cache.read_text())
            else:
                specification = dict(
                    next(spec for spec in archive_specs() if spec["archive_id"] == first["archive_id"]),
                    archive_url=candidate["url"],
                )
                mirror_records = inventory_archive(specification)
                write_json(cache, mirror_records)
            pairs = match_mirror_records(records, mirror_records)
            for row, match in pairs:
                row["download_member"] = match["archive_member"]
                for column in ("zip_header_offset", "zip_entry_end", "compressed_bytes", "zip_compression"):
                    row[column] = match[column]
            candidate = dict(candidate, selected_primary_matches=len(pairs))
            print(
                f"Verified source for {first['archive_partition']}/{first['native_method']}: "
                f"{candidate['kind']}, {len(pairs)} exact path/CRC/length matches",
                flush=True,
            )
            return candidate
        except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
            failures.append({
                "url": candidate["url"], "repository": candidate.get("repository"),
                "error": str(error),
            })
    write_json(Path(root) / "metadata" / f"source_failure_{first['archive_partition']}_{first['native_method']}.json", failures)
    raise RuntimeError(
        f"No source reproduces selected {first['archive_partition']}/{first['native_method']} images. "
        f"Available candidates={len(candidates)}; errors={failures}"
    )


def adopt_stream_records(root, archive_id, rows, file_records):
    path = Path(root) / "metadata" / "stream_downloads.json"
    if not path.is_file():
        return
    external = json.loads(path.read_text()).get(archive_id, {})
    if not external:
        return
    selected = {row["sample_id"]: row for row in rows.to_dict("records")}
    source = external.get("download_source", {})
    for sample_id, saved in external.get("files", {}).items():
        if sample_id not in selected or sample_id in file_records:
            continue
        row = selected[sample_id]
        expected = Path("images") / row["archive_partition"] / row["archive_member"]
        if saved.get("relative_path") != expected.as_posix():
            raise RuntimeError(f"External stream path differs from primary member: {sample_id}")
        if saved.get("zip_crc32") != row["zip_crc32"] or int(saved["bytes"]) != int(row["uncompressed_bytes"]):
            raise RuntimeError(f"External stream metadata differs from primary inventory: {sample_id}")
        image = (Path(root) / expected).read_bytes()
        sha = hashlib.sha256(image).hexdigest()
        crc = f"{zlib.crc32(image) & 0xffffffff:08x}"
        if len(image) != int(row["uncompressed_bytes"]) or crc != row["zip_crc32"] or sha != saved.get("sha256"):
            raise RuntimeError(f"External stream image failed primary CRC/length/SHA check: {sample_id}")
        file_records[sample_id] = dict(saved, download_source=source)


def load_frozen_inventory(root):
    repository = Path(__file__).resolve().parents[2]
    snapshot = repository / "data" / "manifests" / "df40_inventory.parquet"
    provenance_path = snapshot.with_name("df40_inventory.provenance.json")
    if not snapshot.is_file() or not provenance_path.is_file():
        return None
    provenance = json.loads(provenance_path.read_text())
    if file_sha256(snapshot) != provenance.get("snapshot_sha256"):
        raise RuntimeError("Bundled primary DF40 inventory checksum mismatch")
    frame = pd.read_parquet(snapshot)
    if len(frame) != provenance.get("snapshot_population"):
        raise RuntimeError("Bundled primary DF40 inventory row count mismatch")
    if set(frame["native_method"]) != set(METHODS):
        raise RuntimeError("Bundled primary DF40 inventory method coverage mismatch")
    if not frame["target"].isin([0, 1]).all() or frame["sample_id"].duplicated().any():
        raise RuntimeError("Bundled primary DF40 inventory has invalid targets or identifiers")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / "inventory.parquet.tmp"
    shutil.copyfile(snapshot, temporary)
    temporary.replace(root / "inventory.parquet")
    provenance["inventory_origin"] = "bundled_primary_snapshot"
    write_json(root / "metadata" / "provenance.json", provenance)
    return frame


def _lineage_text(value):
    if value is None or value is pd.NA:
        return ""
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return ""
    return str(value)


def _generation_batch(method, domain, tail, archive_partition):
    parent = str(PurePosixPath(tail).parent)
    return f"df40:{method}:{domain}:{archive_partition}:batch:{parent}"


def apply_source_group_policy(frame):
    data = frame.copy()
    if "dataset" in data:
        selected = data["dataset"].fillna("").astype(str).str.casefold().eq("df40")
    elif "native_method" in data:
        selected = data["native_method"].isin(METHODS)
    else:
        selected = pd.Series(False, index=data.index)
    indices = data.index[selected]
    canonical_methods = {method.casefold(): method for method in METHODS}
    columns = {
        "source_group": [], "identity_group": [], "source_group_basis": [],
        "generation_batch": [], "lineage_status": [], "source_group_policy_version": [],
    }
    changed = 0
    for row in data.loc[selected].itertuples(index=False):
        method = _lineage_text(getattr(row, "native_method", ""))
        if method not in METHODS:
            method = canonical_methods.get(_lineage_text(getattr(row, "generator", "")).casefold(), method)
        domain = _lineage_text(getattr(row, "domain", "")) or "unknown"
        partition = _lineage_text(getattr(row, "archive_partition", "")) or "unknown"
        member = _lineage_text(getattr(row, "archive_member", ""))
        image_id = _lineage_text(getattr(row, "image_sample_id", ""))
        if not image_id:
            image_id = re.sub(r":face-\d+$", "", _lineage_text(getattr(row, "sample_id", "")))
        if not member and not image_id:
            raise ValueError("DF40 lineage correction requires archive_member or image sample_id")
        tail = _member_tail(member, method) if member else image_id
        old_source = _lineage_text(getattr(row, "source_group", ""))
        identity = _lineage_text(getattr(row, "identity_group", ""))
        basis = _lineage_text(getattr(row, "source_group_basis", ""))
        family = METHODS.get(method, (_lineage_text(getattr(row, "native_family", "")),))[0]
        target = getattr(row, "target", None)
        source = old_source
        if target == 0 and method in METHODS:
            source, identity, basis = _source_fields(method, domain, tail, 0, archive_partition=partition)
        elif family == "EFS":
            source = f"df40:efs:{method}:{domain}:{partition}:image:{tail}"
            identity, basis = "", "generated_image_unknown_identity"
        elif family == "FE" and method in METHODS and basis != "published_source_identifier":
            source, identity, basis = _source_fields(method, domain, tail, 1, archive_partition=partition)
        elif not source and method in METHODS:
            source, identity, basis = _source_fields(method, domain, tail, int(target), archive_partition=partition)
        changed += source != old_source or identity != _lineage_text(getattr(row, "identity_group", ""))
        columns["source_group"].append(source)
        columns["identity_group"].append(identity)
        columns["source_group_basis"].append(basis)
        columns["generation_batch"].append(
            _generation_batch(method, domain, tail, partition) if target != 0 else ""
        )
        columns["lineage_status"].append("unknown" if "unknown" in basis else "source_identifier")
        columns["source_group_policy_version"].append(3)
    for column, values in columns.items():
        if column not in data:
            data[column] = None
        data.loc[indices, column] = values
    audit = {
        "version": 3, "df40_records": int(selected.sum()),
        "changed_source_records": int(changed),
        "efs_unit": "generated_image_unknown_identity",
        "generation_batch_is_grouping_key": False,
        "generic_e4e_inversions_unit": "edited_image_unknown_identity",
        "unverified_fe_unit": "edited_image_unknown_identity",
        "explicit_source_identifiers_preserved": True,
    }
    data.attrs = {**frame.attrs, "df40_lineage_policy": audit}
    return data
