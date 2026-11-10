from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
POST_GROUP_FLAGS = frozenset({
    "author_split_overlap", "exact_face_duplicate",
    "contradictory_duplicate_target", "near_duplicate_verification_missing",
})
SIGNATURE_COLUMNS = (
    "sample_id", "dataset", "source_group", "identity_group", "target", "eligible", "bbox",
    "image_sha256", "decoded_sha256", "crop_sha256", "crop_file_sha256", "download_sha256",
    "file_sha256", "phash", "face_mask_fraction_of_crop", "mask_source",
    "source_group_basis", "source_group_policy_version", "lineage_status",
    "flags", "exclusion_flags",
)
_LOGIC = {
    "src/preprocessing/splits.py": (
        "_connect", "_Groups", "_HashChunks", "_HashTree", "_phash", "_value",
        "_GROUP_COLUMNS", "_HASH_COLUMNS",
    ),
    "src/preprocessing/df40.py": (
        "_source_fields", "apply_source_group_policy", "_member_tail",
        "_lineage_text", "_generation_batch", "METHODS",
    ),
    "src/preprocessing/quality.py": (
        "apply_quality_policy", "_flags", "MASK_FRACTION_COLUMN", "MASK_QUALITY_FLAGS",
    ),
    "src/preprocessing/near_duplicates.py": (
        "PixelNearDuplicateVerifier", "structural_similarity", "_text",
    ),
}


def _native(value):
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in sorted(value.items())}
    if isinstance(value, (set, frozenset)):
        return sorted(_native(item) for item in value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_native(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _json(value):
    return json.dumps(_native(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flags(value):
    if value is None or value is pd.NA or isinstance(value, float) and np.isnan(value):
        return set()
    if isinstance(value, str):
        if value.lstrip().startswith("["):
            return set(json.loads(value))
        return {item.strip() for item in value.split(";") if item.strip()}
    return {str(item) for item in value}


def grouping_fingerprints(source_commit=None):
    fingerprints = {}
    for relative, names in _LOGIC.items():
        if source_commit is None:
            source = (ROOT / relative).read_text()
        else:
            result = subprocess.run(
                ["git", "-C", str(ROOT), "show", f"{source_commit}:{relative}"],
                capture_output=True, text=True, check=True,
            )
            source = result.stdout
        nodes = {}
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                nodes[node.name] = node
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        nodes[target.id] = node
        for name in names:
            if name not in nodes:
                raise ValueError(f"Grouping dependency missing: {relative}:{name}")
            node = nodes[name]
            if name == "METHODS":
                registry = ast.literal_eval(node.value)
                payload = _json({method: entry[0] for method, entry in registry.items()})
            else:
                payload = ast.dump(node, include_attributes=False)
            fingerprints[f"{relative}:{name}"] = hashlib.sha256(payload.encode()).hexdigest()
    return fingerprints


def _row_table(data):
    from src.preprocessing.splits import _phash

    if "sample_id" not in data or data["sample_id"].isna().any():
        raise ValueError("Verified-group inputs require non-null sample_id")
    if data["sample_id"].astype(str).duplicated().any():
        raise ValueError("Verified-group inputs require unique sample_id")
    required = {"dataset", "source_group", "identity_group", "target", "eligible"}
    if not required <= set(data):
        raise ValueError(f"Unprepared grouping inputs: missing {sorted(required - set(data))}")
    selected = data.reindex(columns=SIGNATURE_COLUMNS)
    records = []
    for values in selected.itertuples(index=False, name=None):
        row = dict(zip(SIGNATURE_COLUMNS, values))
        row["sample_id"] = str(row["sample_id"])
        for column in ("flags", "exclusion_flags"):
            row[column] = sorted(_flags(row[column]))
        row["phash"] = _phash(row["phash"])
        signature = hashlib.sha256(_json(row).encode()).digest()
        records.append((row["sample_id"], signature))
    return pd.DataFrame(records, columns=["sample_id", "row_signature"]).sort_values("sample_id", kind="stable").reset_index(drop=True)


def _records_digest(table, include_group=False):
    digest = hashlib.sha256()
    for row in table.sort_values("sample_id", kind="stable").itertuples(index=False):
        identifier = str(row.sample_id).encode()
        digest.update(len(identifier).to_bytes(8, "big"))
        digest.update(identifier)
        digest.update(bytes(row.row_signature))
        if include_group:
            group = str(row.group_id).encode()
            digest.update(len(group).to_bytes(8, "big"))
            digest.update(group)
    return digest.hexdigest()


def _verification_settings(audit):
    return {key: value for key, value in audit["settings"].items() if key not in {"opencv_threads", "cache_size"}}


def _validate_verification(audit):
    from src.preprocessing.near_duplicates import PixelNearDuplicateVerifier

    if audit.get("mode") != "pixel_verification":
        raise ValueError("Only an actual pixel-verification audit can establish a reusable cache")
    if any(int(audit.get(key, 0)) for key in ("unreadable_pairs", "unreadable_rows", "excluded_rows")):
        raise ValueError("Verified-group cache requires a run with no unreadable verification candidates")
    if int(audit.get("candidate_pairs", -1)) != int(audit.get("verified_pairs", -1)) + int(audit.get("rejected_pairs", -1)):
        raise ValueError("Pixel-verification candidate accounting is inconsistent")
    verifier = PixelNearDuplicateVerifier(pd.DataFrame())
    settings = verifier.settings
    verifier.close()
    expected = {key: value for key, value in settings.items() if key not in {"opencv_threads", "cache_size"}}
    if _verification_settings(audit) != expected:
        raise ValueError("Pixel-verification settings differ from the cached verification")


def _validate_groups(data, groups):
    from src.preprocessing.splits import _GROUP_COLUMNS, _HASH_COLUMNS, _value

    identifiers = data["sample_id"].astype(str)
    labels = pd.Series(groups, index=data.index, dtype=object).astype(str)
    if labels.eq("").any() or not set(labels) <= set(identifiers):
        raise ValueError("Cached group_id must name an input sample")
    minima = identifiers.groupby(labels).min()
    if any(str(group) != str(minimum) for group, minimum in minima.items()):
        raise ValueError("Cached group_id is not the canonical minimum sample_id")
    eligible = data["eligible"].to_numpy(dtype=bool)
    datasets = data["dataset"].astype(str).to_numpy()
    label_values = labels.to_numpy()
    for column in (*_GROUP_COLUMNS, *_HASH_COLUMNS):
        seen = {}
        for position, (value, dataset) in enumerate(zip(data[column].to_numpy(), datasets)):
            if column == "crop_sha256" and not eligible[position]:
                continue
            value = _value(value)
            if not value:
                continue
            if column in _GROUP_COLUMNS and not value.startswith("global:"):
                value = (str(dataset), value)
            previous = seen.setdefault(value, label_values[position])
            if previous != label_values[position]:
                raise ValueError(f"Cached groups split a required {column} connection")


def _seal(manifest):
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return hashlib.sha256(_json(payload).encode()).hexdigest()


def _reuse_audit(path, manifest, reused):
    return {
        "cache_path": str(Path(path).resolve()),
        "reused": reused,
        "verified_now": not reused and manifest["provenance"]["origin"] == "fresh_pixel_verification",
        "origin": manifest["provenance"]["origin"],
        "bootstrap_from_previous_verified_run": manifest["provenance"]["origin"] == "bootstrap_previous_verified_run",
        "rows": manifest["rows"], "groups": manifest["groups"],
        "input_digest": manifest["input_digest"],
        "cache_file_sha256": manifest["cache_file_sha256"],
        "logic_fingerprints_sha256": hashlib.sha256(_json(manifest["logic_fingerprints"]).encode()).hexdigest(),
        "previous_run_provenance": manifest["provenance"],
    }


def write_verified_groups(data, path, near_duplicate_threshold, verification_audit, provenance=None):
    _validate_verification(verification_audit)
    if not isinstance(near_duplicate_threshold, int) or not 0 <= near_duplicate_threshold <= 16:
        raise ValueError("near_duplicate_threshold must be an integer from zero to sixteen")
    if "group_id" not in data:
        raise ValueError("Verified groups require group_id from the actual grouping run")
    _validate_groups(data, data["group_id"].to_numpy())
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing group cache: {path}")
    path.mkdir(parents=True, exist_ok=True)
    table = _row_table(data)
    lookup = data.assign(sample_id=data["sample_id"].astype(str)).set_index("sample_id")["group_id"]
    table["group_id"] = table["sample_id"].map(lookup).astype(str)
    destination = path / "groups.parquet"
    temporary = path / "groups.parquet.tmp"
    table.to_parquet(temporary, index=False, compression="zstd", compression_level=19)
    temporary.replace(destination)
    origin = {"origin": "fresh_pixel_verification"} if provenance is None else dict(provenance)
    manifest = {
        "schema_version": 1,
        "signature_columns": list(SIGNATURE_COLUMNS),
        "rows": len(table), "groups": int(table["group_id"].nunique()),
        "input_digest": _records_digest(table),
        "cache_records_digest": _records_digest(table, include_group=True),
        "cache_file_sha256": _sha256(destination),
        "near_duplicate_threshold": near_duplicate_threshold,
        "logic_fingerprints": grouping_fingerprints(),
        "verification_audit": verification_audit,
        "provenance": origin,
    }
    manifest["manifest_sha256"] = _seal(manifest)
    temporary_manifest = path / "manifest.json.tmp"
    temporary_manifest.write_text(_json(manifest) + "\n")
    temporary_manifest.replace(path / "manifest.json")
    return _reuse_audit(path, manifest, reused=False)


def load_verified_groups(data, path, near_duplicate_threshold):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("manifest_sha256") != _seal(manifest):
        raise ValueError("Verified-group cache manifest integrity mismatch")
    if manifest.get("schema_version") != 1 or manifest.get("signature_columns") != list(SIGNATURE_COLUMNS):
        raise ValueError("Unsupported verified-group cache signature schema")
    if manifest["near_duplicate_threshold"] != near_duplicate_threshold:
        raise ValueError("Verified-group cache near-duplicate threshold mismatch")
    if manifest["logic_fingerprints"] != grouping_fingerprints():
        raise ValueError("Verified-group cache grouping logic fingerprint mismatch")
    _validate_verification(manifest["verification_audit"])
    destination = path / "groups.parquet"
    if _sha256(destination) != manifest["cache_file_sha256"]:
        raise ValueError("Verified-group cache file integrity mismatch")
    table = pd.read_parquet(destination)
    if list(table.columns) != ["sample_id", "row_signature", "group_id"] or table["sample_id"].duplicated().any():
        raise ValueError("Invalid verified-group cache records")
    if len(table) != manifest["rows"] or table["group_id"].nunique() != manifest["groups"]:
        raise ValueError("Verified-group cache record accounting mismatch")
    if _records_digest(table, include_group=True) != manifest["cache_records_digest"]:
        raise ValueError("Verified-group cache record digest mismatch")
    current = _row_table(data)
    if _records_digest(current) != manifest["input_digest"]:
        old = table.set_index("sample_id")["row_signature"]
        new = current.set_index("sample_id")["row_signature"]
        if set(old.index) != set(new.index):
            raise ValueError("Verified-group cache input sample IDs changed")
        changed = [identifier for identifier in new.index if bytes(new[identifier]) != bytes(old[identifier])]
        raise ValueError(f"Verified-group cache input signatures changed: {changed[:8]}")
    groups = data["sample_id"].astype(str).map(table.set_index("sample_id")["group_id"]).to_numpy()
    _validate_groups(data, groups)
    return groups, manifest["verification_audit"], _reuse_audit(path, manifest, reused=True)


def _prepare_raw(frame):
    from src.preprocessing.df40 import apply_source_group_policy
    from src.preprocessing.labels import normalize_labels
    from src.preprocessing.splits import _prepare

    return _prepare(apply_source_group_policy(normalize_labels(frame)))


def bootstrap_verified_groups(raw, previous, previous_audit, path, near_duplicate_threshold, provenance):
    raw = _prepare_raw(raw)
    old = previous.copy()
    if "group_id" not in old or old["sample_id"].duplicated().any():
        raise ValueError("Previous exports must contain unique sample IDs and group_id")
    for column in ("flags", "exclusion_flags"):
        if column not in old:
            raise ValueError(f"Previous exports require {column} for quality restoration")
    had_post_flags = old["flags"].map(lambda value: bool(_flags(value) & POST_GROUP_FLAGS))
    had_post_flags |= old["exclusion_flags"].map(lambda value: bool(_flags(value) & POST_GROUP_FLAGS))
    for column in ("flags", "exclusion_flags"):
        old[column] = old[column].map(lambda value: sorted(_flags(value) - POST_GROUP_FLAGS))
    if had_post_flags.any():
        restored_eligible = old.loc[had_post_flags, "exclusion_flags"].map(lambda value: not value).astype(bool)
        old.loc[had_post_flags, "eligible"] = restored_eligible
    restored = _prepare_raw(old)
    raw_table, old_table = _row_table(raw), _row_table(restored)
    if not raw_table.equals(old_table):
        raise ValueError("Previous verified exports do not match canonical raw grouping inputs and restored quality")
    verification = previous_audit["near_duplicate_verification"]
    if int(previous_audit.get("input_rows", -1)) != len(raw):
        raise ValueError("Previous verification audit input row count differs")
    historical = provenance.get("historical_logic_fingerprints")
    if historical is None or historical != grouping_fingerprints():
        raise ValueError("Historical grouping logic does not match current grouping dependencies")
    if not provenance.get("executed_notebook_sha256") or not provenance.get("source_commit"):
        raise ValueError("Bootstrap requires previous executed-notebook SHA256 and source commit")
    lookup = old.assign(sample_id=old["sample_id"].astype(str)).set_index("sample_id")["group_id"]
    raw["group_id"] = raw["sample_id"].astype(str).map(lookup)
    origin = {
        **provenance, "origin": "bootstrap_previous_verified_run",
        "verification_performed_now": False,
        "raw_quality_restored_from_exports": True,
        "stripped_post_group_flags": sorted(POST_GROUP_FLAGS),
        "previous_run_audit": previous_audit,
    }
    result = write_verified_groups(raw, path, near_duplicate_threshold, verification, provenance=origin)
    result["verified_now"] = False
    return result
