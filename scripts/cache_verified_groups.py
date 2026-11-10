from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.group_cache import bootstrap_verified_groups, grouping_fingerprints, _sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--previous-splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data/group_cache/verified_groups")
    parser.add_argument("--near-duplicate-threshold", type=int, default=4)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-logic", type=Path)
    parser.add_argument("--executed-notebook", type=Path)
    parser.add_argument("--expected-notebook-sha256", required=True)
    args = parser.parse_args()
    notebook = args.executed_notebook or args.previous_splits / "eda_df40_executed.ipynb"
    notebook_sha256 = _sha256(notebook)
    if notebook_sha256 != args.expected_notebook_sha256:
        parser.error("Previous executed notebook SHA256 mismatch")
    if args.source_logic:
        evidence = json.loads(args.source_logic.read_text())
        if evidence["source_commit"] != args.source_commit:
            parser.error("Historical logic evidence source commit mismatch")
        historical = evidence["logic_fingerprints"]
    else:
        historical = grouping_fingerprints(source_commit=args.source_commit)
    manifests = [path.resolve() for path in args.manifest]
    partitions = [args.previous_splits / f"{name}.parquet" for name in ("train", "validate", "test", "excluded")]
    audit_path = args.previous_splits / "audit.json"
    audit = json.loads(audit_path.read_text())
    raw = pd.concat([pd.read_parquet(path) for path in manifests], ignore_index=True)
    previous = pd.concat([pd.read_parquet(path) for path in partitions], ignore_index=True)
    provenance = {
        "source_commit": args.source_commit,
        "executed_notebook_sha256": notebook_sha256,
        "executed_notebook_path": str(notebook.resolve()),
        "historical_logic_fingerprints": historical,
        "previous_audit_sha256": _sha256(audit_path),
        "previous_exports": [{"path": str(path.resolve()), "sha256": _sha256(path)} for path in partitions],
        "retained_input_manifests": [{"path": str(path), "sha256": _sha256(path)} for path in manifests],
        "original_raw_input_hashes_available": False,
        "input_hash_capture": "Current retained metadata at bootstrap; historical input file hashes were not recorded.",
        "old_partitions_used_for_assignment": False,
        "old_eligibility_used_for_output": False,
    }
    if args.source_logic:
        provenance["historical_logic_evidence_sha256"] = _sha256(args.source_logic)
    result = bootstrap_verified_groups(
        raw, previous, audit, args.output, args.near_duplicate_threshold, provenance,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
