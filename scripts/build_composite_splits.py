from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.splits import build_composite_splits


def _default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def _write_manifest(frame, output, partition):
    parquet = output / f"{partition}.parquet"
    temporary = parquet.with_name(parquet.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(parquet)
    csv = frame.copy()
    for column in csv:
        if csv[column].dtype == object:
            csv[column] = csv[column].map(
                lambda value: json.dumps(value.tolist() if isinstance(value, np.ndarray) else value,
                                         ensure_ascii=False, default=_default)
                if isinstance(value, (list, tuple, dict, np.ndarray)) else value
            )
    csv.to_csv(output / f"{partition}.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--test-fraction", type=float)
    parser.add_argument("--author-validation-fraction", type=float)
    parser.add_argument("--near-duplicate-threshold", type=int)
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()
    config = {}
    if args.config:
        import yaml

        config = yaml.safe_load(args.config.read_text()) or {}
        config = config.get("splits", config)
    validation_fraction = args.validation_fraction if args.validation_fraction is not None else config.get("validation_fraction", 0.15)
    test_fraction = args.test_fraction if args.test_fraction is not None else config.get("test_fraction", 0.15)
    author_validation_fraction = args.author_validation_fraction if args.author_validation_fraction is not None else config.get("author_validation_fraction", 0.2)
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    threshold = args.near_duplicate_threshold if args.near_duplicate_threshold is not None else config.get("near_duplicate_threshold", 4)
    frames = [pd.read_parquet(path) for path in args.manifest]
    data = pd.concat(frames, ignore_index=True)
    cv2.setNumThreads(1)
    result = build_composite_splits(data, validation_fraction=validation_fraction, seed=seed,
                                    near_duplicate_threshold=threshold,
                                    test_fraction=test_fraction, author_validation_fraction=author_validation_fraction)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for partition in ("train", "validate", "test", "excluded"):
        _write_manifest(result[partition], output, partition)
    report = result["report"]
    report["input_manifests"] = [str(path.resolve()) for path in args.manifest]
    report["ready_for_training"] = bool(len(result["train"]) and len(result["validate"]) and len(result["test"]))
    (output / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_default) + "\n")
    pd.DataFrame(report["balance"]).to_json(output / "balance.json", orient="records", force_ascii=False, indent=2)
    print(json.dumps({key: report[key] for key in ("train_rows", "validate_rows", "test_rows", "excluded_rows", "ready_for_training")}))
    if not report["ready_for_training"] and not args.allow_empty:
        print("Empty train, validation or test partition. Supply the required source partitions or explicitly use --allow-empty for an audit-only run.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
