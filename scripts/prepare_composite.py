from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("df40", "140k", "openforensics", "mffi")


def build_plan(config, profile="full", workers=4, raw_root=None, config_path=None, skip_download=False, compact_storage=False, reuse_full=False):
    limits = config["sampling"]["profiles"][profile]
    if set(limits) != set(DATASETS) or any(type(value) is not int or value <= 0 for value in limits.values()):
        raise ValueError("Every dataset must have a positive integer image limit")
    if workers < 1:
        raise ValueError("workers must be positive")
    preprocessing = config.get("preprocessing", {})
    if preprocessing.get("output_format", "PNG") != "PNG":
        raise ValueError("Composite preprocessing requires lossless PNG crops")
    seed = config.get("splits", {}).get("seed", 42)
    configured_raw_root = raw_root or os.environ.get("DEEPFAKE_DATA_ROOT")
    default_raw_root = ROOT / "data/raw" / "demo" if profile == "demo" else ROOT / "data/raw"
    raw_root = Path(configured_raw_root or default_raw_root).expanduser()
    if not raw_root.is_absolute():
        raw_root = ROOT / raw_root
    config_path = Path(config_path or ROOT / "configs/composite.yml").resolve()
    processed = ROOT / "data/processed" / profile
    output = ROOT / "data/splits" / ("composite_demo" if profile == "demo" else "composite_full")
    commands = []

    def command(script, *args):
        commands.append([sys.executable, str(ROOT / "scripts" / script), *map(str, args)])

    if reuse_full:
        if profile != "demo":
            raise ValueError("--reuse-full applies only to the demo profile")
        manifests = [ROOT / "data/processed/full" / dataset / "metadata.parquet" for dataset in DATASETS]
        command("sample_processed.py", "--manifest", *manifests, "--limit", 20000,
                "--seed", seed, "--output-root", processed)
        command("build_composite_splits.py", "--manifest",
                *(processed / dataset / "metadata.parquet" for dataset in DATASETS),
                "--config", config_path, "--output", output, "--group-cache", processed / "verified_groups",
                "--group-cache-policy", "versioned")
        return {"profile": profile, "image_limits": dict(limits), "raw_root": str(raw_root),
                "processed_root": str(processed), "split_output": str(output),
                "reuses_full_preprocessing": True, "commands": commands}
    if not skip_download:
        command("download_assets.py", "models")
        command("download_df40.py", "--root", raw_root / "df40", "--max-images", limits["df40"], "--seed", seed)
    options = {
        "workers": workers, "size": preprocessing.get("size", 256),
        "expansion": preprocessing.get("expansion", 1.3),
        "crop-policy": preprocessing.get("crop_policy", "adaptive"),
        "min-face-size": preprocessing.get("min_face_size", 32),
        "min-mask-fraction": preprocessing.get("min_mask_fraction", 0.25),
        "detection-threshold": preprocessing.get("detection_threshold", 0.9),
    }
    preprocessing_args = [item for key, value in options.items() for item in ("--" + key, str(value))]
    if compact_storage:
        preprocessing_args.append("--compact-verification-crops")
    for dataset in DATASETS:
        if dataset == "openforensics" and not skip_download:
            part_args = ["--root", ROOT, "--raw-root", raw_root, "--preprocess-output", processed / dataset,
                         "--limit", limits[dataset], "--seed", seed, "--workers", workers,
                         "--preprocessing-config", config_path]
            if compact_storage:
                part_args.append("--compact-storage")
            else:
                part_args.append("--keep-raw")
            command("download_openforensics_parts.py", *part_args)
            continue
        if not skip_download and dataset != "df40":
            download_args = ["datasets", "--datasets", dataset, "--limit", limits[dataset], "--seed", seed]
            if dataset == "mffi":
                download_args.append("--mffi-test-only")
            command("download_assets.py", *download_args)
        extension = ".parquet" if dataset == "df40" else ".jsonl"
        manifest = processed / "manifests" / (dataset + extension)
        index_args = ["--dataset", dataset, "--root", raw_root / dataset,
                      "--limit", limits[dataset], "--require-count", "--seed", seed, "--output", manifest]
        if dataset == "mffi":
            index_args.append("--mffi-test-only")
        command("index_datasets.py", *index_args)
        command("preprocess.py", "--manifest", manifest, "--output-dir", processed / dataset, *preprocessing_args)
        if compact_storage:
            if dataset == "df40":
                command("cache_df40_eda_pixels.py", "--metadata", processed / dataset / "metadata.parquet")
            command("release_processed_sources.py", "--metadata", processed / dataset / "metadata.parquet",
                    "--raw-root", raw_root / dataset, "--delete-checkpoint")
    manifests = [processed / dataset / "metadata.parquet" for dataset in DATASETS]
    command("build_composite_splits.py", "--manifest", *manifests, "--config", config_path,
            "--output", output, "--group-cache", processed / "verified_groups",
            "--group-cache-policy", "versioned")
    return {
        "profile": profile, "image_limits": dict(limits), "raw_root": str(raw_root),
        "processed_root": str(processed), "split_output": str(output), "commands": commands,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("full", "demo"), default="full")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/composite.yml")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("EDA_WORKERS", "4")))
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--compact-storage", action="store_true")
    parser.add_argument("--reuse-full", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    plan = build_plan(config, args.profile, args.workers, args.raw_root, args.config,
                      args.skip_download, args.compact_storage, args.reuse_full)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    environment = os.environ.copy()
    environment["DEEPFAKE_DATA_ROOT"] = plan["raw_root"]
    for command in plan["commands"]:
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
