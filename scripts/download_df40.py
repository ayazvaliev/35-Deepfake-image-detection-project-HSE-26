import argparse
import json
import os
import subprocess
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.df40 import acquire_selected, build_inventory, sample_inventory, write_json


def _stream_selection_ready(root, selected):
    needed = set(selected.loc[
        selected["archive_partition"].eq("test")
        & selected["native_method"].isin(["e4e", "pixart"]), "sample_id"
    ])
    checkpoint_path = root / "metadata" / "stream_downloads.json"
    if not needed:
        return True
    if not checkpoint_path.is_file():
        return False
    checkpoint = json.loads(checkpoint_path.read_text())
    recovered = {
        sample_id for archive in checkpoint.values()
        for sample_id in archive.get("files", {})
    }
    return needed <= recovered


def _active_tar_recovery(root):
    processes = Path("/proc")
    if not processes.is_dir():
        return None
    for process in processes.iterdir():
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            arguments = (process / "cmdline").read_bytes().decode().split("\0")
            if not any(Path(value).name == "recover_df40_tar.py" for value in arguments):
                continue
            if "--root" not in arguments:
                continue
            value = Path(arguments[arguments.index("--root") + 1])
            directory = Path(os.readlink(process / "cwd"))
            active_root = value if value.is_absolute() else directory / value
            if active_root.resolve() == root.resolve():
                return int(process.name)
        except (OSError, ValueError, IndexError):
            continue
    return None


def _recover_native_tar(root, selected):
    if _stream_selection_ready(root, selected):
        return
    process = _active_tar_recovery(root)
    if process is not None:
        print(f"Waiting for existing native-image recovery process {process}", flush=True)
        while _active_tar_recovery(root) is not None:
            if _stream_selection_ready(root, selected):
                return
            time.sleep(10)
        if _stream_selection_ready(root, selected):
            return
    helper = Path(__file__).with_name("recover_df40_tar.py")
    if not helper.is_file():
        raise RuntimeError("Missing verified native-image tar recovery helper")
    subprocess.run(
        [sys.executable, str(helper), "--root", str(root.resolve()),
         "--probe-bytes", "8000000000"],
        check=True,
    )
    if not _stream_selection_ready(root, selected):
        raise RuntimeError("Native-image tar recovery did not complete the selected e4e/pixart images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--refresh-inventory", action="store_true")
    parser.add_argument("--archive-mode", choices=["selective", "full"], default="selective")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--backend", choices=["auto", "primary", "hf-verified"], default="auto")
    parser.add_argument("--recover-tar", action=argparse.BooleanOptionalAction, default=True)
    arguments = parser.parse_args()
    if arguments.max_images <= 0:
        parser.error("--max-images must be positive")
    population = build_inventory(arguments.root, arguments.refresh_inventory)
    selected = sample_inventory(population, arguments.max_images, arguments.seed)
    arguments.root.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(arguments.root / "selection.parquet", index=False)
    write_json(arguments.root / "sampling.json", {
        **selected.attrs.get("sampling", {}),
        "scope": "Native EFS/FE still outputs and native FFHQ/CelebA real images only; no FS/FR and no unedited real FF++/CDF video frames.",
        "format_stratification": "Source format inferred from extension during inventory; headers verified during preprocessing.",
        "source_group_policy": "Conditional FE source identifiers are retained. Each EFS generated image and FE output without a documented source identifier is an individual unknown-lineage unit; generation_batch is audit-only, not identity/source grouping. Borrowed frame filenames are not input identities. Native real serial filenames remain method-local.",
        "source_group_semantics_version": 3,
        "author_split_policy": "Combined DF40_all.json exact-path assignments override released archive train/test; per-method held-out JSONs are not mixed into this protocol. Repeated test/val paths retain test.",
        "author_mapping_counts": population["author_metadata_status"].value_counts().to_dict(),
        "author_partition_counts": population["author_split"].value_counts().to_dict(),
        "author_split_inventory": sorted(population["author_split"].unique().tolist()),
        "population_by_stratum": population.groupby(
            ["native_method", "target", "domain", "author_split", "source_format"],
            dropna=False,
        ).size().reset_index(name="n").to_dict("records"),
    })
    print(json.dumps({
        "population": len(population), "selected": len(selected),
        "native_methods": population["native_method"].nunique(),
        "targets": population["target"].value_counts().to_dict(),
    }), flush=True)
    if not arguments.inventory_only:
        try:
            manifest = acquire_selected(arguments.root, selected, arguments.archive_mode, arguments.workers, arguments.backend)
        except RuntimeError:
            status_path = arguments.root / "metadata" / "acquisition_status.json"
            status = json.loads(status_path.read_text()) if status_path.is_file() else {}
            missing = status.get("deferred_sources", [])
            supported = all(message.split(":", 1)[0] in {"test/e4e", "test/pixart"}
                            for message in missing)
            if not arguments.recover_tar or arguments.backend == "primary" or not missing or not supported:
                raise
            print("Recovering the remaining selected native images from a pinned, verified tar source", flush=True)
            _recover_native_tar(arguments.root, selected)
            manifest = acquire_selected(arguments.root, selected, arguments.archive_mode, arguments.workers, arguments.backend)
        print(f"Acquired {len(manifest)} native images: {arguments.root / 'sample.parquet'}", flush=True)


if __name__ == "__main__":
    main()
