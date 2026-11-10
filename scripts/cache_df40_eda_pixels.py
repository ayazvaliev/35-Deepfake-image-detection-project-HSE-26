from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.notebook_src import df40_eda as E


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/processed/full/df40/metadata.parquet")
    args = parser.parse_args()
    metadata = args.metadata if args.metadata.is_absolute() else ROOT / args.metadata
    faces = pd.read_parquet(metadata)
    E.style()
    result = E.pixel_illustrations(ROOT, faces, metadata)
    if result["mode"] != "fresh_pixels":
        raise RuntimeError("Fresh raw originals and identical face crops are required before source cleanup")
    print(json.dumps({
        "mode": result["mode"], "metadata": str(metadata),
        "metadata_sha256": result["manifest"]["metadata_sha256"],
        "quality_sample_size": result["manifest"]["quality_sample_size"],
        "figures": {name: str(path) for name, path in result["figure_paths"].items()},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
