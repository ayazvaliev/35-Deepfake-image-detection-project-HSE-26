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
    E.style()
    result = E.refresh_pixel_distribution_cache(ROOT, pd.read_parquet(metadata), metadata)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
