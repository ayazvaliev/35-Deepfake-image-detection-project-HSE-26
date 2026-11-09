#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
workers=${EDA_WORKERS:-4}
mkdir -p data/processed/manifests
python scripts/download_assets.py models
python scripts/download_df40.py --root "${DEEPFAKE_DATA_ROOT:-data/raw}/df40" --max-images 100000 --seed 42
python scripts/download_assets.py datasets --datasets 140k mffi openforensics --limit 20000 --seed 42
python scripts/preprocess.py --manifest "${DEEPFAKE_DATA_ROOT:-data/raw}/df40/sample.parquet" --output-dir data/processed/df40 --workers "$workers" --crop-policy adaptive
for dataset in 140k openforensics mffi; do
    python scripts/index_datasets.py --dataset "$dataset" --limit 20000 --seed 42 --output "data/processed/manifests/$dataset.jsonl"
    python scripts/preprocess.py --manifest "data/processed/manifests/$dataset.jsonl" --output-dir "data/processed/$dataset" --workers "$workers" --crop-policy adaptive
done
python scripts/build_composite_splits.py --manifest data/processed/df40/metadata.parquet data/processed/140k/metadata.parquet data/processed/openforensics/metadata.parquet data/processed/mffi/metadata.parquet --config configs/composite.yml --output data/splits/composite
