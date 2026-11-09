#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/prepare_composite.sh
python scripts/download_assets.py datasets --datasets openforensics_crops
for notebook in notebooks/eda_140k.ipynb notebooks/eda_mffi.ipynb notebooks/eda_openforensics.ipynb notebooks/eda_df40.ipynb; do
    python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=deepfake-eda --ExecutePreprocessor.timeout=-1 "$notebook"
done
