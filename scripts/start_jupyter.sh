#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
conda_root=${CONDA_ROOT:-"$HOME/miniconda3"}
exec "$conda_root/bin/conda" run --no-capture-output -n deepfake-eda python scripts/start_jupyter.py --port="${JUPYTER_PORT:-8888}"
