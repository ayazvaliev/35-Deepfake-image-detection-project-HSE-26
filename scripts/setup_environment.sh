#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
conda_root=${CONDA_ROOT:-"$HOME/miniconda3"}
if [ ! -x "$conda_root/bin/conda" ]; then
    installer_dir=$(mktemp -d)
    trap 'rm -rf "$installer_dir"' EXIT
    python3 - "$installer_dir" <<'PY'
import hashlib
import platform
import re
import sys
import urllib.request
from pathlib import Path

system = {"Linux": "Linux", "Darwin": "MacOSX"}[platform.system()]
arch = {"x86_64": "x86_64", "arm64": "arm64", "aarch64": "aarch64"}[platform.machine()]
name = f"Miniconda3-latest-{system}-{arch}.sh"
index = urllib.request.urlopen("https://repo.anaconda.com/miniconda/", timeout=60).read().decode()
row = next(row for row in re.findall(r"<tr>(.*?)</tr>", index, re.S) if f">{name}<" in row)
expected = re.findall(r"\b[a-f0-9]{64}\b", row)[0]
path = Path(sys.argv[1]) / "miniconda.sh"
urllib.request.urlretrieve(f"https://repo.anaconda.com/miniconda/{name}", path)
with path.open("rb") as stream:
    actual = hashlib.sha256(stream.read()).hexdigest()
if actual != expected:
    raise RuntimeError("Miniconda SHA256 mismatch")
PY
    bash "$installer_dir/miniconda.sh" -b -p "$conda_root"
    rm -rf "$installer_dir"
    trap - EXIT
fi
condarc=$(mktemp)
trap 'rm -f "$condarc"' EXIT
printf "channels:\n  - conda-forge\ndefault_channels: []\n" > "$condarc"
CONDA_NO_PLUGINS=true CONDA_SOLVER=classic CONDARC="$condarc" "$conda_root/bin/conda" env update -f environment.yml
"$conda_root/bin/conda" run -n deepfake python -m pip check
"$conda_root/bin/conda" run -n deepfake python -m ipykernel install --user --name deepfake --display-name 'Python (deepfake)'
