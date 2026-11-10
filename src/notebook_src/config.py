import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = Path(os.environ.get("DEEPFAKE_DATA_ROOT", ROOT / "data" / "raw")).expanduser()
if not RAW_DIR.is_absolute():
    RAW_DIR = ROOT / RAW_DIR
PROCESSED_DIR = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "data" / "splits"

SEED = 42

COLORS = {"real": "#2a78b5", "fake": "#d9534f"}

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
