import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import pandas as pd

from src.notebook_src.of_data import CACHE, ROOT, SPLITS, load_all, mark_cache, read_rgb


COLUMNS = ["image_id", "det_x", "det_y", "det_w", "det_h", "det_score", "covered",
           "eye_rx", "eye_ry", "eye_lx", "eye_ly", "nose_x", "nose_y"]


def detector():
    directory = Path(os.environ.get("DEEPFAKE_MODEL_ROOT", ROOT / "models"))
    candidates = sorted(directory.glob("*yunet*.onnx"))
    if not candidates:
        raise FileNotFoundError("Missing YuNet ONNX; run bash scripts/download_models.sh")
    return cv2.FaceDetectorYN.create(str(candidates[0]), "", (320, 320), 0.8, 0.3, 5000)


def detect(model, rgb):
    h, w = rgb.shape[:2]
    model.setInputSize((w, h))
    _, faces = model.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return np.empty((0, 15), dtype=float) if faces is None else faces


def scan(split):
    frames, faces = load_all()
    frames, faces = frames[frames["split"] == split], faces[faces["split"] == split]
    groups = dict(tuple(faces.groupby("image_id", sort=False)))
    model, rows = detector(), []
    for frame in frames.to_dict("records"):
        found = detect(model, read_rgb(frame["path"]))
        annotated = groups[frame["image_id"]][["x", "y", "w", "h"]].to_numpy(float)
        for box in found:
            x, y, w, h = [float(value) for value in box[:4]]
            iw = np.maximum(0, np.minimum(x + w, annotated[:, 0] + annotated[:, 2]) - np.maximum(x, annotated[:, 0]))
            ih = np.maximum(0, np.minimum(y + h, annotated[:, 1] + annotated[:, 3]) - np.maximum(y, annotated[:, 1]))
            covered = float((iw * ih).max() / max(w * h, 1e-12))
            rows.append([frame["image_id"], x, y, w, h, float(box[-1]), covered, *map(float, box[4:10])])
    pd.DataFrame(rows, columns=COLUMNS).to_parquet(CACHE / f"yunet_frames_{split}.parquet", index=False)
    mark_cache("yunet_frames", split, frames, faces)
    print(f"YuNet {split}: {len(rows)} faces", flush=True)


if __name__ == "__main__":
    for selected in sys.argv[1:] or SPLITS:
        scan(selected)
