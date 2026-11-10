import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = Path(os.environ.get("DEEPFAKE_DATA_ROOT", ROOT / "data" / "raw")).expanduser()
if not RAW.is_absolute():
    RAW = ROOT / RAW
DATA = RAW / "openforensics"
CACHE = ROOT / "data" / "processed" / "openforensics"
RESULTS = ROOT / "results" / "openforensics"
SPLITS = ["Train", "Val", "Test-Dev"]
SEED = 42
PLOT_COLOR = {"real": "#2878b5", "fake": "#e66b3d"}
SPLIT_COLOR = {"Train": "#2878b5", "Val": "#e69f00", "Test-Dev": "#7b5aa6"}
BOX_RGB = {"real": (40, 190, 90), "fake": (235, 65, 65)}
ANNOT = DATA / "annotations"
IMAGES = DATA / "images"
_IMAGE_DIRS = {}


def annotation_path(split):
    preferred = ANNOT / f"{split}_poly.json"
    if preferred.exists():
        return preferred
    candidates = sorted(DATA.rglob(f"{split}_poly.json"))
    if not candidates:
        raise FileNotFoundError(f"Missing {split}_poly.json under {DATA}; run scripts/download_datasets.sh --datasets openforensics")
    return candidates[0]


def image_path(split, filename):
    filename = Path(filename)
    candidates = [IMAGES / split / filename.name, DATA / split / filename.name,
                  IMAGES / filename, DATA / filename]
    if split not in _IMAGE_DIRS:
        _IMAGE_DIRS[split] = sorted(directory for directory in DATA.rglob(split) if directory.is_dir())
    for directory in _IMAGE_DIRS[split]:
        candidates.extend([directory / filename.name, directory / "images" / filename.name])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def load_all():
    global ANNOT, IMAGES
    for directory in [CACHE, RESULTS]:
        directory.mkdir(parents=True, exist_ok=True)
    frame_parts, face_parts = [], []
    for split in SPLITS:
        annotation = annotation_path(split)
        ANNOT = annotation.parent
        payload = json.loads(annotation.read_text())
        frame = pd.DataFrame(payload["images"]).rename(columns={"id": "image_id"})
        frame["split"] = split
        frame["path"] = [str(image_path(split, name)) for name in frame["file_name"]]
        if len(frame):
            IMAGES = Path(frame.iloc[0]["path"]).parent.parent
        population_frames = len(frame)
        population_faces = len(payload["annotations"])
        frame = frame[frame["path"].map(lambda path: Path(path).is_file())].copy()
        face = pd.DataFrame(payload["annotations"]).rename(columns={"id": "face_id"})
        face = face[face["image_id"].isin(frame["image_id"])].copy()
        print(f"OpenForensics {split}: available {len(frame):,}/{population_frames:,} frames, {len(face):,}/{population_faces:,} annotated faces", flush=True)
        face["split"] = split
        face[["x", "y", "w", "h"]] = pd.DataFrame(face["bbox"].tolist(), index=face.index)
        if not face["category_id"].isin([0, 1]).all():
            raise ValueError("OpenForensics category_id must use 0=real and 1=fake")
        face["label"] = face["category_id"].map({0: "real", 1: "fake"})
        sizes = frame.set_index("image_id")[["width", "height"]]
        face["cx"] = (face["x"] + face["w"] / 2) / face["image_id"].map(sizes["width"])
        face["cy"] = (face["y"] + face["h"] / 2) / face["image_id"].map(sizes["height"])
        frame_parts.append(frame)
        face_parts.append(face)
    return pd.concat(frame_parts, ignore_index=True), pd.concat(face_parts, ignore_index=True)


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def polygon_mask(segmentation, height, width):
    mask = np.zeros((height, width), np.uint8)
    polygons = [np.rint(np.asarray(p).reshape(-1, 2)).astype(np.int32)
                for p in segmentation if len(p) >= 6]
    if polygons:
        cv2.fillPoly(mask, polygons, 1)
    return mask


def polygon_area(segmentation):
    total = 0.0
    for polygon in segmentation:
        points = np.asarray(polygon, dtype=float).reshape(-1, 2)
        if len(points) >= 3:
            total += abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) -
                         np.dot(points[:, 1], np.roll(points[:, 0], 1))) / 2
    return total


def square_box(x, y, width, height, scale=1.3):
    side = scale * np.maximum(width, height)
    return x + width / 2 - side / 2, y + height / 2 - side / 2, side


def crop_face(image, x, y, width, height, scale=1.3, out=256):
    left, top, side = square_box(x, y, width, height, scale)
    left, top = int(np.floor(left)), int(np.floor(top))
    side = max(1, int(np.ceil(side)))
    shape = (side, side) + image.shape[2:]
    crop = np.full(shape, 127, dtype=image.dtype)
    h, w = image.shape[:2]
    x0, y0, x1, y1 = max(left, 0), max(top, 0), min(left + side, w), min(top + side, h)
    if x1 > x0 and y1 > y0:
        crop[y0 - top:y1 - top, x0 - left:x1 - left] = image[y0:y1, x0:x1]
    interpolation = cv2.INTER_AREA if side > out else cv2.INTER_CUBIC
    return cv2.resize(crop, (out, out), interpolation=interpolation)


def draw_faces(image, faces, masks=True, letters=True):
    result = image.copy()
    thickness = max(1, round(max(image.shape[:2]) / 350))
    for _, face in faces.iterrows():
        color = BOX_RGB[face["label"]]
        if masks:
            mask = polygon_mask(face["segmentation"], *image.shape[:2]).astype(bool)
            result[mask] = (0.75 * result[mask] + 0.25 * np.asarray(color)).astype(np.uint8)
        x, y, w, h = [int(round(face[c])) for c in ["x", "y", "w", "h"]]
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        if letters:
            cv2.putText(result, face["label"][0].upper(), (x, max(15, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.4, thickness / 2), color, thickness)
    return result


def dhash(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def sample_rows(data, count, random_state=42):
    return data.sample(min(count, len(data)), random_state=random_state)


def cache_fingerprint(frames, faces, split):
    frame_keys = frames.loc[frames["split"] == split, ["image_id", "file_name"]]
    face_keys = faces.loc[faces["split"] == split, ["face_id", "image_id", "category_id", "x", "y", "w", "h"]]
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(frame_keys, index=False).values.tobytes())
    digest.update(pd.util.hash_pandas_object(face_keys, index=False).values.tobytes())
    return digest.hexdigest()


def mark_cache(name, split, frames, faces):
    marker = CACHE / f"{name}_{split}.source.json"
    marker.write_text(json.dumps({"schema": 1, "fingerprint": cache_fingerprint(frames, faces, split)}))


def cache_ready(name, split, frames, faces):
    marker = CACHE / f"{name}_{split}.source.json"
    required = [CACHE / f"{name}_{split}.parquet"]
    if name == "scan_frames":
        required.extend([CACHE / f"scan_faces_{split}.parquet", CACHE / f"thumbs_{split}.npy", CACHE / f"face_thumbs_{split}.npy"])
    if not marker.exists() or not all(path.exists() for path in required):
        return False
    metadata = json.loads(marker.read_text())
    return metadata.get("schema") == 1 and metadata.get("fingerprint") == cache_fingerprint(frames, faces, split)
