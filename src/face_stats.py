import math

import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from PIL import Image, ImageCms
from scipy.stats import ks_2samp

from src.config import PROCESSED_DIR, RAW_DIR, ROOT, SEED

MODEL_PATH = ROOT / "models" / "face_landmarker.task"
FIELDS = ["n_faces", "face_area", "face_cx", "face_cy", "yaw", "roll", "brightness", "bg_brightness"]
IMAGE_FIELDS = ["sharpness", "chroma", "is_gray", "face_sharpness", "bg_sharpness"]
# preliminary threshold on mean LAB chroma for near-grayscale images
GRAY_CHROMA = 3
TO_LAB = ImageCms.buildTransform(ImageCms.createProfile("sRGB"), ImageCms.createProfile("LAB"), "RGB", "LAB")


def face_sample(index):
    train_valid = index[index["split"] != "test"]
    big = train_valid[train_valid["dataset"] == "140k"].groupby("label").sample(5000, random_state=SEED)
    of = train_valid[train_valid["dataset"] == "openforensics"]
    return pd.concat([big, of], ignore_index=True)


def face_box(landmarks):
    xs = np.clip([p.x for p in landmarks], 0, 1)
    ys = np.clip([p.y for p in landmarks], 0, 1)
    return xs.min(), ys.min(), xs.max(), ys.max()


def laplacian(x):
    return x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:] - 4 * x[1:-1, 1:-1]


def image_face_stats(path, landmarker):
    with Image.open(path) as im:
        rgb = im.convert("RGB")
    # Pillow stores L of LAB as 0..255, rescale to 0..100
    lab = ImageCms.applyTransform(rgb, TO_LAB)
    lightness = np.array(lab.getchannel("L"), dtype=np.float64) * 100 / 255
    # a and b are stored with +128 offset
    a = np.array(lab.getchannel("A"), dtype=np.float64) - 128
    b = np.array(lab.getchannel("B"), dtype=np.float64) - 128
    chroma = np.sqrt(a**2 + b**2).mean()
    lap = laplacian(lightness)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(rgb)))
    stats = {
        "n_faces": len(result.face_landmarks),
        "brightness": lightness.mean(),
        "sharpness": lap.var(),
        "chroma": chroma,
        "is_gray": chroma < GRAY_CHROMA,
    }
    if not result.face_landmarks:
        return stats

    boxes = [face_box(face) for face in result.face_landmarks]
    areas = [(x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes]
    i = int(np.argmax(areas))
    x0, y0, x1, y1 = boxes[i]

    h, w = lightness.shape
    outside = np.ones((h, w), dtype=bool)
    outside[round(y0 * h):round(y1 * h), round(x0 * w):round(x1 * w)] = False

    rot = np.array(result.facial_transformation_matrixes[i])[:3, :3]
    stats.update({
        "face_area": areas[i],
        "face_cx": (x0 + x1) / 2,
        "face_cy": (y0 + y1) / 2,
        "yaw": math.degrees(math.asin(-np.clip(rot[2, 0], -1, 1))),
        "roll": math.degrees(math.atan2(rot[1, 0], rot[0, 0])),
        "bg_brightness": lightness[outside].mean() if outside.any() else np.nan,
        "face_sharpness": lap[~outside[1:-1, 1:-1]].var(),
        "bg_sharpness": lap[outside[1:-1, 1:-1]].var() if outside[1:-1, 1:-1].any() else np.nan,
    })
    return stats


def build_face_stats():
    sample = face_sample(pd.read_csv(PROCESSED_DIR / "index.csv"))
    cache = PROCESSED_DIR / "face_stats.csv"
    if cache.exists():
        stats = pd.read_csv(cache)
        fresh = stats["path"].tolist() == sample["path"].tolist()
        if fresh and set(FIELDS + IMAGE_FIELDS) <= set(stats.columns):
            return stats

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_faces=5,
        output_facial_transformation_matrixes=True,
    )
    with FaceLandmarker.create_from_options(options) as landmarker:
        rows = [image_face_stats(RAW_DIR / p, landmarker) for p in sample["path"]]
    stats = pd.concat([sample, pd.DataFrame(rows, columns=FIELDS + IMAGE_FIELDS)], axis=1)
    stats.to_csv(cache, index=False)
    return stats


def ks_table(df, fields):
    rows = []
    for ds, group in df.groupby("dataset"):
        for field in fields:
            real = group.loc[group["label"] == 0, field].dropna()
            fake = group.loc[group["label"] == 1, field].dropna()
            ks = ks_2samp(real, fake)
            rows.append({
                "dataset": ds, "field": field, "ks_d": round(ks.statistic, 3), "p_value": ks.pvalue,
                "n_real": len(real), "n_fake": len(fake),
            })
    return pd.DataFrame(rows)
