import math
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "notebook_src"

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

from .of_data import CACHE, ROOT, SPLITS, crop_face, load_all, mark_cache, polygon_mask, read_rgb
from .of_dedup import iou


FIELDS = ["face_id", "pose_ok", "yaw", "pitch", "roll", "yn_ok", "yunet_found", "yn_yaw_proxy",
          "face_L_mean", "face_L_std", "bg_L_mean", "lr_asym", "overexp", "underexp", "ita",
          "sharp_lap", "noise_est", "hf_ratio"]


def landmarker():
    directory = Path(os.environ.get("DEEPFAKE_MODEL_ROOT", ROOT / "models")).expanduser()
    if not directory.is_absolute():
        directory = ROOT / directory
    model = directory / "face_landmarker.task"
    if not model.exists():
        raise FileNotFoundError("Missing face_landmarker.task; run bash scripts/download_models.sh")
    options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model)),
                                   running_mode=RunningMode.IMAGE, num_faces=1,
                                   output_facial_transformation_matrixes=True)
    return FaceLandmarker.create_from_options(options)


def attributes(image, face, model, detections):
    rgb = crop_face(image, face["x"], face["y"], face["w"], face["h"], out=256)
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    full_mask = polygon_mask(face["segmentation"], *image.shape[:2]) * 255
    mask = crop_face(full_mask, face["x"], face["y"], face["w"], face["h"], out=256) > 127
    valid = crop_face(np.full(image.shape[:2], 255, np.uint8), face["x"], face["y"], face["w"], face["h"], out=256) > 127
    lightness = lab[:, :, 0]
    values = lightness[mask]
    background = lightness[valid & ~mask]
    left = lightness[:, :128][mask[:, :128]]
    right = lightness[:, 128:][mask[:, 128:]]
    small = cv2.cvtColor(cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float64)
    lap = cv2.Laplacian(small, cv2.CV_64F)
    residual = small - cv2.GaussianBlur(small, (0, 0), 1)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(small - small.mean()))) ** 2
    yy, xx = np.indices(small.shape)
    radius = np.sqrt((xx - 64) ** 2 + (yy - 64) ** 2)
    row = {"face_id": face["face_id"], "pose_ok": False,
           "yaw": np.nan, "pitch": np.nan, "roll": np.nan,
           "yn_ok": False, "yunet_found": False, "yn_yaw_proxy": np.nan,
           "face_L_mean": float(values.mean()) if values.size else np.nan,
           "face_L_std": float(values.std()) if values.size else np.nan,
           "bg_L_mean": float(background.mean()) if background.size else np.nan,
           "lr_asym": float(left.mean() - right.mean()) if left.size and right.size else np.nan,
           "overexp": float((values >= 95).mean()) if values.size else np.nan,
           "underexp": float((values <= 5).mean()) if values.size else np.nan,
           "ita": np.nan, "sharp_lap": float(lap.var()),
           "noise_est": float(np.median(np.abs(residual - np.median(residual))) / 0.6745),
           "hf_ratio": float(spectrum[radius > 128 / 3].sum() / max(spectrum.sum(), 1e-12))}
    if detections is not None and len(detections):
        boxes = detections[["det_x", "det_y", "det_w", "det_h"]].to_numpy(float)
        overlaps = np.array([iou(face[["x", "y", "w", "h"]], box) for box in boxes])
        best = int(overlaps.argmax())
        if overlaps[best] >= 0.2:
            found = detections.iloc[best]
            midpoint = (found["eye_lx"] + found["eye_rx"]) / 2
            distance = abs(found["eye_lx"] - found["eye_rx"])
            row.update(yn_ok=True, yunet_found=True,
                       yn_yaw_proxy=float((found["nose_x"] - midpoint) / max(distance, 1e-6)))
    result = model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    if result.face_landmarks:
        rotation = np.asarray(result.facial_transformation_matrixes[0])[:3, :3]
        u, _, vt = np.linalg.svd(rotation)
        rotation = u @ vt
        row.update(pose_ok=True, yaw=math.degrees(math.asin(-np.clip(rotation[2, 0], -1, 1))),
                   pitch=math.degrees(math.atan2(rotation[2, 1], rotation[2, 2])),
                   roll=math.degrees(math.atan2(rotation[1, 0], rotation[0, 0])))
        landmarks = result.face_landmarks[0]
        cheeks = np.zeros((256, 256), np.uint8)
        for indices in [[50, 101, 205], [280, 330, 425]]:
            points = np.array([(landmarks[i].x, landmarks[i].y) for i in indices]) * 256
            center = np.rint(points.mean(axis=0)).astype(int)
            cv2.circle(cheeks, tuple(center), 8, 1, -1)
        cheek_mask = cheeks.astype(bool) & mask
        if cheek_mask.any():
            lmean, bmean = lab[:, :, 0][cheek_mask].mean(), lab[:, :, 2][cheek_mask].mean()
            row["ita"] = math.degrees(math.atan2(float(lmean - 50), float(bmean)))
            if row["ita"] > 90:
                row["ita"] -= 180
            elif row["ita"] < -90:
                row["ita"] += 180
    return row


def scan(split):
    frames, faces = load_all()
    frames, faces = frames[frames["split"] == split], faces[faces["split"] == split]
    paths = frames.set_index("image_id")["path"].to_dict()
    detections = pd.read_parquet(CACHE / f"yunet_frames_{split}.parquet")
    detection_groups = dict(tuple(detections.groupby("image_id", sort=False)))
    rows = []
    with landmarker() as model:
        for image_id, group in faces.groupby("image_id", sort=False):
            image = read_rgb(paths[image_id])
            found = detection_groups.get(image_id)
            for _, face in group.iterrows():
                rows.append(attributes(image, face, model, found))
    table = pd.DataFrame(rows, columns=FIELDS).set_index("face_id").loc[faces["face_id"]].reset_index()
    table.to_parquet(CACHE / f"attr_raw_{split}.parquet", index=False)
    mark_cache("attr_raw", split, frames, faces)
    print(f"Attributes {split}: {len(rows)} faces", flush=True)


if __name__ == "__main__":
    for selected in sys.argv[1:] or SPLITS:
        scan(selected)
