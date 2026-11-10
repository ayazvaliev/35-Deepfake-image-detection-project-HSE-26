import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .config import PROCESSED_DIR, RAW_DIR, ROOT
from .duplicates import file_list

YUNET_PATH = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
SCORE_THRESHOLD = 0.9
MIN_FACE = 32
MARGIN = 1.3
OUT = 256
COLUMNS = ["path", "x0", "y0", "side", "scale", "shifted", "excluded", "reason"]


def make_detector():
    return cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (OUT, OUT), SCORE_THRESHOLD)


def read_rgb(path):
    with Image.open(RAW_DIR / path) as im:
        return np.asarray(im.convert("RGB"))


def largest_face(rgb, detector):
    h, w = rgb.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(np.ascontiguousarray(rgb[..., ::-1]))
    if faces is None:
        return None
    return max(faces[:, :4], key=lambda f: f[2] * f[3])


def crop_params(rgb, detector):
    h, w = rgb.shape[:2]
    face = largest_face(rgb, detector)
    if face is None:
        return {"excluded": True, "reason": "no_face"}

    x, y, fw, fh = face
    if max(fw, fh) < MIN_FACE:
        return {"excluded": True, "reason": "small_face"}
    side = round(MARGIN * max(fw, fh))
    if side > min(w, h):
        return {"excluded": True, "reason": "too_big"}

    x0 = round(x + fw / 2 - side / 2)
    y0 = round(y + fh / 2 - side / 2)
    x0_in = min(max(x0, 0), w - side)
    y0_in = min(max(y0, 0), h - side)
    return {
        "x0": x0_in, "y0": y0_in, "side": side, "scale": OUT / side,
        "shifted": (x0_in, y0_in) != (x0, y0), "excluded": False, "reason": "",
    }


def build_crops():
    files = file_list()
    cache = PROCESSED_DIR / "crops.csv"
    if cache.exists():
        crops = pd.read_csv(cache, keep_default_na=False, na_values=[""])
        if crops["path"].tolist() == files["path"].tolist() and list(crops.columns) == COLUMNS:
            return crops

    detector = make_detector()
    rows = [crop_params(read_rgb(p), detector) for p in files["path"]]
    crops = pd.concat([files[["path"]], pd.DataFrame(rows)], axis=1)[COLUMNS]
    crops.to_csv(cache, index=False)
    return crops


def load_crop(path, crop):
    x0, y0, side = int(crop["x0"]), int(crop["y0"]), int(crop["side"])
    part = read_rgb(path)[y0:y0 + side, x0:x0 + side]
    interpolation = cv2.INTER_AREA if side > OUT else cv2.INTER_CUBIC
    return Image.fromarray(cv2.resize(part, (OUT, OUT), interpolation=interpolation))
