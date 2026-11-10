from io import BytesIO

import numpy as np
from PIL import Image

from src.config import PROCESSED_DIR, RAW_DIR

SIZE = 256
RADII = np.arange(1, SIZE // 2)
WINDOW = np.outer(np.hanning(SIZE), np.hanning(SIZE))
_yy, _xx = np.indices((SIZE, SIZE))
RADIUS = np.round(np.hypot(_yy - SIZE // 2, _xx - SIZE // 2)).astype(int)


def load_gray(path, jpeg_quality=None):
    with Image.open(RAW_DIR / path) as im:
        rgb = im.convert("RGB")
    if jpeg_quality:
        buffer = BytesIO()
        rgb.save(buffer, format="JPEG", quality=jpeg_quality)
        rgb = Image.open(buffer).convert("RGB")
    return np.asarray(rgb.convert("L"), dtype=np.float64)


def log_spectrum(gray):
    windowed = (gray - gray.mean()) * WINDOW
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(windowed))) + 1e-8)


def radial_profile(gray):
    spectrum = log_spectrum(gray)
    sums = np.bincount(RADIUS.ravel(), weights=spectrum.ravel())
    counts = np.bincount(RADIUS.ravel())
    return (sums / counts)[RADII]


def build_fft(paths, name, jpeg_quality=None):
    paths = list(paths)
    cache = PROCESSED_DIR / f"fft_{name}.npz"
    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        if saved["paths"].tolist() == paths:
            return saved["profiles"]

    profiles = np.array([radial_profile(load_gray(p, jpeg_quality)) for p in paths], dtype=np.float32)
    np.savez(cache, paths=np.array(paths), profiles=profiles)
    return profiles
