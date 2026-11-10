from io import BytesIO

import numpy as np
from PIL import Image

from src.config import PROCESSED_DIR, RAW_DIR

SIZE = 256
RADII = np.arange(1, SIZE // 2)
WINDOW = np.outer(np.hanning(SIZE), np.hanning(SIZE))
_yy, _xx = np.indices((SIZE, SIZE))
RADIUS = np.round(np.hypot(_yy - SIZE // 2, _xx - SIZE // 2)).astype(int)
# spectrum is symmetric, so opposite sectors are merged: angle modulo 180
_angle = np.degrees(np.arctan2(-(_yy - SIZE // 2), _xx - SIZE // 2)) % 180
SECTOR = np.round(_angle / 45).astype(int) % 4
DIR_INDEX = (SECTOR * SIZE + RADIUS).ravel()
DIR_COUNTS = np.bincount(DIR_INDEX, minlength=4 * SIZE)
DIR_KEEP = (np.arange(4)[:, None] * SIZE + RADII).ravel()


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


def directional_profile(gray):
    sums = np.bincount(DIR_INDEX, weights=log_spectrum(gray).ravel(), minlength=4 * SIZE)
    return sums[DIR_KEEP] / DIR_COUNTS[DIR_KEEP]


def build_fft(paths, name, jpeg_quality=None, directional=False):
    paths = list(paths)
    cache = PROCESSED_DIR / (f"fft_{name}_dir.npz" if directional else f"fft_{name}.npz")
    profile = directional_profile if directional else radial_profile
    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        if saved["paths"].tolist() == paths:
            return saved["profiles"]

    profiles = np.array([profile(load_gray(p, jpeg_quality)) for p in paths], dtype=np.float32)
    np.savez(cache, paths=np.array(paths), profiles=profiles)
    return profiles
