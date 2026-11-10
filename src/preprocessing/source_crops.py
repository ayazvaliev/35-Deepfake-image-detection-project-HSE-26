from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np


SOURCE_CROP_VERSION = 1


def _pixel_sha256(image):
    image = np.ascontiguousarray(image)
    return hashlib.sha256(f"{image.dtype}:{image.shape}:".encode() + image.tobytes()).hexdigest()


@lru_cache(maxsize=8)
def _decode_source(content):
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Source crop original cannot be decoded as BGR pixels")
    return image


def _source_pixels(filename, expected_sha256):
    content = Path(filename).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("Source crop original file SHA256 mismatch")
    return _decode_source(content)


def reconstruct_source_crop(image, bounds, size, padding_value=127):
    if len(bounds) != 4 or any(not np.isfinite(value) or float(value) != int(value) for value in bounds):
        raise ValueError("Source crop bounds must be four finite integers")
    left, top, width, height = map(int, bounds)
    size = int(size)
    if width != height or width < 1 or size < 1 or width > 2 * max(image.shape[:2]) + 256:
        raise ValueError("Invalid source crop square or output size")
    crop = np.full((height, width, 3), int(padding_value), dtype=np.uint8)
    x1, y1 = max(0, left), max(0, top)
    x2, y2 = min(image.shape[1], left + width), min(image.shape[0], top + height)
    if x2 > x1 and y2 > y1:
        crop[y1 - top:y2 - top, x1 - left:x2 - left] = image[y1:y2, x1:x2]
    if width != size:
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA if width > size else cv2.INTER_CUBIC)
    return crop


def _descriptor_crop(descriptor, source_pixels=None):
    if descriptor.get("schema_version") != SOURCE_CROP_VERSION or descriptor.get("kind") != "exact_source_crop":
        raise ValueError("Unsupported source crop descriptor")
    if descriptor.get("padding_value") != 127 or descriptor.get("downsampling") != "INTER_AREA" or descriptor.get("upsampling") != "INTER_CUBIC":
        raise ValueError("Source crop pixel policy differs from preprocessing")
    image = source_pixels if source_pixels is not None else _source_pixels(
        descriptor["source_path"], descriptor["source_sha256"],
    )
    if list(image.shape) != descriptor["source_shape"]:
        raise ValueError("Source crop original dimensions differ from the descriptor")
    return reconstruct_source_crop(image, descriptor["bounds"], descriptor["size"], descriptor["padding_value"])


def save_source_crop(crop, bounds, original_metadata, directory, source_pixels=None):
    if original_metadata.get("original_verification_codec") != "source_bytes":
        return None
    if len(bounds) != 4 or any(not np.isfinite(value) or float(value) != int(value) for value in bounds):
        raise ValueError("Source crop bounds must retain the exact preprocessing integers")
    source_path = str(original_metadata["original_verification_path"])
    source_sha = str(original_metadata["original_verification_sha256"])
    image = source_pixels if source_pixels is not None else _source_pixels(source_path, source_sha)
    descriptor = {
        "schema_version": SOURCE_CROP_VERSION, "kind": "exact_source_crop",
        "source_path": source_path, "source_sha256": source_sha,
        "source_shape": list(image.shape), "bounds": [int(value) for value in bounds],
        "size": int(crop.shape[0]), "padding_value": 127,
        "downsampling": "INTER_AREA", "upsampling": "INTER_CUBIC",
        "pixel_sha256": _pixel_sha256(crop),
    }
    reconstructed = _descriptor_crop(descriptor, source_pixels=image)
    if not np.array_equal(reconstructed, crop):
        raise ValueError("Source crop reconstruction changes preprocessing pixels")
    content = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(content).hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (digest + ".source_crop.json")
    if not destination.exists():
        with tempfile.NamedTemporaryFile(dir=directory, prefix=digest + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.replace(destination)
    elif hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
        raise ValueError("Existing source crop descriptor fails file integrity")
    return {
        "crop_verification_path": str(destination), "crop_verification_sha256": digest,
        "crop_verification_pixel_sha256": descriptor["pixel_sha256"],
        "crop_verification_codec": "source_crop", "crop_verification_size": descriptor["size"],
        "crop_verification_policy_version": SOURCE_CROP_VERSION,
        "crop_verification_bytes": len(content), "crop_verification_aliases_original": False,
    }


def read_source_crop(path, file_sha256, pixel_sha256, size):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != file_sha256:
        raise ValueError("Source crop descriptor file SHA256 mismatch")
    descriptor = json.loads(content)
    if descriptor.get("pixel_sha256") != pixel_sha256 or descriptor.get("size") != int(size):
        raise ValueError("Source crop descriptor differs from the recorded pixel identity")
    crop = _descriptor_crop(descriptor)
    if _pixel_sha256(crop) != pixel_sha256:
        raise ValueError("Source crop reconstructed pixel SHA256 mismatch")
    return crop
