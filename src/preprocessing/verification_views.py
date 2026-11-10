from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

import cv2
import numpy as np


ORIGINAL_VERIFICATION_SIZE = 256
VERIFICATION_VIEW_VERSION = 2


def view_pixel_sha256(image):
    image = np.ascontiguousarray(image)
    prefix = f"{image.dtype}:{image.shape}:".encode()
    return hashlib.sha256(prefix + image.tobytes()).hexdigest()


def normalized_original_view(image, size=ORIGINAL_VERIFICATION_SIZE):
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 1:
        raise ValueError("A verification view requires nonempty uint8 BGR pixels")
    interpolation = cv2.INTER_AREA if max(image.shape[:2]) >= size else cv2.INTER_CUBIC
    return cv2.resize(image, (size, size), interpolation=interpolation)


def _lossless_encoding(image):
    for extension, codec, options in (
        (".webp", "webp_lossless", [cv2.IMWRITE_WEBP_QUALITY, 101]),
        (".png", "png_lossless", [cv2.IMWRITE_PNG_COMPRESSION, 9]),
    ):
        try:
            success, encoded = cv2.imencode(extension, image, options)
        except cv2.error:
            continue
        if success and np.array_equal(cv2.imdecode(encoded, cv2.IMREAD_COLOR), image):
            return encoded.tobytes(), extension, codec
    raise OSError("No supported lossless verification codec preserves the exact pixels")


def _write_view(content, directory, key, extension):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (key + extension)
    digest = hashlib.sha256(content).hexdigest()
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == digest:
        return str(destination), digest
    with tempfile.NamedTemporaryFile(dir=directory, prefix=key + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    temporary.replace(destination)
    return str(destination), digest


def save_original_verification_view(image, original_content, directory):
    view = normalized_original_view(image)
    small_jpeg = original_content[:2] == b"\xff\xd8" and len(original_content) <= 64 * 1024
    if small_jpeg:
        decoded = cv2.imdecode(np.frombuffer(original_content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None or not np.array_equal(normalized_original_view(decoded), view):
            raise ValueError("Original source bytes differ from the supplied verification pixels")
        content, extension, codec = original_content, ".source", "source_bytes"
    else:
        content, extension, codec = _lossless_encoding(view)
        if len(original_content) <= len(content):
            decoded = cv2.imdecode(np.frombuffer(original_content, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is not None and np.array_equal(normalized_original_view(decoded), view):
                content, extension, codec = original_content, ".source", "source_bytes"
    source_digest = hashlib.sha256(original_content).hexdigest()
    path, digest = _write_view(content, directory, source_digest, extension)
    return {
        "original_verification_path": path,
        "original_verification_sha256": digest,
        "original_verification_pixel_sha256": view_pixel_sha256(view),
        "original_verification_codec": codec,
        "original_verification_size": ORIGINAL_VERIFICATION_SIZE,
        "original_verification_policy_version": VERIFICATION_VIEW_VERSION,
        "original_aspect_ratio": image.shape[1] / image.shape[0],
        "original_verification_bytes": len(content),
        "original_verification_storage_policy": "prefer_small_source_bytes_otherwise_min",
    }


def save_crop_verification_view(crop, directory, original_metadata=None, crop_bounds=None, source_pixels=None):
    pixel_digest = view_pixel_sha256(crop)
    original_metadata = original_metadata or {}
    if (original_metadata.get("original_verification_pixel_sha256") == pixel_digest
            and original_metadata.get("original_verification_path")):
        return {
            "crop_verification_path": original_metadata["original_verification_path"],
            "crop_verification_sha256": original_metadata["original_verification_sha256"],
            "crop_verification_pixel_sha256": pixel_digest,
            "crop_verification_codec": original_metadata["original_verification_codec"],
            "crop_verification_size": int(crop.shape[0]),
            "crop_verification_policy_version": VERIFICATION_VIEW_VERSION,
            "crop_verification_bytes": original_metadata["original_verification_bytes"],
            "crop_verification_aliases_original": True,
        }
    if crop_bounds is not None and original_metadata.get("original_verification_codec") == "source_bytes":
        from src.preprocessing.source_crops import save_source_crop

        return save_source_crop(crop, crop_bounds, original_metadata, directory, source_pixels=source_pixels)
    content, extension, codec = _lossless_encoding(crop)
    path, digest = _write_view(content, directory, pixel_digest, extension)
    return {
        "crop_verification_path": path, "crop_verification_sha256": digest,
        "crop_verification_pixel_sha256": pixel_digest, "crop_verification_codec": codec,
        "crop_verification_size": int(crop.shape[0]),
        "crop_verification_policy_version": VERIFICATION_VIEW_VERSION,
        "crop_verification_bytes": len(content), "crop_verification_aliases_original": False,
    }


def read_verification_view(path, file_sha256, pixel_sha256, size, codec):
    if codec == "source_crop":
        from src.preprocessing.source_crops import read_source_crop

        if not file_sha256 or not pixel_sha256:
            raise ValueError("Source crop lacks complete integrity metadata")
        return read_source_crop(path, file_sha256, pixel_sha256, size)
    if not file_sha256 or not pixel_sha256 or codec not in {"source_bytes", "webp_lossless", "png_lossless"}:
        raise ValueError("Verification view lacks complete integrity metadata")
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != file_sha256:
        raise ValueError("Verification view file SHA256 mismatch")
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Verification view cannot be decoded")
    if codec == "source_bytes":
        image = normalized_original_view(image, int(size))
    if image.shape != (int(size), int(size), 3) or image.dtype != np.uint8:
        raise ValueError("Verification view dimensions differ from the recorded view")
    if view_pixel_sha256(image) != pixel_sha256:
        raise ValueError("Verification view pixel SHA256 mismatch")
    return image
