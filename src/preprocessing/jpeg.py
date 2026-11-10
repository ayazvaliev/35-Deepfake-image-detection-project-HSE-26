from __future__ import annotations

from functools import lru_cache
import hashlib
from io import BytesIO
import json
import math
from numbers import Integral
from pathlib import Path

from PIL import Image, JpegImagePlugin


JPEG_METADATA_VERSION = 1
JPEG_QUALITY_REFERENCE = "https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/src/jcparam.c"
JPEG_QUALITY_SEMANTICS = "Exact IJG-equivalent quality from all used quantization tables; the original encoder setting is not stored in JPEG."
IJG_LUMA = (
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
)
IJG_CHROMA = (
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
)


@lru_cache(maxsize=400)
def ijg_quantization_table(quality, chroma=False, baseline=True):
    if not isinstance(quality, Integral) or isinstance(quality, bool) or not 1 <= quality <= 100:
        raise ValueError("IJG quality must be an integer from 1 to 100")
    scale = 5000 // quality if quality < 50 else 200 - 2 * quality
    maximum = 255 if baseline else 32767
    standard = IJG_CHROMA if chroma else IJG_LUMA
    return tuple(min(maximum, max(1, (value * scale + 50) // 100)) for value in standard)


def _quality(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer() or not 1 <= numeric <= 100:
        return None
    return int(numeric)


def jpeg_quality_bin(value):
    quality = _quality(value)
    if quality is None:
        return "unknown"
    if quality <= 49:
        return "<=49"
    if quality <= 74:
        return "50-74"
    if quality <= 89:
        return "75-89"
    if quality <= 94:
        return "90-94"
    if quality == 95:
        return "95"
    if quality <= 99:
        return "96-99"
    return "100"


def _empty(status):
    marker = "not_jpeg" if status == "not_jpeg" else "unknown"
    return {
        "source_jpeg_metadata_version": JPEG_METADATA_VERSION,
        "source_jpeg_quality": None, "source_jpeg_quality_level": marker,
        "source_jpeg_quality_bin": marker, "source_jpeg_quality_method": status,
        "source_jpeg_quality_status": status, "source_jpeg_qtable_hash": None,
        "source_jpeg_qtables": None, "source_jpeg_subsampling": marker,
        "source_jpeg_progressive": None, "source_jpeg_metadata_error": None,
    }


def _table_roles(image):
    layers = tuple(getattr(image, "layer", ()))
    if len(layers) == 1:
        return ((int(layers[0][3]), False),)
    component_ids = tuple(layer[0] for layer in layers)
    if component_ids in ((82, 71, 66), (67, 77, 89, 75)):
        return tuple((int(layer[3]), False) for layer in layers)
    if len(layers) == 3:
        return tuple((int(layer[3]), index > 0) for index, layer in enumerate(layers))
    if len(layers) == 4 and image.info.get("adobe_transform") == 2:
        return tuple((int(layer[3]), index in (1, 2)) for index, layer in enumerate(layers))
    return ()


@lru_cache(maxsize=2048)
def _exact_quality(tables, roles):
    lookup = dict(tables)
    if not roles or any(table_id not in lookup for table_id, _ in roles):
        return (), "unknown_table_roles"
    candidates = []
    for quality in range(1, 101):
        for baseline in (True, False):
            if all(lookup[table_id] == ijg_quantization_table(quality, chroma, baseline)
                   for table_id, chroma in roles):
                candidates.append(quality)
                break
    candidates = tuple(sorted(set(candidates)))
    return candidates, "exact" if len(candidates) == 1 else "ambiguous" if candidates else "custom"


def _from_image(image):
    if image.format not in {"JPEG", "MPO"}:
        return _empty("not_jpeg")
    metadata = _empty("custom")
    metadata["source_jpeg_quality_method"] = "custom_qtables"
    metadata["source_jpeg_progressive"] = bool(image.info.get("progressive") or image.info.get("progression"))
    layers = tuple(getattr(image, "layer", ()))
    if len(layers) == 1:
        sampling = "4:0:0"
    elif tuple(layer[0] for layer in layers) == (67, 77, 89, 75):
        sampling = "not_applicable_cmyk"
    else:
        sampling = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}.get(JpegImagePlugin.get_sampling(image), "unknown")
    metadata["source_jpeg_subsampling"] = sampling
    quantization = getattr(image, "quantization", None)
    if not quantization:
        metadata["source_jpeg_quality_status"] = "missing_qtables"
        metadata["source_jpeg_quality_method"] = "unknown"
        return metadata
    tables = tuple(sorted((int(identifier), tuple(int(value) for value in values))
                          for identifier, values in quantization.items()))
    payload = json.dumps({str(identifier): values for identifier, values in tables},
                         sort_keys=True, separators=(",", ":"))
    metadata["source_jpeg_qtables"] = payload
    metadata["source_jpeg_qtable_hash"] = hashlib.sha256(payload.encode()).hexdigest()
    if any(len(values) != 64 or min(values) < 1 or max(values) > 65535 for _, values in tables):
        metadata["source_jpeg_quality_status"] = "invalid_qtables"
        return metadata
    candidates, status = _exact_quality(tables, _table_roles(image))
    metadata["source_jpeg_quality_status"] = status
    if len(candidates) == 1:
        quality = candidates[0]
        metadata.update(
            source_jpeg_quality=quality, source_jpeg_quality_level=str(quality),
            source_jpeg_quality_bin=jpeg_quality_bin(quality),
            source_jpeg_quality_method="exact_ijg_qtables",
        )
    elif status != "custom":
        metadata["source_jpeg_quality_method"] = "unknown"
    return metadata


def extract_jpeg_metadata(source):
    try:
        if isinstance(source, Image.Image):
            return _from_image(source)
        if isinstance(source, (bytes, bytearray, memoryview)):
            source = BytesIO(bytes(source))
        elif isinstance(source, (str, Path)):
            source = Path(source).expanduser()
        with Image.open(source) as image:
            return _from_image(image)
    except (OSError, ValueError, TypeError, SyntaxError, EOFError) as error:
        metadata = _empty("unreadable")
        metadata["source_jpeg_metadata_error"] = type(error).__name__ + ": " + str(error)[:200]
        return metadata
