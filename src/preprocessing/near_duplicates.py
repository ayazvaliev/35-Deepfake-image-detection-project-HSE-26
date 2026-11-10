from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.preprocessing.verification_views import (
    VERIFICATION_VIEW_VERSION, normalized_original_view, read_verification_view,
)


def _text(value):
    if value is None or value is pd.NA or isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def structural_similarity(first, second):
    first = first.astype(np.float32)
    second = second.astype(np.float32)
    mean_first = cv2.GaussianBlur(first, (11, 11), 1.5)
    mean_second = cv2.GaussianBlur(second, (11, 11), 1.5)
    variance_first = np.maximum(cv2.GaussianBlur(first * first, (11, 11), 1.5) - mean_first ** 2, 0)
    variance_second = np.maximum(cv2.GaussianBlur(second * second, (11, 11), 1.5) - mean_second ** 2, 0)
    covariance = cv2.GaussianBlur(first * second, (11, 11), 1.5) - mean_first * mean_second
    numerator = (2 * mean_first * mean_second + 6.5025) * (2 * covariance + 58.5225)
    denominator = (mean_first ** 2 + mean_second ** 2 + 6.5025) * (variance_first + variance_second + 58.5225)
    return float((numerator / denominator)[5:-5, 5:-5].mean())


class PixelNearDuplicateVerifier:
    def __init__(self, frame, ssim_min=0.95, rgb_mae_max=5 / 255, view_size=256,
                 aspect_ratio_tolerance=0.02, min_gray_std=8 / 255,
                 min_laplacian_variance=4.0, cache_size=512):
        self.settings = {
            "ssim_min": float(ssim_min), "ssim_space": "RGB", "ssim_window": 11, "ssim_sigma": 1.5,
            "comparison_order": ["crop", "original"], "rgb_mae_max": float(rgb_mae_max),
            "view_size": int(view_size), "aspect_ratio_tolerance": float(aspect_ratio_tolerance),
            "min_gray_std": float(min_gray_std),
            "min_laplacian_variance": float(min_laplacian_variance), "cache_size": int(cache_size),
            "opencv_threads": cv2.getNumThreads(),
            "verification_view_version": VERIFICATION_VIEW_VERSION,
        }
        if not 0 < ssim_min <= 1 or not 0 <= rgb_mae_max < 1 or view_size < 16:
            raise ValueError("Invalid near-duplicate similarity settings")
        if min_gray_std < 0 or min_laplacian_variance < 0 or aspect_ratio_tolerance < 0 or cache_size < 1:
            raise ValueError("Invalid near-duplicate view settings")
        columns = [name for name in ("sample_id", "input_path", "crop_path", "path", "image_sha256",
                                      "decoded_sha256", "crop_sha256", "original_verification_path",
                                      "original_verification_sha256", "original_verification_pixel_sha256",
                                      "original_verification_codec", "original_verification_size", "original_aspect_ratio",
                                      "crop_verification_path", "crop_verification_sha256", "crop_verification_pixel_sha256",
                                      "crop_verification_codec", "crop_verification_size") if name in frame]
        self.rows = frame[columns].to_dict("index")
        self.counts = Counter()
        self.unreadable_indices = set()
        self.examples = []
        self._view = lru_cache(maxsize=cache_size)(self._load_view)

    def _load_view(self, filename, file_sha256="", pixel_sha256="", stored_size=None,
                   codec="", aspect_ratio=None):
        if not filename or not Path(filename).is_file():
            self.counts["missing_proxy_views"] += bool(file_sha256)
            return None
        try:
            if file_sha256:
                if aspect_ratio is not None and int(stored_size) != self.settings["view_size"]:
                    raise ValueError("Original proxy normalization differs from the requested view")
                image = read_verification_view(filename, file_sha256, pixel_sha256, stored_size, codec)
                self.counts["loaded_bytes"] += Path(filename).stat().st_size
                ratio = 1.0 if aspect_ratio is None else float(aspect_ratio)
                if not np.isfinite(ratio) or ratio <= 0:
                    raise ValueError("Invalid original aspect ratio in verification metadata")
                self.counts["decoded_proxy_views"] += 1
            else:
                content = Path(filename).read_bytes()
                self.counts["loaded_bytes"] += len(content)
                image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None or min(image.shape[:2]) < 1:
                    return None
                ratio = image.shape[1] / image.shape[0]
            self.counts["decoded_views"] += 1
            image = normalized_original_view(image, self.settings["view_size"])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return (image, ratio, float(gray.std()) / 255,
                    float(cv2.Laplacian(gray, cv2.CV_32F).var()))
        except (OSError, ValueError, TypeError, cv2.error):
            self.counts["proxy_integrity_failures"] += bool(file_sha256)
            return None

    def _row_view(self, index, row, kind):
        path = _text(row.get("input_path" if kind == "original" else "crop_path"))
        if kind == "crop":
            path = path or _text(row.get("path"))
        view = self._view(path)
        if view is not None:
            return view, False, False
        proxy = _text(row.get(kind + "_verification_path"))
        if not proxy:
            return None, False, False
        file_sha = _text(row.get(kind + "_verification_sha256"))
        pixel_sha = _text(row.get(kind + "_verification_pixel_sha256"))
        size = row.get(kind + "_verification_size")
        codec = _text(row.get(kind + "_verification_codec"))
        ratio = row.get("original_aspect_ratio") if kind == "original" else None
        if not file_sha or not pixel_sha or not codec:
            view = None
        else:
            view = self._view(proxy, file_sha, pixel_sha, size, codec, ratio)
        if view is None:
            self.unreadable_indices.add(index)
            return None, True, True
        return view, True, False

    def _unreadable(self, first, second, missing, reason):
        self.counts["unreadable_pairs"] += 1
        self.unreadable_indices.update(missing)
        if len(self.examples) < 10:
            self.examples.append({
                "sample_ids": [_text(first.get("sample_id")), _text(second.get("sample_id"))],
                "reason": reason,
            })
        return False

    def _compare(self, first, second):
        if min(first[2], second[2]) < self.settings["min_gray_std"]:
            return False, "low_texture"
        if min(first[3], second[3]) < self.settings["min_laplacian_variance"]:
            return False, "low_texture"
        mae = float(cv2.absdiff(first[0], second[0]).mean()) / 255
        if mae > self.settings["rgb_mae_max"]:
            return False, "mae"
        return structural_similarity(first[0], second[0]) >= self.settings["ssim_min"], "ssim"

    def __call__(self, first_index, second_index):
        self.counts["candidate_pairs"] += 1
        first, second = self.rows[first_index], self.rows[second_index]
        for column in ("image_sha256", "decoded_sha256", "crop_sha256"):
            value = _text(first.get(column))
            if value and value == _text(second.get(column)):
                self.counts["verified_pairs"] += 1
                self.counts["exact_metadata_pairs"] += 1
                return True
        reasons = []
        available = {first_index: False, second_index: False}
        for kind in ("crop", "original"):
            first_view, first_proxy, first_invalid = self._row_view(first_index, first, kind)
            second_view, second_proxy, second_invalid = self._row_view(second_index, second, kind)
            if first_invalid or second_invalid:
                self.counts["unreadable_proxy_pairs"] += 1
                missing = {index for index, invalid in ((first_index, first_invalid), (second_index, second_invalid)) if invalid}
                return self._unreadable(first, second, missing, "Verification proxy is missing or fails integrity checks")
            available[first_index] |= first_view is not None
            available[second_index] |= second_view is not None
            if first_view is None or second_view is None:
                continue
            if kind == "original":
                ratio_error = abs(first_view[1] / second_view[1] - 1)
                if ratio_error > self.settings["aspect_ratio_tolerance"]:
                    continue
            verified, reason = self._compare(first_view, second_view)
            reasons.append(reason)
            if verified:
                self.counts["verified_pairs"] += 1
                self.counts["verified_" + kind + "_pairs"] += 1
                self.counts["verified_" + kind + "_proxy_pairs"] += first_proxy or second_proxy
                return True
        if reasons:
            self.counts["rejected_pairs"] += 1
            self.counts["low_texture_rejections"] += "low_texture" in reasons
        else:
            missing = {index for index, found in available.items() if not found} or set(available)
            return self._unreadable(first, second, missing, "No readable compatible original pair or crop pair")
        return False

    def report(self):
        keys = ("candidate_pairs", "verified_pairs", "rejected_pairs", "unreadable_pairs",
                "exact_metadata_pairs", "verified_original_pairs", "verified_crop_pairs", "low_texture_rejections",
                "decoded_views", "loaded_bytes", "decoded_proxy_views", "proxy_integrity_failures",
                "missing_proxy_views", "unreadable_proxy_pairs", "verified_crop_proxy_pairs", "verified_original_proxy_pairs")
        return {
            "mode": "pixel_verification", **{key: int(self.counts[key]) for key in keys},
            "unreadable_rows": len(self.unreadable_indices), "excluded_rows": len(self.unreadable_indices),
            "settings": self.settings, "unreadable_examples": self.examples,
            "cache_hits": self._view.cache_info().hits, "cache_misses": self._view.cache_info().misses,
        }

    def close(self):
        self._view.cache_clear()
