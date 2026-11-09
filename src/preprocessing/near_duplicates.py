from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


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
        }
        if not 0 < ssim_min <= 1 or not 0 <= rgb_mae_max < 1 or view_size < 16:
            raise ValueError("Invalid near-duplicate similarity settings")
        if min_gray_std < 0 or min_laplacian_variance < 0 or aspect_ratio_tolerance < 0 or cache_size < 1:
            raise ValueError("Invalid near-duplicate view settings")
        columns = [name for name in ("sample_id", "input_path", "crop_path", "path", "image_sha256",
                                      "decoded_sha256", "crop_sha256") if name in frame]
        self.rows = frame[columns].to_dict("index")
        self.counts = Counter()
        self.unreadable_indices = set()
        self.examples = []
        self._view = lru_cache(maxsize=cache_size)(self._load_view)

    def _load_view(self, filename):
        if not filename or not Path(filename).is_file():
            return None
        try:
            content = Path(filename).read_bytes()
            self.counts["loaded_bytes"] += len(content)
            image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or min(image.shape[:2]) < 1:
                return None
            self.counts["decoded_views"] += 1
            height, width = image.shape[:2]
            size = self.settings["view_size"]
            interpolation = cv2.INTER_AREA if max(height, width) >= size else cv2.INTER_CUBIC
            image = cv2.resize(image, (size, size), interpolation=interpolation)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return (image, width / height, float(gray.std()) / 255,
                    float(cv2.Laplacian(gray, cv2.CV_32F).var()))
        except (OSError, ValueError, cv2.error):
            return None

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
        for kind, column in (("crop", "crop_path"), ("original", "input_path")):
            first_path = _text(first.get(column))
            second_path = _text(second.get(column))
            if kind == "crop":
                first_path = first_path or _text(first.get("path"))
                second_path = second_path or _text(second.get("path"))
            first_view, second_view = self._view(first_path), self._view(second_path)
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
                return True
        if reasons:
            self.counts["rejected_pairs"] += 1
            self.counts["low_texture_rejections"] += "low_texture" in reasons
        else:
            self.counts["unreadable_pairs"] += 1
            missing = {index for index, found in available.items() if not found} or set(available)
            self.unreadable_indices.update(missing)
            if len(self.examples) < 10:
                self.examples.append({
                    "sample_ids": [_text(first.get("sample_id")), _text(second.get("sample_id"))],
                    "reason": "No readable compatible original pair or crop pair",
                })
        return False

    def report(self):
        keys = ("candidate_pairs", "verified_pairs", "rejected_pairs", "unreadable_pairs",
                "exact_metadata_pairs", "verified_original_pairs", "verified_crop_pairs", "low_texture_rejections",
                "decoded_views", "loaded_bytes")
        return {
            "mode": "pixel_verification", **{key: int(self.counts[key]) for key in keys},
            "unreadable_rows": len(self.unreadable_indices), "excluded_rows": len(self.unreadable_indices),
            "settings": self.settings, "unreadable_examples": self.examples,
            "cache_hits": self._view.cache_info().hits, "cache_misses": self._view.cache_info().misses,
        }

    def close(self):
        self._view.cache_clear()
