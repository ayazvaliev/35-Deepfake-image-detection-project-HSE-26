from __future__ import annotations

import hashlib
from io import BytesIO
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image

from src.preprocessing.jpeg import extract_jpeg_metadata
from src.preprocessing.verification_views import save_original_verification_view, save_crop_verification_view


FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)
EXCLUSION_FLAGS = frozenset({
    "face_too_small", "annotation_conflict", "opposite_target_in_crop",
    "ambiguous_multi_face_target", "missing_binary_target", "invalid_bbox",
    "decoding_failed", "no_face", "visible_face_too_small",
})


def array_sha256(array):
    array = np.ascontiguousarray(array)
    prefix = f"{array.dtype}:{array.shape}:".encode()
    return hashlib.sha256(prefix + array.tobytes()).hexdigest()


def perceptual_hash(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(gray.astype(np.float32))[:8, :8]
    threshold = np.median(coefficients.reshape(-1)[1:])
    value = 0
    for bit in (coefficients > threshold).reshape(-1):
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def angle_bin(angle):
    if angle is None or not np.isfinite(angle):
        return "unknown"
    angle = abs(float(angle))
    if angle < 15:
        return "frontal"
    if angle <= 45:
        return "three_quarter"
    return "profile"


def lightness_bin(value):
    if value is None or not np.isfinite(value):
        return "unknown"
    if value < 35:
        return "dark"
    if value < 65:
        return "medium"
    return "bright"


def box_iou(first, second):
    x1, y1, w1, h1 = map(float, first)
    x2, y2, w2, h2 = map(float, second)
    intersection = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
    intersection *= max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union if union > 0 else 0.0


def boxes_intersect(first, second):
    x1, y1, w1, h1 = map(float, first)
    x2, y2, w2, h2 = map(float, second)
    return min(x1 + w1, x2 + w2) > max(x1, x2) and min(y1 + h1, y2 + h2) > max(y1, y2)


def square_bounds(bbox, expansion=1.3):
    x, y, width, height = map(float, bbox)
    if not all(np.isfinite([x, y, width, height])) or min(width, height) <= 0:
        raise ValueError("bbox must contain finite x, y and positive width, height")
    side = int(math.ceil(expansion * max(width, height)))
    left = int(math.floor(x + width / 2 - side / 2 + 0.5))
    top = int(math.floor(y + height / 2 - side / 2 + 0.5))
    return left, top, side, side


def square_face_crop(image, bbox, size=256, expansion=1.3):
    left, top, side, _ = square_bounds(bbox, expansion)
    height, width = image.shape[:2]
    right, bottom = left + side, top + side
    x1, y1 = max(left, 0), max(top, 0)
    x2, y2 = min(right, width), min(bottom, height)
    square = np.full((side, side, 3), 127, dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        square[y1 - top:y2 - top, x1 - left:x2 - left] = image[y1:y2, x1:x2]
    if side == size:
        crop = square
    else:
        interpolation = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
        crop = cv2.resize(square, (size, size), interpolation=interpolation)
    visible_area = max(0, x2 - x1) * max(0, y2 - y1)
    return crop, (left, top, side, side), 1 - visible_area / (side * side)


def crop_padding_fraction(image_shape, bounds):
    height, width = image_shape[:2]
    left, top, crop_width, crop_height = bounds
    visible_width = max(0, min(width, left + crop_width) - max(0, left))
    visible_height = max(0, min(height, top + crop_height) - max(0, top))
    return 1 - visible_width * visible_height / (crop_width * crop_height)


def adaptive_square_bounds(bbox, image_shape, expansion=1.3):
    x, y, face_width, face_height = map(float, bbox)
    if not np.isfinite([x, y, face_width, face_height, expansion]).all() or min(face_width, face_height) <= 0:
        raise ValueError("bbox and expansion must be finite with positive face dimensions")
    if not 0 < expansion <= 1.3:
        raise ValueError("adaptive expansion must be positive and at most 1.3")
    height, width = image_shape[:2]
    center_x, center_y = x + face_width / 2, y + face_height / 2
    capacity = 2 * min(center_x, center_y, width - center_x, height - center_y)
    side = int(math.floor(min(expansion * max(face_width, face_height), capacity)))
    if not 0 < center_x < width or not 0 < center_y < height or side < 1:
        raise ValueError("no centered square of at least one pixel fits inside the image")
    left = int(math.floor(center_x - side / 2 + 0.5))
    top = int(math.floor(center_y - side / 2 + 0.5))
    left = min(max(left, 0), width - side)
    top = min(max(top, 0), height - side)
    return left, top, side, side


def adaptive_face_crop(image, bbox, size=256, expansion=1.3):
    left, top, side, _ = adaptive_square_bounds(bbox, image.shape, expansion)
    square = image[top:top + side, left:left + side]
    if side == size:
        crop = square.copy()
    else:
        interpolation = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
        crop = cv2.resize(square, (size, size), interpolation=interpolation)
    return crop, (left, top, side, side), 0.0


def face_visibility(bbox, bounds, image_shape):
    x, y, width, height = map(float, bbox)
    left, top, side, _ = bounds
    image_height, image_width = image_shape[:2]
    visible_width = max(0.0, min(x + width, left + side, image_width) - max(x, left, 0))
    visible_height = max(0.0, min(y + height, top + side, image_height) - max(y, top, 0))
    coverage = min(1.0, max(0.0, visible_width * visible_height / (width * height)))
    return coverage, min(visible_width, visible_height)


def matrix_angles(matrix):
    rotation = np.asarray(matrix, dtype=float)[:3, :3]
    left, _, right = np.linalg.svd(rotation)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    pitch, yaw, roll = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
    return float(yaw), float(pitch), float(roll)


class YuNetFacePreprocessor:
    def __init__(
        self, yunet_model=None, landmarker_model=None, *, size=256,
        expansion=1.3, min_face_size=32, detection_threshold=0.9,
        annotation_conflict_iou=0.8, crop_policy="adaptive", detector=None, landmarker=None,
        compact_verification_crops=False,
    ):
        self.size = int(size)
        self.expansion = float(expansion)
        self.min_face_size = float(min_face_size)
        self.detection_threshold = float(detection_threshold)
        self.annotation_conflict_iou = float(annotation_conflict_iou)
        self.crop_policy = str(crop_policy)
        self.compact_verification_crops = bool(compact_verification_crops)
        self._mediapipe = None
        if self.size <= 0 or self.expansion < 1:
            raise ValueError("size must be positive and expansion must be at least one")
        if self.crop_policy not in {"adaptive", "padded"}:
            raise ValueError("crop_policy must be adaptive or padded")
        if self.crop_policy == "adaptive" and self.expansion > 1.3:
            raise ValueError("adaptive expansion must be at most 1.3")
        if detector is None:
            if yunet_model is None:
                raise ValueError("yunet_model or detector is required")
            detector = cv2.FaceDetectorYN.create(
                str(yunet_model), "", (320, 320), detection_threshold, 0.3, 5000
            )
        self.detector = detector
        if landmarker is None and landmarker_model is not None:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(landmarker_model)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                output_facial_transformation_matrixes=True,
            )
            landmarker = vision.FaceLandmarker.create_from_options(options)
            self._mediapipe = mp
        self.landmarker = landmarker

    def close(self):
        if self.landmarker is not None and hasattr(self.landmarker, "close"):
            self.landmarker.close()

    def _detect(self, image):
        height, width = image.shape[:2]
        inputs = [("native", image)]
        if max(height, width) > 320:
            factor = 320 / max(height, width)
            resized_width = max(1, int(round(width * factor)))
            resized_height = max(1, int(round(height * factor)))
            resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
            inputs.append(("downscaled", resized))
        predictions = []
        counts = {"native": 0, "downscaled": 0}
        for name, detector_image in inputs:
            detector_height, detector_width = detector_image.shape[:2]
            self.detector.setInputSize((detector_width, detector_height))
            _, detections = self.detector.detect(detector_image)
            if detections is None:
                continue
            scale_x = width / detector_width
            scale_y = height / detector_height
            for detection in detections:
                row = np.asarray(detection, dtype=float).copy()
                if row.shape != (15,):
                    raise ValueError("YuNet detection must have bbox, five landmarks and score")
                if not np.isfinite(row).all() or min(row[2:4]) <= 0 or row[14] < self.detection_threshold:
                    continue
                row[[0, 2, 4, 6, 8, 10, 12]] *= scale_x
                row[[1, 3, 5, 7, 9, 11, 13]] *= scale_y
                predictions.append(row.tolist())
                counts[name] += 1
        self._detection_audit = {
            "detector_policy": "native_plus_downscaled_320",
            "detector_native_faces": counts["native"],
            "detector_downscaled_faces": counts["downscaled"],
            "detector_downscaled_applied": len(inputs) > 1,
        }
        if not predictions:
            return []
        indices = cv2.dnn.NMSBoxes(
            [row[:4] for row in predictions], [row[14] for row in predictions],
            self.detection_threshold, 0.3,
        )
        if indices is None or len(indices) == 0:
            return []
        return [predictions[int(index)] for index in np.asarray(indices).reshape(-1)]

    def _segmentation_mask(self, segmentation, height, width):
        if segmentation is None:
            return None
        if isinstance(segmentation, np.ndarray):
            if segmentation.shape != (height, width):
                raise ValueError("segmentation mask shape differs from image")
            return segmentation.astype(bool)
        if isinstance(segmentation, dict):
            if isinstance(segmentation.get("counts"), list):
                flat = np.zeros(height * width, dtype=np.uint8)
                offset, fill = 0, 0
                for length in segmentation["counts"]:
                    length = int(length)
                    if length < 0 or offset + length > flat.size:
                        raise ValueError("invalid uncompressed RLE")
                    flat[offset:offset + length] = fill
                    offset += length
                    fill = 1 - fill
                if offset != flat.size:
                    raise ValueError("RLE does not cover image")
                return flat.reshape((height, width), order="F").astype(bool)
            from pycocotools import mask as mask_utils

            rle = dict(segmentation)
            if isinstance(rle.get("counts"), str):
                rle["counts"] = rle["counts"].encode()
            decoded = mask_utils.decode(rle)
            if decoded.shape != (height, width):
                raise ValueError("segmentation RLE shape differs from image")
            return decoded.astype(bool)
        polygons = segmentation
        if polygons and isinstance(polygons[0], (float, int)):
            polygons = [polygons]
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in polygons:
            points = np.asarray(polygon, dtype=float).reshape(-1, 2)
            if len(points) >= 3 and np.isfinite(points).all():
                cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
        return mask.astype(bool)

    def _measure(self, image, bbox, annotation, crop, bounds):
        measurements = {
            "yaw": None, "pitch": None, "roll": None, "yaw_bin": "unknown",
            "pitch_bin": "unknown", "roll_bin": "unknown",
            "lightness_lstar": None, "light_bin": "unknown",
            "lightness_bin": "unknown", "mask_source": "missing",
            "face_mask_fraction_of_crop": None,
        }
        flags = []
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        landmarks = None
        if self.landmarker is not None:
            if self._mediapipe is None:
                result = self.landmarker.detect(rgb)
            else:
                mp_image = self._mediapipe.Image(
                    image_format=self._mediapipe.ImageFormat.SRGB, data=rgb
                )
                result = self.landmarker.detect(mp_image)
            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                matrices = result.facial_transformation_matrixes
                if len(matrices):
                    yaw, pitch, roll = matrix_angles(matrices[0])
                    for name, value in (("yaw", yaw), ("pitch", pitch), ("roll", roll)):
                        measurements[name] = value
                        measurements[name + "_bin"] = angle_bin(value)
        if landmarks is None:
            flags.append("landmark_missing")
        height, width = image.shape[:2]
        segmentation = annotation.get("segmentation")
        try:
            mask = self._segmentation_mask(segmentation, height, width)
        except (ValueError, TypeError, KeyError, IndexError):
            mask = None
            flags.append("annotation_mask_invalid")
        if mask is not None and mask.any():
            measurements["mask_source"] = "annotation"
        else:
            if mask is not None:
                flags.append("annotation_mask_empty")
            mask = None
            if landmarks is not None:
                left, top, side, _ = bounds
                points = np.asarray([
                    (left + landmarks[i].x * side, top + landmarks[i].y * side)
                    for i in FACE_OVAL
                ])
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
                mask = mask.astype(bool)
                measurements["mask_source"] = "mediapipe_face_oval"
        if mask is not None:
            left, top, side, _ = bounds
            x1, y1 = max(left, 0), max(top, 0)
            x2, y2 = min(left + side, width), min(top + side, height)
            crop_mask = mask[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else np.zeros((0, 0), dtype=bool)
            measurements["face_mask_fraction_of_crop"] = float(crop_mask.sum() / (side * side))
        else:
            crop_mask = None
        if crop_mask is not None and crop_mask.any():
            pixels = image[y1:y2, x1:x2][crop_mask].reshape(-1, 1, 3).astype(np.float32) / 255
            lab = cv2.cvtColor(pixels, cv2.COLOR_BGR2LAB)
            value = float(lab[..., 0].mean())
            measurements["lightness_lstar"] = value
            measurements["light_bin"] = lightness_bin(value)
            measurements["lightness_bin"] = lightness_bin(value)
        else:
            flags.append("face_mask_missing")
        return measurements, flags

    def process(
        self, image_path, *, target=None, annotations=None, output_dir=None,
        image_label_all_faces=False,
    ):
        image_path = Path(image_path)
        content = image_path.read_bytes()
        image_hash = hashlib.sha256(content).hexdigest()
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        base = {
            "input_path": str(image_path), "image_sha256": image_hash,
            "source_extension": image_path.suffix.lower(), "file_size": len(content),
            "source_format": "unknown", "source_format_verified": False,
            "source_width": None, "source_height": None,
            "crop_policy": self.crop_policy, "output_size": self.size,
            "expansion_requested": self.expansion, "proposed_expansion": 1.3,
            "detector_policy": "native_plus_downscaled_320",
            "detector_native_faces": None, "detector_downscaled_faces": None,
            "detector_downscaled_applied": None,
        }
        try:
            with Image.open(BytesIO(content)) as source:
                base.update(
                    source_format=source.format or "unknown", source_format_verified=True,
                    source_width=source.width, source_height=source.height,
                )
                base.update(extract_jpeg_metadata(source))
        except (OSError, ValueError):
            base.update(extract_jpeg_metadata(content))
        if image is None:
            return [dict(base, target=target, flags=["decoding_failed"],
                         exclusion_flags=["decoding_failed"], eligible=False,
                         path=None, crop_path=None)]
        base.update(
            decoded_sha256=array_sha256(image),
            image_width=image.shape[1], image_height=image.shape[0],
        )
        if output_dir is not None:
            base.update(save_original_verification_view(
                image, content, Path(output_dir) / "verification_originals",
            ))
        detections = self._detect(image)
        base.update(self._detection_audit)
        supplied = annotations is not None
        if supplied:
            faces = [dict(item) for item in annotations]
        else:
            faces = [{"bbox": row[:4], "target": target} for row in detections]
        if not faces:
            return [dict(base, target=target, flags=["no_face"], exclusion_flags=["no_face"],
                         eligible=False, detected_faces=len(detections), path=None, crop_path=None)]
        output = []
        for face_index, face in enumerate(faces):
            row = dict(base, face_index=face_index, detected_faces=len(detections),
                       target=face.get("target"), annotation_id=face.get("annotation_id"))
            bbox = face.get("bbox")
            flags = []
            if row["target"] not in (0, 1):
                flags.append("missing_binary_target")
            if not supplied and len(faces) > 1 and not image_label_all_faces:
                flags.append("ambiguous_multi_face_target")
            try:
                proposed_bounds = square_bounds(bbox, expansion=1.3)
                row["proposed_crop_bounds"] = list(proposed_bounds)
                row["proposed_padding_fraction"] = float(crop_padding_fraction(image.shape, proposed_bounds))
                if self.crop_policy == "adaptive":
                    crop, bounds, padding = adaptive_face_crop(image, bbox, self.size, self.expansion)
                else:
                    crop, bounds, padding = square_face_crop(image, bbox, self.size, self.expansion)
            except (ValueError, TypeError):
                flags.append("invalid_bbox")
                exclusions = sorted(set(flags) & EXCLUSION_FLAGS)
                output.append(dict(row, bbox=bbox, flags=flags, exclusion_flags=exclusions,
                                   eligible=False, path=None, crop_path=None))
                continue
            bbox = [float(v) for v in bbox]
            if min(bbox[2:]) < self.min_face_size:
                flags.append("face_too_small")
            coverage, visible_min_side = face_visibility(bbox, bounds, image.shape)
            truncates_bbox = coverage < 1 - 1e-9
            if truncates_bbox:
                flags.append("crop_truncates_bbox")
            if visible_min_side < self.min_face_size:
                flags.append("visible_face_too_small")
            scores = [box_iou(bbox, detection[:4]) for detection in detections]
            best = int(np.argmax(scores)) if scores else None
            verified = best is not None and scores[best] >= 0.3
            if supplied and not verified:
                flags.append("detector_missing")
            for other_index, other in enumerate(faces):
                if other_index == face_index or other.get("target") not in (0, 1):
                    continue
                if other["target"] == row["target"]:
                    continue
                try:
                    if box_iou(bbox, other["bbox"]) >= self.annotation_conflict_iou:
                        flags.append("annotation_conflict")
                    if boxes_intersect(bounds, other["bbox"]):
                        flags.append("opposite_target_in_crop")
                except (ValueError, TypeError):
                    continue
            measurements, measurement_flags = self._measure(image, bbox, face, crop, bounds)
            flags.extend(measurement_flags)
            crop_hash = array_sha256(crop)
            crop_path = None
            if output_dir is not None:
                destination = Path(output_dir)
                destination.mkdir(parents=True, exist_ok=True)
                identity = hashlib.sha256(
                    (str(image_path.resolve()) + ":" + str(face_index) + ":" + str(bbox)
                     + ":" + self.crop_policy + ":" + str(bounds) + ":" + str(self.size)).encode()
                ).hexdigest()[:24]
                crop_path = destination / f"{identity}.png"
                if not cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_PNG_COMPRESSION, 0 if self.compact_verification_crops else 9]):
                    raise OSError(f"cannot write crop {crop_path}")
                row["crop_file_sha256"] = hashlib.sha256(crop_path.read_bytes()).hexdigest()
                if self.compact_verification_crops:
                    row.update(save_crop_verification_view(
                        crop, destination / "verification_crops", original_metadata=base,
                        crop_bounds=bounds, source_pixels=image,
                    ))
                    crop_path.unlink()
                    row["training_crop_removed"] = True
                else:
                    row["training_crop_removed"] = False
                crop_path = str(crop_path)
            flags = sorted(set(flags))
            exclusions = sorted(set(flags) & EXCLUSION_FLAGS)
            row.update(
                bbox=bbox, crop_bounds=list(bounds), padding_fraction=float(padding),
                expansion_applied=float(bounds[2] / max(bbox[2:])),
                crop_truncates_bbox=bool(truncates_bbox),
                face_coverage_fraction=float(coverage), visible_face_min_side=float(visible_min_side),
                detector_verified=verified,
                detector_confidence=float(detections[best][-1]) if verified else None,
                flags=flags, exclusion_flags=exclusions, eligible=not exclusions,
                crop_sha256=crop_hash, phash=perceptual_hash(crop),
                path=crop_path, crop_path=crop_path,
            )
            row.update(measurements)
            output.append(row)
        return output

    def process_record(self, record, output_dir=None):
        input_path = record.get("input_path", record.get("path"))
        annotations = record.get("annotations")
        if annotations is None and record.get("bbox") is not None:
            annotations = [{
                "bbox": record["bbox"], "target": record.get("target"),
                "segmentation": record.get("segmentation"),
                "annotation_id": record.get("annotation_id"),
            }]
        rows = self.process(
            input_path, target=record.get("target"), annotations=annotations,
            output_dir=output_dir,
            image_label_all_faces=record.get("image_label_all_faces", False),
        )
        metadata = {key: value for key, value in record.items()
                    if key not in {"annotations", "segmentation"}}
        return [dict(metadata, **row) for row in rows]
