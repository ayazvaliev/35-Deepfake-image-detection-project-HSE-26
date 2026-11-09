from __future__ import annotations

import numpy as np
import pandas as pd


def _bbox(value):
    try:
        values = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if values.shape != (4,) or not np.isfinite(values).all() or min(values[2:]) <= 0:
        return None
    return values


def _iou(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union > 0 else 0.0


def matched_face_conflicts(frame, min_iou=0.8):
    if not 0 < min_iou <= 1:
        raise ValueError('min_iou must lie in (0, 1]')
    if 'bbox' not in frame or not {'dataset', 'target'} <= set(frame):
        return set()
    eligible = frame['eligible'].fillna(False).astype(bool) if 'eligible' in frame else pd.Series(True, index=frame.index)
    candidates = frame.loc[eligible & frame['target'].isin([0, 1])]
    boxes = {index: _bbox(value) for index, value in candidates['bbox'].items()}
    datasets = candidates['dataset'].astype(str).str.lower().to_dict()
    targets = candidates['target'].to_dict()
    conflicts = set()
    checked = set()
    for column in ('decoded_sha256', 'image_sha256'):
        if column not in candidates:
            continue
        known = candidates.loc[candidates[column].notna() & candidates[column].astype(str).ne('')]
        for _, group in known.groupby(column, sort=False):
            if group['dataset'].astype(str).str.lower().nunique() < 2 or group['target'].nunique() < 2:
                continue
            indices = list(group.index)
            for position, first in enumerate(indices):
                if boxes[first] is None:
                    continue
                for second in indices[position + 1:]:
                    pair = (first, second)
                    if pair in checked or datasets[first] == datasets[second] or targets[first] == targets[second] or boxes[second] is None:
                        continue
                    checked.add(pair)
                    if _iou(boxes[first], boxes[second]) >= min_iou:
                        conflicts.update(pair)
    return conflicts
