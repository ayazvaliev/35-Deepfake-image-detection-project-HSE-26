from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd


MASK_FRACTION_COLUMN = "face_mask_fraction_of_crop"
MASK_QUALITY_FLAGS = frozenset({"face_mask_too_small", "face_mask_missing"})


def _flags(value):
    if value is None or value is pd.NA:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        if value.lstrip().startswith("["):
            try:
                parsed = json.loads(value)
            except ValueError:
                return [value]
            if isinstance(parsed, list):
                return _flags(parsed)
        return [value]
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [str(item) for item in value if item is not None and item is not pd.NA and str(item)]
    if pd.isna(value):
        return []
    return [str(value)]


def apply_quality_policy(frame, min_mask_fraction=0.25):
    threshold = float(min_mask_fraction)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("min_mask_fraction must be finite and between 0 and 1")
    result = frame.copy(deep=True)
    if MASK_FRACTION_COLUMN not in result:
        result[MASK_FRACTION_COLUMN] = np.nan
    fraction = pd.to_numeric(result[MASK_FRACTION_COLUMN], errors="coerce")
    known = fraction.notna() & np.isfinite(fraction) & fraction.between(0, 1)
    too_small = known & fraction.le(threshold)
    missing = ~known
    original_flags = result["flags"].tolist() if "flags" in result else [[] for _ in range(len(result))]
    original_exclusions = result["exclusion_flags"].tolist() if "exclusion_flags" in result else [[] for _ in range(len(result))]
    if "eligible" in result:
        eligible_before = result["eligible"].fillna(False).astype(bool).to_numpy()
    else:
        eligible_before = np.ones(len(result), dtype=bool)
    flags, exclusions, eligible = [], [], []
    for previous_flags, previous_exclusions, was_eligible, is_small, is_missing in zip(
        original_flags, original_exclusions, eligible_before, too_small, missing
    ):
        current_flags = set(_flags(previous_flags))
        current_exclusions = set(_flags(previous_exclusions))
        if is_small:
            current_flags.add("face_mask_too_small")
        if is_missing:
            current_flags.add("face_mask_missing")
        current_exclusions.update(current_flags & MASK_QUALITY_FLAGS)
        flags.append(sorted(current_flags))
        exclusions.append(sorted(current_exclusions))
        eligible.append(bool(was_eligible and not current_exclusions))
    result["flags"] = flags
    result["exclusion_flags"] = exclusions
    result["eligible"] = pd.Series(eligible, index=result.index, dtype=bool)
    eligible_after = np.asarray(eligible, dtype=bool)
    audit = {
        "schema_version": 1,
        "column": MASK_FRACTION_COLUMN,
        "min_mask_fraction": threshold,
        "threshold_comparison": "<=",
        "missing_fraction_policy": "exclude",
        "valid_fraction_range": [0.0, 1.0],
        "exclusion_flags": sorted(MASK_QUALITY_FLAGS),
        "processing_records": len(result),
        "known_fraction_records": int(known.sum()),
        "missing_fraction_records": int(missing.sum()),
        "too_small_fraction_records": int(too_small.sum()),
        "eligible_before": int(eligible_before.sum()),
        "eligible_after": int(eligible_after.sum()),
        "newly_excluded_records": int((eligible_before & ~eligible_after).sum()),
    }
    result.attrs = {**frame.attrs, "quality_policy": audit}
    return result, audit
