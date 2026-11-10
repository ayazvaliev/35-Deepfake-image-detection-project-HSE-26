from __future__ import annotations

from collections import Counter
import re

import numpy as np
import pandas as pd


_GROUP_COLUMNS = ("source_group", "identity_group")
_HASH_COLUMNS = ("image_sha256", "decoded_sha256", "crop_sha256")
_SAMPLE_STRATA = ("target", "generator", "manipulation", "domain", "author_split")
_BALANCE_COLUMNS = ("dataset", "target", "generator", "manipulation", "domain", "yaw_bin", "pitch_bin", "roll_bin", "light_bin", "source_format", "face_size_bin", "padding_bin", "source_resolution_bin", "face_mask_fraction_of_crop_bin")
_BALANCE_LIST_COLUMNS = ("transformation_tags",)
_MASK_OCCUPANCY_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)
_MASK_OCCUPANCY_LABELS = ("[0,0.25)", "[0.25,0.5)", "[0.5,0.75)", "[0.75,1]")


def _value(value):
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def _flags(value):
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return {_value(item) for item in value if _value(item)}
    return {item.strip() for item in _value(value).split(";") if item.strip()}


def _tags(value):
    if isinstance(value, (list, tuple, set, np.ndarray)):
        items = value
    else:
        text = _value(value)
        if text.startswith("["):
            import json

            try:
                items = json.loads(text)
            except (TypeError, ValueError):
                items = text.split(";")
        else:
            items = text.split(";")
    return sorted({_value(item) for item in items if _value(item) not in {"", "unknown"}})


def _partition(value):
    name = re.sub(r"[^a-z0-9]", "", _value(value).lower())
    aliases = {
        "": "", "unknown": "", "unspecified": "",
        "train": "train", "training": "train", "learn": "train",
        "val": "validate", "valid": "validate", "validation": "validate",
        "validate": "validate", "dev": "validate",
        "test": "test", "testdev": "test", "testset": "test",
        "testprivate": "test", "testpublic": "test",
    }
    if name not in aliases:
        raise ValueError(f"Unknown author_split: {value!r}")
    return aliases[name]


def proportional_sample(frame, n=100_000, strata=None, seed=42):
    if not isinstance(n, (int, np.integer)) or n < 0 or n > len(frame):
        raise ValueError("n must be an integer between zero and the population size")
    columns = list(strata) if strata is not None else [c for c in _SAMPLE_STRATA if c in frame]
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"Missing sampling strata: {sorted(missing)}")
    rng = np.random.default_rng(seed)
    if not columns:
        positions = np.sort(rng.choice(len(frame), size=n, replace=False))
        return frame.iloc[positions].copy()
    keys = frame[columns].map(_value)
    groups = list(keys.groupby(columns, sort=True, dropna=False).indices.items())
    sizes = np.array([len(positions) for _, positions in groups], dtype=np.int64)
    quotas = sizes * (n / len(frame)) if len(frame) else np.zeros(len(sizes))
    allocation = np.floor(quotas).astype(np.int64)
    remaining = n - int(allocation.sum())
    order = np.argsort(-(quotas - allocation), kind="stable")
    allocation[order[:remaining]] += 1
    selected = []
    rows = []
    for (key, positions), count, size in zip(groups, allocation, sizes):
        selected.extend(rng.choice(positions, size=int(count), replace=False).tolist())
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(columns, key)), "population_n": int(size), "sample_n": int(count)})
    result = frame.iloc[sorted(selected)].copy()
    result.attrs["sampling"] = {
        "population_n": len(frame), "sample_n": len(result), "seed": seed,
        "strata": columns, "allocation": rows,
        "measured_feature_population_balance": False,
    }
    return result


class _Groups:
    def __init__(self, size):
        self.parent = list(range(size))
        self.sizes = [1] * size

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.sizes[left] < self.sizes[right]:
            left, right = right, left
        self.parent[right] = left
        self.sizes[left] += self.sizes[right]


class _HashChunks:
    def __init__(self, threshold):
        width, extra = divmod(64, threshold + 1)
        self.chunks = []
        offset = 0
        for index in range(threshold + 1):
            length = width + (index < extra)
            self.chunks.append((offset, (1 << length) - 1, {}))
            offset += length

    def query(self, value, threshold):
        candidates = set()
        for offset, mask, table in self.chunks:
            candidates.update(table.get((value >> offset) & mask, ()))
        return [other for other in sorted(candidates) if (value ^ other).bit_count() <= threshold]

    def insert(self, value):
        for offset, mask, table in self.chunks:
            table.setdefault((value >> offset) & mask, []).append(value)


class _HashTree:
    def __init__(self):
        self.root = None

    def query(self, value, threshold):
        found = []
        pending = [self.root] if self.root is not None else []
        while pending:
            node = pending.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= threshold:
                found.append(node[0])
            pending.extend(child for edge, child in node[1].items()
                           if distance - threshold <= edge <= distance + threshold)
        return found

    def insert(self, value):
        if self.root is None:
            self.root = [value, {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                return
            if distance not in node[1]:
                node[1][distance] = [value, {}]
                return
            node = node[1][distance]


def _phash(value):
    value = _value(value) if not isinstance(value, (int, np.integer)) else value
    if value == "":
        return None
    parsed = int(value, 16) if isinstance(value, str) else int(value)
    if not 0 <= parsed < 2 ** 64:
        raise ValueError("phash must be an unsigned 64-bit integer or hexadecimal string")
    return parsed


def _mask_occupancy(data):
    column = "face_mask_fraction_of_crop"
    if column in data:
        values = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    else:
        values = np.full(len(data), np.nan)
    valid = np.isfinite(values) & (values >= 0) & (values <= 1)
    values = np.where(valid, values, np.nan)
    labels = np.full(len(data), "unknown", dtype=object)
    positions = np.searchsorted(_MASK_OCCUPANCY_EDGES[1:-1], values[valid], side="right")
    labels[valid] = np.asarray(_MASK_OCCUPANCY_LABELS, dtype=object)[positions]
    data[column] = values
    data[column + "_bin"] = labels


def _prepare(frame):
    required = {"dataset", "sample_id", "target"}
    if not required <= set(frame):
        raise ValueError(f"Missing manifest columns: {sorted(required - set(frame))}")
    data = frame.copy().reset_index(drop=True)
    if data["sample_id"].isna().any() or data["sample_id"].astype(str).duplicated().any():
        raise ValueError("sample_id must be present and globally unique")
    if data["dataset"].map(_value).eq("").any():
        raise ValueError("dataset must be present")
    for column in (*_GROUP_COLUMNS, *_HASH_COLUMNS, "phash", "author_split",
                   "generator", "manipulation", "yaw_bin", "light_bin"):
        if column not in data:
            data[column] = ""
    data["author_partition"] = data["author_split"].map(_partition)
    data["flags"] = data["flags"].map(_flags) if "flags" in data else [set() for _ in range(len(data))]
    if "eligible" in data:
        if not data["eligible"].isin([True, False, 0, 1]).all():
            raise ValueError("eligible must contain explicit booleans")
        data["eligible"] = data["eligible"].astype(bool)
    else:
        data["eligible"] = data["flags"].map(lambda value: not value)
    numeric = pd.to_numeric(data["target"], errors="coerce")
    valid_target = numeric.isin([0, 1])
    if (data["eligible"] & ~valid_target).any():
        raise ValueError("eligible target must contain binary 0/1 values")
    data["target"] = numeric.where(valid_target).astype("Int8")
    if "bbox" in data and "face_size_bin" not in data:
        sizes = []
        for value in data["bbox"]:
            try:
                box = np.asarray(value, dtype=float)
                sizes.append(float(min(box[2:])) if box.shape == (4,) and np.isfinite(box).all() else np.nan)
            except (TypeError, ValueError):
                sizes.append(np.nan)
        data["face_size_bin"] = pd.cut(
            pd.Series(sizes), [0, 32, 64, 128, 256, np.inf],
            labels=["<32", "32-64", "64-128", "128-256", ">=256"], right=False,
        ).astype(object).fillna("unknown")
    if "padding_fraction" in data and "padding_bin" not in data:
        padding = pd.to_numeric(data["padding_fraction"], errors="coerce")
        data["padding_bin"] = np.select(
            [padding.eq(0).fillna(False).to_numpy(bool),
             (padding.gt(0) & padding.le(0.25)).fillna(False).to_numpy(bool),
             padding.gt(0.25).fillna(False).to_numpy(bool)],
            ["none", "some", "large"], default="unknown",
        )
    if {"image_width", "image_height"} <= set(data) and "source_resolution_bin" not in data:
        dimensions = data[["image_width", "image_height"]].apply(pd.to_numeric, errors="coerce")
        minimum = dimensions.min(axis=1, skipna=False)
        data["source_resolution_bin"] = pd.cut(
            minimum, [0, 256, 512, np.inf],
            labels=["<256", "256-512", ">=512"], right=False,
        ).astype(object).fillna("unknown")
    _mask_occupancy(data)
    from src.preprocessing.quality import apply_quality_policy

    data, quality_audit = apply_quality_policy(data)
    data["flags"] = data["flags"].map(_flags)
    data.attrs["quality_policy"] = quality_audit
    missing_sources = data["dataset"].astype(str).str.lower().eq("df40") & data["source_group"].map(_value).eq("")
    if missing_sources.any():
        raise ValueError("DF40 rows require source_group linking all frames and related source videos")
    return data


def _connect(data, threshold, verifier=None):
    groups = _Groups(len(data))
    data["group_id"] = ""
    for column in (*_GROUP_COLUMNS, *_HASH_COLUMNS):
        seen = {}
        for index, (raw, dataset) in enumerate(zip(data[column], data["dataset"])):
            if column == "crop_sha256" and not data.at[index, "eligible"]:
                continue
            value = _value(raw)
            if not value:
                continue
            if column in _GROUP_COLUMNS and not value.startswith("global:"):
                value = (dataset, value)
            key = (column, value)
            if key in seen:
                groups.union(index, seen[key])
            else:
                seen[key] = index
    by_hash = {}
    for index, value in enumerate(data["phash"]):
        if not data.at[index, "eligible"]:
            continue
        value = _phash(value)
        if value is not None:
            by_hash.setdefault(value, []).append(index)
    candidate_count = 0
    tree = _HashChunks(threshold) if threshold <= 4 else _HashTree()
    for value, indices in sorted(by_hash.items()):
        candidate_count += len(indices) * (len(indices) - 1) // 2
        if verifier is None:
            for index in indices[1:]:
                groups.union(indices[0], index)
        else:
            for position, first in enumerate(indices):
                for second in indices[position + 1:]:
                    if verifier(first, second):
                        groups.union(first, second)
        for other in tree.query(value, threshold):
            candidate_count += len(indices) * len(by_hash[other])
            if verifier is None:
                groups.union(indices[0], by_hash[other][0])
            else:
                for first in indices:
                    for second in by_hash[other]:
                        if verifier(first, second):
                            groups.union(first, second)
        tree.insert(value)
    members = {}
    for index in range(len(data)):
        members.setdefault(groups.find(index), []).append(index)
    labels = [""] * len(data)
    sample_ids = data["sample_id"].astype(str).to_list()
    for indices in members.values():
        group_id = min(sample_ids[index] for index in indices)
        for index in indices:
            labels[index] = group_id
    data["group_id"] = labels
    return candidate_count


def _feature_counts(data, weights=None):
    counts = Counter()
    counts[("size",)] = len(data) if weights is None else float(weights.loc[data.index].sum())
    for column in _BALANCE_COLUMNS:
        if column not in data:
            continue
        condition = [] if column == "dataset" else ["dataset"]
        if column not in ("dataset", "target"):
            condition.append("target")
        if column == "source_format" and "generator" in data:
            condition.append("generator")
        columns = [*condition, column]
        if weights is None:
            grouped = data.groupby(columns, dropna=False, sort=True).size()
        else:
            weighted = data[columns].copy()
            weighted["_weight"] = weights.loc[data.index].to_numpy()
            grouped = weighted.groupby(columns, dropna=False, sort=True)["_weight"].sum()
        for values, count in grouped.items():
            values = values if isinstance(values, tuple) else (values,)
            counts[(column, *map(_value, values))] = float(count)
    for column in _BALANCE_LIST_COLUMNS:
        if column not in data:
            continue
        for index, dataset, target, value in data[["dataset", "target", column]].itertuples(index=True, name=None):
            weight = 1.0 if weights is None else float(weights.loc[index])
            for tag in _tags(value) or ["unknown"]:
                counts[(column, _value(dataset), _value(target), tag)] += weight
    return counts


def _assign_flexible(data, fraction, seed, row_probabilities=None):
    pool = data.loc[data["eligible"] & data["partition"].ne("test")]
    totals = _feature_counts(pool)
    targets = ({key: count * fraction for key, count in totals.items()}
               if row_probabilities is None else _feature_counts(pool, row_probabilities))
    weights = {key: (500.0 if key == ("size",) else 1.0) / max(count, 5)
               for key, count in totals.items()}
    current = _feature_counts(pool.loc[pool["partition"].eq("validate")])
    flexible = []
    unassigned = pool.loc[pool["partition"].eq("")]
    indices_by_group = unassigned.groupby("group_id", sort=True).groups
    scalar_columns = [column for column in _BALANCE_COLUMNS if column in unassigned]
    list_columns = [column for column in _BALANCE_LIST_COLUMNS if column in unassigned]
    columns = [*scalar_columns, *list_columns]
    counters = {}
    for row in unassigned[["group_id", *columns]].itertuples(index=False, name=None):
        group_id, *raw_values = row
        raw_values = dict(zip(columns, raw_values))
        values = {column: _value(raw_values[column]) for column in scalar_columns}
        counts = counters.setdefault(group_id, Counter())
        counts[("size",)] += 1
        for column in scalar_columns:
            condition = [] if column == "dataset" else [values["dataset"]]
            if column not in ("dataset", "target"):
                condition.append(values["target"])
            if column == "source_format" and "generator" in values:
                condition.append(values["generator"])
            counts[(column, *condition, values[column])] += 1
        for column in list_columns:
            for tag in _tags(raw_values[column]) or ["unknown"]:
                counts[(column, values["dataset"], values["target"], tag)] += 1
    for group_id, counts in sorted(counters.items()):
        rarity = min(totals[key] for key in counts if key != ("size",))
        flexible.append((group_id, list(indices_by_group[group_id]), counts, rarity))
    rng = np.random.default_rng(seed)
    noise = dict(zip([item[0] for item in flexible], rng.random(len(flexible))))
    flexible.sort(key=lambda item: (item[3], -len(item[1]), noise[item[0]]))

    def delta(counts, direction):
        return sum((2 * (current[key] - targets[key]) * direction * count + count * count)
                   * weights[key] for key, count in counts.items())

    assignments = {}
    final_targets = targets
    targets = current.copy()
    for group_id, indices, counts, _ in flexible:
        probability = fraction if row_probabilities is None else float(row_probabilities.loc[indices[0]])
        for key, count in counts.items():
            targets[key] += probability * count
        partition = "validate" if delta(counts, 1) < 0 else "train"
        assignments[group_id] = partition
        if partition == "validate":
            current.update(counts)
    targets = final_targets
    for _ in range(2):
        changed = False
        for group_id, indices, counts, _ in flexible:
            direction = -1 if assignments[group_id] == "validate" else 1
            if delta(counts, direction) < -1e-10:
                assignments[group_id] = "validate" if direction == 1 else "train"
                for key, count in counts.items():
                    current[key] += direction * count
                changed = True
        if not changed:
            break
    def candidates(partition, direction):
        ordered = sorted(
            (item for item in flexible if assignments[item[0]] == partition),
            key=lambda item: delta(item[2], direction),
        )
        selected = []
        seen = set()
        for item in ordered:
            signature = frozenset(item[2].items())
            if signature not in seen:
                selected.append(item)
                seen.add(signature)
            if len(selected) == 32:
                break
        return selected

    for _ in range(4):
        validation_candidates = candidates("validate", -1)
        training_candidates = candidates("train", 1)
        for _ in range(16):
            best = None
            for left in validation_candidates:
                for right in training_candidates:
                    cross = sum(count * right[2].get(key, 0) * weights[key]
                                for key, count in left[2].items())
                    improvement = delta(left[2], -1) + delta(right[2], 1) - 2 * cross
                    if improvement < -1e-10 and (best is None or improvement < best[0]):
                        best = improvement, left, right
            if best is None:
                break
            _, left, right = best
            assignments[left[0]] = "train"
            assignments[right[0]] = "validate"
            for key, count in left[2].items():
                current[key] -= count
            current.update(right[2])
            validation_candidates.remove(left)
            training_candidates.remove(right)
    for partition in ("train", "validate"):
        indices = [index for group_id, group_indices, _, _ in flexible
                   if assignments[group_id] == partition for index in group_indices]
        if indices:
            data.loc[indices, "partition"] = partition


def _mask_source_coverage(data):
    if "mask_source" not in data:
        return []
    columns = ["dataset", "target", "partition", "mask_source"]
    coverage = data[columns].copy()
    for column in columns:
        coverage[column] = coverage[column].map(_value).replace("", "unknown")
    return coverage.groupby(columns, sort=True, dropna=False).size().reset_index(name="rows").to_dict("records")


def _balance_record(column, condition, category, counts, denominators, multilabel=False):
    partitions = ("train", "validate", "test")
    shares = {part: counts[part] / denominators[part] if denominators[part] else None for part in partitions}
    gaps = []
    for first, second in (("train", "validate"), ("train", "test"), ("validate", "test")):
        gaps.append(abs(shares[first] - shares[second])
                    if shares[first] is not None and shares[second] is not None else None)
    common_support = all(denominators[part] > 0 for part in partitions)
    return {
        "feature": column, "condition": condition, "category": category,
        **{part + "_n": int(counts[part]) for part in partitions},
        **{part + "_total": int(denominators[part]) for part in partitions},
        **{part + "_share": shares[part] for part in partitions},
        "abs_gap": gaps[0], "abs_gap_train_test": gaps[1], "abs_gap_validate_test": gaps[2],
        "max_pairwise_gap": max(gaps) if common_support else None,
        "common_support": common_support,
        "missing_in_partition": any(counts[part] == 0 for part in partitions),
        "multilabel": multilabel,
    }


def _global_balance(data):
    partitions = ("train", "validate", "test")
    kept = data.loc[data["partition"].isin(partitions)]
    denominators = {part: int(kept["partition"].eq(part).sum()) for part in partitions}
    records = []
    for column in ("dataset", "target", "source_format"):
        if column not in kept:
            continue
        counts = {part: kept.loc[kept["partition"].eq(part), column].map(_value).value_counts()
                  for part in partitions}
        for category in sorted(set().union(*(set(values.index) for values in counts.values()))):
            records.append(_balance_record(column, {}, category,
                           {part: int(counts[part].get(category, 0)) for part in partitions}, denominators))
    return records


def _balance(data):
    records = []
    partitions = ("train", "validate", "test")
    combined = data.loc[data["partition"].isin(partitions)].copy()

    def add_record(column, condition, category, counts, denominators, multilabel=False):
        records.append(_balance_record(column, condition, category, counts, denominators, multilabel))

    for column in _BALANCE_COLUMNS:
        if column not in combined:
            continue
        condition = [] if column == "dataset" else ["dataset"]
        if column not in ("dataset", "target"):
            condition.append("target")
        if column == "source_format" and "generator" in combined:
            condition.append("generator")
        conditions = [((), combined)] if not condition else combined.groupby(condition, sort=True, dropna=False)
        for key, rows in conditions:
            key = key if isinstance(key, tuple) else (key,)
            counts = {
                part: rows.loc[rows["partition"].eq(part), column].map(_value).value_counts()
                for part in partitions
            }
            categories = sorted(set().union(*(set(value.index) for value in counts.values())))
            denominators = {part: int(counts[part].sum()) for part in partitions}
            for category in categories:
                add_record(column, dict(zip(condition, map(_value, key))), category,
                           {part: int(counts[part].get(category, 0)) for part in partitions}, denominators)
    for column in _BALANCE_LIST_COLUMNS:
        if column not in combined:
            continue
        for key, rows in combined.groupby(["dataset", "target"], sort=True, dropna=False):
            values = rows[column].map(lambda value: _tags(value) or ["unknown"])
            tags = sorted({tag for items in values for tag in items})
            denominators = {part: int(rows["partition"].eq(part).sum()) for part in partitions}
            for tag in tags:
                present = values.map(lambda items: tag in items)
                counts = {part: int((present & rows["partition"].eq(part)).sum()) for part in partitions}
                add_record(column, dict(zip(("dataset", "target"), map(_value, key))), tag,
                           counts, denominators, multilabel=True)
    return records


def _inventory_partitions(value):
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return {_partition(item) for item in value if _value(item)}
    text = _value(value)
    if not text:
        return set()
    if text.startswith("["):
        import json

        return {_partition(item) for item in json.loads(text)}
    return {_partition(item) for item in re.split(r"[,;|]", text) if item.strip()}


def _author_policies(data):
    reports = {}
    declared = data["author_partition"].copy()
    pinned = pd.Series("", index=data.index, dtype=object)
    scopes = pd.Series("three_way", index=data.index, dtype=object)
    data["split_policy"] = ""
    for dataset, rows in data.groupby("dataset", sort=True):
        inventory = set()
        if "author_split_inventory" in rows:
            for value in rows["author_split_inventory"]:
                inventory.update(_inventory_partitions(value))
        inventory.discard("")
        observed = set(rows["author_partition"]) - {""}
        basis = "inventory" if inventory else "inferred_from_selected_rows"
        source = inventory or observed
        if inventory and not observed <= inventory:
            raise ValueError(f"Author partition outside inventory for {dataset}: {observed - inventory}")
        if {"train", "validate", "test"} <= source:
            policy = "preserve_three"
            pinned.loc[rows.index] = rows["author_partition"]
        elif source == {"train", "test"}:
            policy = "derive_validation_from_train"
            pinned.loc[rows.index[rows["author_partition"].eq("test")]] = "test"
            scopes.loc[rows.index] = "train_validation"
        elif source == {"train", "validate"}:
            policy = "author_validation_becomes_test"
            heldout = rows.index[rows["author_partition"].eq("validate")]
            declared.loc[heldout] = "test"
            pinned.loc[heldout] = "test"
            scopes.loc[rows.index] = "train_validation"
        elif source == {"train"}:
            policy = "derive_validation_and_test_from_train"
        elif not source:
            policy = "grouped_three_way"
        elif "train" not in source:
            policy = "holdout_only"
            pinned.loc[rows.index] = rows["author_partition"].replace("", "test")
            if source == {"validate"}:
                declared.loc[rows.index] = "test"
                pinned.loc[rows.index] = "test"
        else:
            raise ValueError(f"Unsupported author partition inventory for {dataset}: {source}")
        data.loc[rows.index, "split_policy"] = policy
        reports[str(dataset)] = {
            "policy": policy, "source_partitions": sorted(source), "basis": basis,
        }
    return declared, pinned, scopes, reports


def build_composite_splits(frame, validation_fraction=0.15, seed=42,
                           near_duplicate_threshold=4,
                           test_fraction=0.15, author_validation_fraction=0.2,
                           near_duplicate_verifier=None, split_policy="global_balanced",
                           verified_group_cache=None):
    if split_policy not in {"global_balanced", "author_preserved"}:
        raise ValueError("split_policy must be global_balanced or author_preserved")
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("validation_fraction and test_fraction must be between zero and one")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction plus test_fraction must be less than one")
    if not 0 < author_validation_fraction < 1:
        raise ValueError("author_validation_fraction must be between zero and one")
    if not isinstance(near_duplicate_threshold, int) or not 0 <= near_duplicate_threshold <= 16:
        raise ValueError("near_duplicate_threshold must be an integer from zero to sixteen")
    from src.preprocessing.labels import normalize_labels

    from src.preprocessing.df40 import apply_source_group_policy
    from src.preprocessing.near_duplicates import PixelNearDuplicateVerifier

    normalized = apply_source_group_policy(normalize_labels(frame))
    lineage_report = normalized.attrs.get("df40_lineage_policy", {})
    normalization_report = normalized.attrs.get("label_normalization", {})
    if "target" not in frame:
        raise ValueError("Missing manifest columns: ['target']")
    if not normalized["target"].reset_index(drop=True).equals(frame["target"].reset_index(drop=True)):
        raise ValueError("Label normalization must preserve binary targets")
    data = _prepare(normalized)
    cache_path = None
    cache_exists = False
    if verified_group_cache is not None:
        from pathlib import Path

        if near_duplicate_verifier is not None:
            raise ValueError("A verified group cache cannot be combined with an injected verifier")
        cache_path = Path(verified_group_cache).expanduser().resolve()
        cache_exists = (cache_path / "manifest.json").is_file()
        if cache_path.exists() and not cache_path.is_dir():
            raise ValueError("The verified group cache must be a directory")
        if cache_path.is_dir() and any(cache_path.iterdir()) and not cache_exists:
            raise ValueError("Incomplete verified group cache; refusing to overwrite it")
    if cache_exists:
        from src.preprocessing.group_cache import load_verified_groups

        group_ids, previous_verification, cache_report = load_verified_groups(
            data, cache_path, near_duplicate_threshold,
        )
        data["group_id"] = group_ids
        candidate_pairs = int(previous_verification["candidate_pairs"])
        verification_report = {
            **previous_verification, "execution_mode": "reused_cache", "verified_now": False,
        }
    else:
        verifier = near_duplicate_verifier if near_duplicate_verifier is not None else PixelNearDuplicateVerifier(data)
        candidate_pairs = _connect(data, near_duplicate_threshold, verifier=verifier)
        if isinstance(verifier, PixelNearDuplicateVerifier):
            verification_report = verifier.report()
            for index in verifier.unreadable_indices:
                data.at[index, "flags"].add("near_duplicate_verification_missing")
                exclusions = _flags(data.at[index, "exclusion_flags"])
                exclusions.add("near_duplicate_verification_missing")
                data.at[index, "exclusion_flags"] = sorted(exclusions)
            data.loc[sorted(verifier.unreadable_indices), "eligible"] = False
            verifier.close()
        else:
            verification_report = {"mode": "injected_verifier", "candidate_pairs": candidate_pairs}
        verification_report = {
            **verification_report, "execution_mode": "verified_now", "verified_now": True,
        }
        cache_report = {"reused": False, "verified_now": True}
        if cache_path is not None:
            from src.preprocessing.group_cache import write_verified_groups

            cache_report = write_verified_groups(
                data, cache_path, near_duplicate_threshold, verification_report,
            )
    initial_ineligible = int((~data["eligible"]).sum())
    nonempty = data.loc[data["eligible"] & data["crop_sha256"].map(_value).ne("")]
    conflicting_hashes = nonempty.groupby("crop_sha256")["target"].nunique()
    conflicting_hashes = set(conflicting_hashes.index[conflicting_hashes.gt(1)])
    crop_conflict_indices = set(data.index[data["eligible"] & data["crop_sha256"].isin(conflicting_hashes)])
    from src.preprocessing.duplicates import matched_face_conflicts

    matched_conflict_indices = set(matched_face_conflicts(data))
    conflict_indices = sorted(crop_conflict_indices | matched_conflict_indices)
    for index in conflict_indices:
        data.at[index, "flags"].add("contradictory_duplicate_target")
    data.loc[conflict_indices, "eligible"] = False
    if split_policy == "global_balanced":
        data["partition"] = ""
        data["split_scope"] = "three_way"
        data["split_policy"] = "global_balanced"
        boundary_removed = 0
        policies = {}
        for dataset, rows in data.groupby("dataset", sort=True):
            source = set(rows["author_partition"]) - {""}
            if "author_split_inventory" in rows:
                for value in rows["author_split_inventory"]:
                    source.update(_inventory_partitions(value))
            source.discard("")
            policies[str(dataset)] = {
                "policy": "global_balanced", "source_partitions": sorted(source),
                "basis": "all_eligible_verified_groups", "author_parts_ignored": True,
            }
    else:
        declared, pinned, scopes, policies = _author_policies(data)
        priority = {"": 0, "train": 1, "validate": 2, "test": 3}
        declared_priority = declared.map(priority).where(data["eligible"], 0)
        group_priority = declared_priority.groupby(data["group_id"]).max()
        winners = data["group_id"].map(group_priority).map({0: "", 1: "train", 2: "validate", 3: "test"})
        removed = data.index[data["eligible"] & declared_priority.gt(0) & declared.ne(winners)]
        boundary_removed = len(removed)
        for index in removed:
            data.at[index, "flags"].add("author_split_overlap")
        data.loc[removed, "eligible"] = False
        pinned_priority = pinned.map(priority).where(data["eligible"], 0)
        group_pins = pinned_priority.groupby(data["group_id"]).max()
        data["partition"] = data["group_id"].map(group_pins).map({0: "", 1: "train", 2: "validate", 3: "test"})
        validation_groups = set(data.loc[data["eligible"] & scopes.eq("train_validation"), "group_id"])
        data["split_scope"] = np.where(data["group_id"].isin(validation_groups), "train_validation", "three_way")
    remaining = data.loc[data["eligible"] & data["crop_sha256"].map(_value).ne("")]
    remaining = remaining.sort_values("sample_id", kind="stable")
    removed = remaining.index[remaining["crop_sha256"].duplicated(keep="first")]
    duplicate_removed = len(removed)
    for index in removed:
        data.at[index, "flags"].add("exact_face_duplicate")
    data.loc[removed, "eligible"] = False
    three_way = data["eligible"] & data["partition"].eq("") & data["split_scope"].eq("three_way")
    temporary = data.loc[three_way].copy()
    _assign_flexible(temporary, test_fraction, seed + 1)
    test_indices = temporary.index[temporary["partition"].eq("validate")]
    data.loc[test_indices, "partition"] = "test"
    probabilities = pd.Series(0.0, index=data.index)
    probabilities.loc[data["partition"].eq("validate")] = 1.0
    flexible = data["eligible"] & data["partition"].eq("")
    probabilities.loc[flexible & data["split_scope"].eq("train_validation")] = author_validation_fraction
    probabilities.loc[flexible & data["split_scope"].eq("three_way")] = validation_fraction / (1 - test_fraction)
    _assign_flexible(data, validation_fraction, seed, row_probabilities=probabilities)
    data.loc[~data["eligible"], "partition"] = "excluded"
    overlap = {}
    kept = data.loc[data["eligible"]]
    for column in ("group_id", *_HASH_COLUMNS):
        hashes = kept.loc[kept[column].map(_value).ne("")]
        overlap[column] = int(hashes.groupby(column)["partition"].nunique().gt(1).sum())
    if any(overlap.values()):
        raise RuntimeError(f"Leakage survived split construction: {overlap}")
    balance = _balance(data)
    report = {
        "seed": seed, "split_policy": split_policy,
        "author_parts_ignored": split_policy == "global_balanced",
        "requested_train_fraction": 1 - validation_fraction - test_fraction,
        "requested_validation_fraction": validation_fraction,
        "requested_test_fraction": test_fraction,
        "author_train_validation_fraction": author_validation_fraction if split_policy == "author_preserved" else None,
        "author_split_policies": policies,
        "input_rows": len(data), "initially_ineligible": initial_ineligible,
        "label_normalization": normalization_report,
        "df40_lineage_policy": lineage_report,
        "near_duplicate_verification": verification_report,
        "group_cache": cache_report,
        "quality_policy": data.attrs.get("quality_policy", {}),
        "transformation_balance": "Canonical tag-presence marginals within dataset and target; empty tag lists are tracked as unknown. Multi-label shares use faces as the denominator.",
        "contradictory_crop_rows": len(crop_conflict_indices),
        "matched_face_conflict_rows": len(matched_conflict_indices),
        "contradictory_duplicate_rows": len(conflict_indices),
        "author_overlap_removed": boundary_removed,
        "exact_face_duplicates_removed": duplicate_removed,
        "near_duplicate_candidate_pairs": candidate_pairs,
        "near_duplicate_policy": "Perceptual hashes retrieve candidates only. Exact hashes or high-precision pixel verification establish must-link edges. Unreadable candidate rows are excluded. Verified groups stay intact under the selected split policy.",
        "overlap_groups": overlap, "balance": balance,
        "global_balance": _global_balance(data),
        "max_common_support_gap": max((record["max_pairwise_gap"] for record in balance
                                       if record["common_support"]), default=None),
        "missing_strata": [record for record in balance if record["missing_in_partition"]],
        "unknown_source_rows": int(data["source_group"].map(_value).eq("").sum()),
        "unknown_identity_rows": int(data["identity_group"].map(_value).eq("").sum()),
        "measured_feature_population_balance": False,
        "final_crop_format": "PNG",
        "source_format_balance": "Decoded original image format, conditional on dataset, target and generator",
        "face_mask_crop_occupancy": {
            "feature": "face_mask_fraction_of_crop",
            "bin_feature": "face_mask_fraction_of_crop_bin",
            "edges": list(_MASK_OCCUPANCY_EDGES),
            "intervals": list(_MASK_OCCUPANCY_LABELS),
            "last_interval_closed": True,
            "condition": ["dataset", "target"],
            "mask_source_column": "mask_source",
            "mask_source_balanced": False,
            "definition": "Area of the chosen face mask inside the actual crop divided by actual crop area before resizing",
            "missing_policy": "Unknown when no mask measurement is available; never inferred from a bounding box",
            "known_rows": int(data["face_mask_fraction_of_crop"].notna().sum()),
            "unknown_rows": int(data["face_mask_fraction_of_crop"].isna().sum()),
            "exclusion_threshold_inclusive": 0.25,
            "excluded_small_mask_rows": int(data["flags"].map(lambda flags: "face_mask_too_small" in flags).sum()),
            "excluded_missing_mask_rows": int(data["flags"].map(lambda flags: "face_mask_missing" in flags).sum()),
            "unknown_excluded_by_occupancy": True,
        },
        "mask_source_coverage": _mask_source_coverage(data),
        "largest_group": int(data.groupby("group_id").size().max()) if len(data) else 0,
    }
    result = {}
    for partition in ("train", "validate", "test", "excluded"):
        subset = data.loc[data["partition"].eq(partition)].copy()
        subset["flags"] = subset["flags"].map(lambda values: sorted(values))
        subset = subset.sort_values(["dataset", "sample_id"], kind="stable").reset_index(drop=True)
        result[partition] = subset
        report[f"{partition}_rows"] = len(subset)
    denominator = max(len(result["train"]) + len(result["validate"]) + len(result["test"]), 1)
    report["actual_train_fraction"] = len(result["train"]) / denominator
    report["actual_validation_fraction"] = len(result["validate"]) / denominator
    report["actual_test_fraction"] = len(result["test"]) / denominator
    result["report"] = report
    result["learn"] = result["train"]
    return result
