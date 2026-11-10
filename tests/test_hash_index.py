import itertools
import random
import unittest

import pandas as pd

from src.preprocessing.splits import _connect


def manifest(values, ineligible=()):
    return pd.DataFrame([
        {
            "sample_id": f"sample-{index:04d}", "dataset": "example",
            "eligible": index not in ineligible,
            "phash": "" if value is None else f"{value:016x}",
            "source_group": "", "identity_group": "",
            "image_sha256": "", "decoded_sha256": "", "crop_sha256": "",
        }
        for index, value in enumerate(values)
    ])


def exhaustive(values, eligible, sample_ids, threshold):
    adjacency = [set() for _ in values]
    pairs = 0
    for left, right in itertools.combinations(range(len(values)), 2):
        if not eligible[left] or not eligible[right] or values[left] is None or values[right] is None:
            continue
        if (values[left] ^ values[right]).bit_count() <= threshold:
            pairs += 1
            adjacency[left].add(right)
            adjacency[right].add(left)
    labels = {}
    for start in range(len(values)):
        if start in labels:
            continue
        component, pending = set(), [start]
        while pending:
            index = pending.pop()
            if index not in component:
                component.add(index)
                pending.extend(adjacency[index] - component)
        group_id = min(sample_ids[index] for index in component)
        labels.update({index: group_id for index in component})
    return pairs, [labels[index] for index in range(len(values))]


class ExactHashNeighborTests(unittest.TestCase):
    def assert_matches_exhaustive(self, values, threshold, ineligible=()):
        data = manifest(values, ineligible)
        count, labels = exhaustive(values, data["eligible"].tolist(), data["sample_id"].tolist(), threshold)
        self.assertEqual(_connect(data, threshold), count)
        self.assertEqual(data["group_id"].tolist(), labels)
        shuffled = manifest(values, ineligible).sample(frac=1, random_state=29).reset_index(drop=True)
        self.assertEqual(_connect(shuffled, threshold), count)
        expected = dict(zip(data["sample_id"], labels))
        self.assertEqual(dict(zip(shuffled["sample_id"], shuffled["group_id"])), expected)

    def test_chunk_thresholds_match_exhaustive_for_random_and_clustered_hashes(self):
        rng = random.Random(71)
        values = [rng.getrandbits(64) for _ in range(60)]
        for base in values[:12]:
            for distance in range(1, 6):
                values.append(base ^ sum(1 << position for position in rng.sample(range(64), distance)))
        values.extend(values[:8] * 3)
        values.extend([None, None, 0, (1 << 64) - 1])
        for threshold in (0, 1, 2, 4):
            with self.subTest(threshold=threshold):
                self.assert_matches_exhaustive(values, threshold, ineligible={2, 9, 120})

    def test_repeated_hashes_keep_row_pair_multiplicity_and_exclude_ineligible_hashes(self):
        values = [0, 0, 0, 1, 1, 3, (1 << 64) - 1, (1 << 64) - 1, 0, None]
        for threshold, expected_pairs in [(0, 5), (1, 13), (2, 16), (4, 16)]:
            with self.subTest(threshold=threshold):
                data = manifest(values, ineligible={8})
                self.assertEqual(_connect(data, threshold), expected_pairs)
                self.assert_matches_exhaustive(values, threshold, ineligible={8})

    def test_hashes_on_contiguous_chunk_boundaries_keep_exact_hamming_neighbors(self):
        four_bits = sum(1 << position for position in (0, 13, 26, 39))
        five_bits = four_bits | (1 << 52)
        values = [0, four_bits, five_bits, (1 << 64) - 1, ((1 << 64) - 1) ^ four_bits]
        self.assert_matches_exhaustive(values, 4)

    def test_larger_threshold_tree_fallback_keeps_exhaustive_semantics(self):
        rng = random.Random(19)
        values = [rng.getrandbits(64) for _ in range(60)]
        values.extend([0, 0, (1 << 5) - 1, (1 << 8) - 1, (1 << 16) - 1, None])
        for threshold in (5, 8, 16):
            with self.subTest(threshold=threshold):
                self.assert_matches_exhaustive(values, threshold)


if __name__ == "__main__":
    unittest.main()
