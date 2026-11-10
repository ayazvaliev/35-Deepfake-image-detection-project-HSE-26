import unittest

import pandas as pd

from src.preprocessing.duplicates import matched_face_conflicts


class MatchedFaceConflictTests(unittest.TestCase):
    def test_same_source_face_conflict_across_datasets(self):
        data = pd.DataFrame([
            dict(dataset='a', target=0, decoded_sha256='same', bbox=[10, 10, 40, 40], eligible=True),
            dict(dataset='b', target=1, decoded_sha256='same', bbox=[11, 10, 40, 40], eligible=True),
        ])
        self.assertEqual(matched_face_conflicts(data), {0, 1})

    def test_mixed_image_with_distinct_faces_is_valid(self):
        data = pd.DataFrame([
            dict(dataset='a', target=0, image_sha256='same', bbox=[10, 10, 40, 40]),
            dict(dataset='b', target=1, image_sha256='same', bbox=[100, 10, 40, 40]),
        ])
        self.assertEqual(matched_face_conflicts(data), set())

    def test_matching_faces_with_same_target_are_valid(self):
        data = pd.DataFrame([
            dict(dataset='a', target=0, image_sha256='same', bbox=[10, 10, 40, 40]),
            dict(dataset='b', target=0, image_sha256='same', bbox=[10, 10, 40, 40]),
        ])
        self.assertEqual(matched_face_conflicts(data), set())

    def test_ineligible_or_missing_boxes_do_not_conflict(self):
        data = pd.DataFrame([
            dict(dataset='a', target=0, image_sha256='same', bbox=[10, 10, 40, 40], eligible=True),
            dict(dataset='b', target=1, image_sha256='same', bbox=[10, 10, 40, 40], eligible=False),
            dict(dataset='c', target=1, image_sha256='same', bbox=None, eligible=True),
        ])
        self.assertEqual(matched_face_conflicts(data), set())


if __name__ == '__main__':
    unittest.main()
