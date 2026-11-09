import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from src.notebook_src import of_data
from src.notebook_src.of_crops import match_detector
from src.notebook_src.of_data import crop_face
from src.notebook_src.of_dedup import bitwise_count, close_pairs, groups_from_pairs


class NotebookHelperTests(unittest.TestCase):
    def test_crop_keeps_missing_pixels_gray(self):
        image = np.full((6, 8, 3), 25, dtype=np.uint8)
        crop = crop_face(image, -2, 1, 4, 4, scale=1, out=4)
        np.testing.assert_array_equal(crop[:, :2], 127)
        np.testing.assert_array_equal(crop[:, 2:], 25)
        self.assertEqual(crop_face(image, 1, 1, 3, 4).shape, (256, 256, 3))

    def test_popcount_matches_integer_hamming(self):
        rng = np.random.default_rng(42)
        values = rng.integers(0, np.iinfo(np.uint64).max, 1000, dtype=np.uint64)
        expected = np.array([int(value).bit_count() for value in values])
        np.testing.assert_array_equal(bitwise_count(values), expected)
        self.assertEqual(int(bitwise_count(np.uint64(2**64 - 1))), 64)

    def test_candidate_search_and_transitive_groups(self):
        values = np.array([0, 1, 3, 15, 2**63], dtype=np.uint64)
        pairs = close_pairs(values, max_dist=1)
        expected = {(i, j) for i in range(len(values)) for j in range(i + 1, len(values))
                    if (int(values[i]) ^ int(values[j])).bit_count() <= 1}
        self.assertEqual(set(pairs[["i", "j"]].itertuples(index=False, name=None)), expected)
        groups = groups_from_pairs(len(values), pairs)
        self.assertEqual(groups[0], groups[2])
        self.assertNotEqual(groups[0], groups[3])

    def test_detection_matches_face_in_the_same_frame(self):
        faces = pd.DataFrame({"split": ["Train", "Val"], "image_id": [7, 7],
                              "x": [10, 10], "y": [20, 20], "w": [40, 40], "h": [50, 50]}, index=[9, 4])
        detections = pd.DataFrame({"split": ["Train"], "image_id": [7], "det_x": [10],
                                   "det_y": [20], "det_w": [40], "det_h": [50], "det_score": [0.99]}, index=[32])
        matches = match_detector(faces, detections)
        self.assertEqual(matches.loc[9, "det_idx"], 32)
        self.assertEqual(matches.loc[4, "det_idx"], -1)

    def test_subset_loader_preserves_face_targets_and_author_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations = root / "annotations"
            images = root / "images"
            annotations.mkdir()
            directories = {}
            for split in of_data.SPLITS:
                directory = images / split
                directory.mkdir(parents=True)
                directories[split] = [directory]
                cv2.imwrite(str(directory / "present.jpg"), np.full((50, 40, 3), 100, np.uint8))
                payload = {"images": [{"id": 1, "file_name": "present.jpg", "width": 40, "height": 50},
                                       {"id": 2, "file_name": "missing.jpg", "width": 40, "height": 50}],
                           "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 2, 10, 20], "area": 200, "segmentation": [[1, 2, 11, 2, 11, 22]]},
                                           {"id": 2, "image_id": 1, "category_id": 1, "bbox": [20, 2, 10, 20], "area": 200, "segmentation": [[20, 2, 30, 2, 30, 22]]},
                                           {"id": 3, "image_id": 2, "category_id": 1, "bbox": [1, 2, 10, 20], "area": 200, "segmentation": [[1, 2, 11, 2, 11, 22]]}]}
                (annotations / f"{split}_poly.json").write_text(json.dumps(payload))
            with patch.multiple(of_data, DATA=root, ANNOT=annotations, IMAGES=images, _IMAGE_DIRS=directories):
                frames, faces = of_data.load_all()
            self.assertEqual(len(frames), 3)
            self.assertEqual(len(faces), 6)
            self.assertEqual(set(faces["label"]), {"real", "fake"})
            self.assertEqual(set(frames["split"]), set(of_data.SPLITS))
            self.assertTrue((faces.groupby(["split", "image_id"])["category_id"].nunique() == 2).all())


if __name__ == "__main__":
    unittest.main()
