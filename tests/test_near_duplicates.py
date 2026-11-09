from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd

from src.preprocessing.near_duplicates import PixelNearDuplicateVerifier
from src.preprocessing.splits import build_composite_splits


def portrait(alternate=False):
    yy, xx = np.mgrid[:256, :256]
    image = np.empty((256, 256, 3), dtype=np.uint8)
    image[:] = np.stack([65 + xx // 5, 85 + yy // 5, 105 + (xx + yy) // 10], axis=-1)
    cv2.ellipse(image, (128, 132), (77, 106), 0, 0, 360, (155, 184, 215), -1)
    cv2.ellipse(image, (128, 53), (77, 43), 0, 180, 360, (34, 42, 53), -1)
    shift = 15 if alternate else 0
    cv2.ellipse(image, (98 - shift, 107), (14, 7), 0, 0, 360, (28, 37, 49), -1)
    cv2.ellipse(image, (158 + shift, 107), (14, 7), 0, 0, 360, (28, 37, 49), -1)
    cv2.line(image, (128, 120), (136 + shift, 156), (104, 132, 160), 4)
    cv2.ellipse(image, (128, 181 + shift), (28, 8), 0, 0, 180, (62, 74, 126), 5)
    return cv2.GaussianBlur(image, (3, 3), 0.7)


def records(paths, **extras):
    return pd.DataFrame([
        {"dataset": "example", "sample_id": f"sample-{index}", "target": 1,
         "source_group": f"image-{index}", "face_mask_fraction_of_crop": 0.5,
         "phash": "0000000000000000", "crop_path": str(path), "path": str(path), **extras}
        for index, path in enumerate(paths)
    ])


class PixelNearDuplicateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def save(self, name, image, quality=95):
        path = self.root / name
        self.assertTrue(cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality]))
        return path

    def test_jpeg85_and_resized_copies_are_verified(self):
        original = self.save("original.png", portrait())
        jpeg = self.save("copy.jpg", portrait(), quality=85)
        resized = self.save("resized.png", cv2.resize(portrait(), (384, 384), interpolation=cv2.INTER_CUBIC))
        verifier = PixelNearDuplicateVerifier(records([original, jpeg, resized]))
        self.addCleanup(verifier.close)
        self.assertTrue(verifier(0, 1))
        self.assertTrue(verifier(0, 2))
        self.assertEqual(verifier.report()["verified_pairs"], 2)

    def test_different_face_like_images_with_the_same_candidate_hash_are_rejected(self):
        first = self.save("first.png", portrait())
        second = self.save("second.png", portrait(alternate=True))
        verifier = PixelNearDuplicateVerifier(records([first, second]))
        self.addCleanup(verifier.close)
        self.assertFalse(verifier(0, 1))
        self.assertEqual(verifier.report()["rejected_pairs"], 1)
        self.assertEqual(verifier.report()["unreadable_pairs"], 0)

    def test_low_texture_similarity_cannot_establish_a_near_duplicate(self):
        first = self.save("flat-a.png", np.full((256, 256, 3), 127, np.uint8))
        second = self.save("flat-b.png", np.full((256, 256, 3), 128, np.uint8))
        verifier = PixelNearDuplicateVerifier(records([first, second]))
        self.addCleanup(verifier.close)
        self.assertFalse(verifier(0, 1))
        self.assertEqual(verifier.report()["low_texture_rejections"], 1)

    def test_matching_originals_can_verify_different_crop_views(self):
        source = self.save("source.png", portrait())
        resized = self.save("source-resized.jpg", cv2.resize(portrait(), (384, 384)), quality=95)
        crop_a = self.save("crop-a.png", portrait())
        crop_b = self.save("crop-b.png", portrait(alternate=True))
        frame = records([crop_a, crop_b])
        frame["input_path"] = [str(source), str(resized)]
        verifier = PixelNearDuplicateVerifier(frame)
        self.addCleanup(verifier.close)
        self.assertTrue(verifier(0, 1))
        self.assertEqual(verifier.report()["verified_original_pairs"], 1)

    def test_equal_phash_candidates_are_each_verified_before_grouping(self):
        first = self.save("first.png", portrait())
        copy = self.save("copy.jpg", portrait(), quality=85)
        other = self.save("other.png", portrait(alternate=True))
        frame = records([first, copy, other])
        frame["author_split"] = ["train", "test", "train"]
        result = build_composite_splits(frame)
        audit = result["report"]["near_duplicate_verification"]
        self.assertEqual(audit["candidate_pairs"], 3)
        self.assertEqual(audit["verified_pairs"], 1)
        self.assertEqual(audit["rejected_pairs"], 2)
        self.assertEqual(result["test"]["sample_id"].tolist(), ["sample-1"])
        self.assertEqual(result["train"]["sample_id"].tolist(), ["sample-2"])
        self.assertEqual(result["excluded"]["sample_id"].tolist(), ["sample-0"])

    def test_unreadable_candidate_rows_are_flagged_and_excluded(self):
        valid = self.save("valid.png", portrait())
        result = build_composite_splits(records([self.root / "missing.png", valid]))
        audit = result["report"]["near_duplicate_verification"]
        self.assertEqual(audit["unreadable_pairs"], 1)
        self.assertEqual(audit["unreadable_rows"], 1)
        excluded = result["excluded"].iloc[0]
        self.assertEqual(excluded["sample_id"], "sample-0")
        self.assertIn("near_duplicate_verification_missing", excluded["flags"])
        self.assertIn("near_duplicate_verification_missing", excluded["exclusion_flags"])
        self.assertEqual(sum(len(result[name]) for name in ("train", "validate", "test")), 1)

    def test_exact_sha_evidence_remains_valid_without_pixel_reads(self):
        verifier = PixelNearDuplicateVerifier(records(["absent-a", "absent-b"], decoded_sha256="verified-exact-source"))
        self.addCleanup(verifier.close)
        self.assertTrue(verifier(0, 1))
        self.assertEqual(verifier.report()["exact_metadata_pairs"], 1)
        self.assertEqual(verifier.report()["decoded_views"], 0)


if __name__ == "__main__":
    unittest.main()
