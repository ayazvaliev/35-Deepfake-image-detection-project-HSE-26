from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image, JpegImagePlugin

from src.preprocessing.jpeg import extract_jpeg_metadata, ijg_quantization_table, jpeg_quality_bin


def image_bytes(quality=95, mode="RGB", **options):
    yy, xx = np.mgrid[:48, :64]
    pixels = np.stack([(xx * 3) % 256, (yy * 5) % 256, ((xx + yy) * 2) % 256], axis=-1).astype(np.uint8)
    image = Image.fromarray(pixels).convert(mode)
    stream = BytesIO()
    image.save(stream, format="JPEG", quality=quality, **options)
    return stream.getvalue()


def header_segments(content):
    position = 2
    segments = []
    while position < len(content):
        if content[position] != 255:
            raise ValueError("Expected JPEG marker")
        marker = content[position + 1]
        length = int.from_bytes(content[position + 2:position + 4], "big")
        end = position + 2 + length
        segments.append((marker, content[position:end]))
        if marker == 218:
            return segments
        position = end
    raise ValueError("JPEG has no start-of-scan marker")


class JpegCodingMetadataTests(unittest.TestCase):
    def test_all_standard_ijg_quality_levels_are_recovered_exactly(self):
        for quality in range(1, 101):
            with self.subTest(quality=quality):
                result = extract_jpeg_metadata(image_bytes(quality))
                self.assertEqual(result["source_jpeg_quality"], quality)
                self.assertEqual(result["source_jpeg_quality_level"], str(quality))
                self.assertEqual(result["source_jpeg_quality_status"], "exact")
                self.assertEqual(result["source_jpeg_quality_method"], "exact_ijg_qtables")

    def test_quality_bins_keep_95_and_100_separate_and_handle_nullable_values(self):
        qualities = [1, 49, 50, 74, 75, 89, 90, 94, 95, 96, 99, 100]
        expected = ["<=49", "<=49", "50-74", "50-74", "75-89", "75-89",
                    "90-94", "90-94", "95", "96-99", "96-99", "100"]
        self.assertEqual([jpeg_quality_bin(value) for value in qualities], expected)
        nullable = pd.Series([95, pd.NA, 100], dtype="Int16")
        self.assertEqual(nullable.map(jpeg_quality_bin).tolist(), ["95", "unknown", "100"])
        for invalid in (None, np.nan, np.inf, -1, 0, 101, 95.5, "custom"):
            self.assertEqual(jpeg_quality_bin(invalid), "unknown")

    def test_subsampling_progressive_and_grayscale_are_header_metadata(self):
        for sampling, expected in ((0, "4:4:4"), (1, "4:2:2"), (2, "4:2:0")):
            result = extract_jpeg_metadata(image_bytes(95, subsampling=sampling, progressive=True))
            self.assertEqual(result["source_jpeg_subsampling"], expected)
            self.assertTrue(result["source_jpeg_progressive"])
            self.assertEqual(result["source_jpeg_quality"], 95)
        gray = extract_jpeg_metadata(image_bytes(100, mode="L"))
        self.assertEqual(gray["source_jpeg_subsampling"], "4:0:0")
        self.assertEqual(gray["source_jpeg_quality"], 100)
        self.assertFalse(gray["source_jpeg_progressive"])

    def test_custom_chroma_or_luma_tables_are_unknown_even_with_a_near_exact_standard_table(self):
        standard = [list(ijg_quantization_table(95)), list(ijg_quantization_table(95, chroma=True))]
        for table_id in (0, 1):
            tables = [table.copy() for table in standard]
            tables[table_id][17] += 1
            result = extract_jpeg_metadata(image_bytes(-1, qtables=tables))
            self.assertIsNone(result["source_jpeg_quality"])
            self.assertEqual(result["source_jpeg_quality_bin"], "unknown")
            self.assertEqual(result["source_jpeg_quality_status"], "custom")
            self.assertEqual(result["source_jpeg_quality_method"], "custom_qtables")
            self.assertEqual(json.loads(result["source_jpeg_qtables"])[str(table_id)], tables[table_id])
        mixed = [list(ijg_quantization_table(95)), list(ijg_quantization_table(100, chroma=True))]
        self.assertIsNone(extract_jpeg_metadata(image_bytes(-1, qtables=mixed))["source_jpeg_quality"])

    def test_header_through_first_sos_is_sufficient_without_decoding_baseline_or_progressive_pixels(self):
        for progressive in (False, True):
            payload = image_bytes(95, progressive=progressive)
            header = b"\xff\xd8" + b"".join(segment for _, segment in header_segments(payload))
            self.assertLess(len(header), len(payload))
            with patch.object(JpegImagePlugin.JpegImageFile, "load", side_effect=AssertionError("Pixel decoding is forbidden")):
                result = extract_jpeg_metadata(header)
            self.assertEqual(result["source_jpeg_quality"], 95)
            self.assertEqual(result["source_jpeg_progressive"], progressive)
            self.assertEqual(result["source_jpeg_qtable_hash"], extract_jpeg_metadata(payload)["source_jpeg_qtable_hash"])

    def test_missing_referenced_chroma_table_is_unknown_instead_of_a_luma_only_guess(self):
        segments = header_segments(image_bytes(95, progressive=True))
        retained = [segment for marker, segment in segments
                    if not (marker == 219 and segment[4] & 15 == 1)]
        self.assertLess(len(retained), len(segments))
        result = extract_jpeg_metadata(b"\xff\xd8" + b"".join(retained))
        self.assertIsNone(result["source_jpeg_quality"])
        self.assertEqual(result["source_jpeg_quality_bin"], "unknown")
        self.assertNotEqual(result["source_jpeg_quality_status"], "exact")

    def test_bytes_path_stream_and_open_pillow_image_have_identical_table_hashes(self):
        payload = image_bytes(95, subsampling=0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "misleading-extension.png"
            path.write_bytes(payload)
            expected = extract_jpeg_metadata(payload)
            self.assertEqual(extract_jpeg_metadata(path), expected)
            self.assertEqual(extract_jpeg_metadata(BytesIO(payload)), expected)
            self.assertEqual(extract_jpeg_metadata(memoryview(payload)), expected)
            with Image.open(BytesIO(payload)) as source:
                self.assertEqual(extract_jpeg_metadata(source), expected)
                self.assertFalse(source.fp.closed)
        quality100 = extract_jpeg_metadata(image_bytes(100))
        self.assertNotEqual(quality100["source_jpeg_qtable_hash"], expected["source_jpeg_qtable_hash"])
        self.assertEqual(len(expected["source_jpeg_qtable_hash"]), 64)

    def test_png_is_not_jpeg_and_unreadable_data_stays_unknown(self):
        stream = BytesIO()
        Image.new("RGB", (20, 20), (120, 80, 40)).save(stream, format="PNG")
        result = extract_jpeg_metadata(stream.getvalue())
        self.assertEqual(result["source_jpeg_quality_bin"], "not_jpeg")
        self.assertEqual(result["source_jpeg_quality_status"], "not_jpeg")
        self.assertIsNone(result["source_jpeg_quality"])
        self.assertIsNone(result["source_jpeg_qtable_hash"])
        broken = extract_jpeg_metadata(b"\xff\xd8\xff\xdb\x00")
        self.assertEqual(broken["source_jpeg_quality_bin"], "unknown")
        self.assertEqual(broken["source_jpeg_quality_status"], "unreadable")
        self.assertTrue(broken["source_jpeg_metadata_error"])


if __name__ == "__main__":
    unittest.main()
