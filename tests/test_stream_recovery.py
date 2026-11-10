import contextlib
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zlib

import pandas as pd

from scripts import recover_df40_tar as recovery


def record(member, content, index):
    method = 'pixart' if member.startswith('pixart') else 'e4e'
    return {'archive_member': member, 'archive_partition': 'test', 'native_method': method,
            'zip_crc32': f'{zlib.crc32(content):08x}', 'uncompressed_bytes': len(content),
            'sample_id': f'selected-{index}', 'archive_id': method + '-test'}


def archive_bytes(members):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name, content in members:
            member = tarfile.TarInfo('DF40/deepfakes_detection_datasets/DF40/' + name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return buffer.getvalue()


def run_fixture(root, records, members, probe_bytes=None):
    frame = pd.DataFrame(records)
    frame.to_parquet(root / 'inventory.parquet', index=False)
    frame.to_parquet(root / 'selection.parquet', index=False)
    payload = archive_bytes(members)
    response = MagicMock()
    response.__enter__.return_value = response
    response.status_code = 200
    response.headers = {'Content-Type': 'application/gzip', 'Content-Length': str(len(payload))}
    response.raw = io.BytesIO(payload)
    with patch.object(recovery.requests, 'get', return_value=response), \
         patch.object(recovery, 'ARCHIVE_BYTES', len(payload)), \
         patch.object(recovery, 'ARCHIVE_SHA256', hashlib.sha256(payload).hexdigest()), \
         contextlib.redirect_stdout(io.StringIO()):
        status = recovery.recover(root, probe_bytes=probe_bytes or len(payload))
    progress = json.loads((root / 'metadata/stream_progress.json').read_text())
    checkpoint = json.loads((root / 'metadata/stream_downloads.json').read_text())
    return status, progress, checkpoint


class StreamRecoveryTests(unittest.TestCase):
    def test_exact_selected_bytes_are_recovered_and_donor_images_omitted(self):
        members = [('pixart/cdf/Fake/id0/000.png', b'original-pixart-bytes'),
                   ('e4e/e4e/ff/000/00001.jpg', b'original-e4e-bytes')]
        records = [record(name, content, index) for index, (name, content) in enumerate(members)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, progress, checkpoint = run_fixture(
                root, records, [('Celeb-DF-v2/Celeb-real/000.png', b'unselected-donor'), *members])
            self.assertEqual(status, 0)
            self.assertEqual(progress['recovered_images'], 2)
            self.assertEqual(progress['first_native_verified']['matched_by'], 'path_crc_size')
            for name, content in members:
                self.assertEqual((root / 'images/test' / name).read_bytes(), content)
            self.assertFalse((root / 'images/test/Celeb-DF-v2').exists())
            self.assertEqual(sum(len(entry['files']) for entry in checkpoint.values()), 2)

    def test_probe_limit_stops_before_native_proof_without_writing_images(self):
        target = 'pixart/cdf/Fake/id0/000.png'
        expected = b'original-pixart-bytes'
        donor = hashlib.shake_256(b'donor').digest(2 * 1024 * 1024)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, progress, checkpoint = run_fixture(
                root, [record(target, expected, 0)],
                [('Celeb-DF-v2/Celeb-real/000.png', donor), (target, expected)], probe_bytes=32_000)
            self.assertEqual(status, 2)
            self.assertEqual(progress['status'], 'probe_limit_reached')
            self.assertIsNone(progress['first_native_verified'])
            self.assertEqual(progress['compressed_bytes'], 32_000)
            self.assertFalse((root / 'images').exists())
            self.assertFalse(any(entry['files'] for entry in checkpoint.values()))

    def test_wrong_crc_never_proves_a_source_or_publishes_an_image(self):
        target = 'pixart/cdf/Fake/id0/000.png'
        expected = b'original-pixart-bytes'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, progress, _ = run_fixture(root, [record(target, expected, 0)],
                                              [(target, b'changed-pixart-bytes!')])
            self.assertEqual(status, 3)
            self.assertIsNone(progress['first_native_verified'])
            self.assertEqual(progress['recovered_images'], 0)
            self.assertFalse((root / 'images').exists())

    def test_renamed_source_requires_an_unambiguous_primary_crc_mapping(self):
        content = b'same-native-bytes'
        records = [record('pixart/cdf/Fake/id0/000.png', content, 0),
                   record('pixart/cdf/Fake/id1/000.png', content, 1)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, progress, _ = run_fixture(root, records, [('pixart/renamed.png', content)])
            self.assertEqual(status, 3)
            self.assertIsNone(progress['first_native_verified'])
            self.assertFalse((root / 'images').exists())


    def test_existing_file_without_stream_provenance_is_not_claimed_as_recovered(self):
        target = 'pixart/cdf/Fake/id0/000.png'
        expected = b'original-pixart-bytes'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / 'images/test' / target
            destination.parent.mkdir(parents=True)
            destination.write_bytes(expected)
            status, progress, checkpoint = run_fixture(
                root, [record(target, expected, 0)], [(target, b'other-source-bytes!!')])
            self.assertEqual(status, 3)
            self.assertEqual(progress['recovered_images'], 0)
            self.assertFalse(any(entry['files'] for entry in checkpoint.values()))
            self.assertEqual(destination.read_bytes(), expected)


if __name__ == '__main__':
    unittest.main()
