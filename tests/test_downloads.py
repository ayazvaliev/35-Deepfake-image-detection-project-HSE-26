import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import download_assets as assets


PAYLOAD = bytes(range(251)) * 14_000


class DownloadHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def do_GET(self):
        server = self.server
        server.calls.append(dict(self.headers))
        if server.mode == 'unauthorized':
            self.send_response(401)
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        if server.mode == 'html':
            body = b'<html>sign in</html>'
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        range_header = self.headers.get('Range')
        offset = int(range_header.removeprefix('bytes=').removesuffix('-')) if range_header else 0
        if offset >= len(server.payload):
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{len(server.payload)}')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        ranged = bool(range_header) and server.mode != 'ignore_range'
        end = len(server.payload) - 1
        if server.mode == 'capped':
            ranged = True
            end = min(offset + 1_000_000 - 1, end)
        if server.mode == 'wrong_start':
            offset += 1
        body = server.payload[offset:end + 1] if ranged else server.payload
        self.send_response(206 if ranged else 200)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('ETag', '"changed"' if server.mode == 'changed_validator' else '"source-v1"')
        if ranged:
            total = len(server.payload) + int(server.mode == 'wrong_total')
            self.send_header('Content-Range', f'bytes {offset}-{end}/{total}')
        self.end_headers()
        if server.mode == 'truncate_once' and not server.truncated:
            server.truncated = True
            self.wfile.write(body[:1_500_000])
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_RDWR)
            self.close_connection = True
            return
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


class DownloadResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), DownloadHandler)
        self.server.daemon_threads = True
        self.server.mode = 'normal'
        self.server.payload = PAYLOAD
        self.server.calls = []
        self.server.truncated = False
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/data'
        self.destination = self.root / 'archive.bin'
        self.part = self.root / 'archive.bin.part'
        self.checksum = hashlib.sha256(PAYLOAD).hexdigest()
        self.progress = patch.object(assets, 'tqdm')
        self.progress.start()

    def tearDown(self):
        self.progress.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temporary.cleanup()

    def run_download(self, **kwargs):
        return assets.download(self.url, self.destination, self.checksum, expected_size=len(PAYLOAD), **kwargs)

    def checkpoint(self, **changes):
        state = {'source_url': self.url, 'expected_checksum': self.checksum,
                 'algorithm': 'sha256', 'expected_size': len(PAYLOAD),
                 'response_size': len(PAYLOAD), 'etag': '"source-v1"', 'last_modified': None}
        state.update(changes)
        (self.root / 'archive.bin.part.download.json').write_text(json.dumps(state))

    def test_truncated_response_resumes_and_verifies_every_byte(self):
        self.server.mode = 'truncate_once'
        with self.assertRaises(requests.RequestException):
            self.run_download()
        offset = self.part.stat().st_size
        self.assertGreater(offset, 0)
        self.assertLess(offset, len(PAYLOAD))
        self.assertFalse(self.destination.exists())
        self.run_download()
        self.assertEqual(self.server.calls[1]['Range'], f'bytes={offset}-')
        self.assertEqual(self.server.calls[1]['If-Range'], '"source-v1"')
        self.assertEqual(self.server.calls[1]['Accept-Encoding'], 'identity')
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)
        self.assertFalse(self.part.exists())
        marker = json.loads((self.root / 'archive.bin.download.json').read_text())
        self.assertEqual(marker['sha256'], self.checksum)

    def test_legacy_partial_with_authoritative_hash_can_resume(self):
        self.part.write_bytes(PAYLOAD[:12345])
        self.run_download()
        self.assertEqual(self.server.calls[0]['Range'], 'bytes=12345-')
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)

    def test_ignored_range_restarts_instead_of_appending(self):
        self.server.mode = 'ignore_range'
        self.part.write_bytes(PAYLOAD[:12345])
        self.run_download()
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)
        self.assertEqual(self.destination.stat().st_size, len(PAYLOAD))

    def test_wrong_range_start_preserves_the_partial_file(self):
        self.server.mode = 'wrong_start'
        original = PAYLOAD[:12345]
        self.part.write_bytes(original)
        with self.assertRaisesRegex(RuntimeError, 'Content-Range does not match'):
            self.run_download()
        self.assertEqual(self.part.read_bytes(), original)
        self.assertFalse(self.destination.exists())

    def test_wrong_range_total_is_rejected(self):
        self.server.mode = 'wrong_total'
        self.part.write_bytes(PAYLOAD[:12345])
        with self.assertRaisesRegex(RuntimeError, 'total differs'):
            self.run_download()
        self.assertFalse(self.destination.exists())

    def test_changed_source_validator_is_rejected(self):
        self.server.mode = 'changed_validator'
        self.part.write_bytes(PAYLOAD[:12345])
        self.checkpoint()
        with self.assertRaisesRegex(RuntimeError, 'validator changed'):
            self.run_download()
        self.assertEqual(self.part.read_bytes(), PAYLOAD[:12345])

    def test_checksum_mismatch_is_never_published(self):
        with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
            assets.download(self.url, self.destination, '0' * 64, expected_size=len(PAYLOAD))
        self.assertFalse(self.destination.exists())
        self.assertTrue(self.part.exists())
        self.assertFalse((self.root / 'archive.bin.download.json').exists())

    def test_complete_partial_is_verified_without_another_http_request(self):
        self.part.write_bytes(PAYLOAD)
        self.run_download()
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)
        self.assertEqual(self.server.calls, [])

    def test_partial_checkpoint_cannot_be_reused_for_another_source(self):
        original = PAYLOAD[:12345]
        self.part.write_bytes(original)
        self.checkpoint(source_url=self.url + '?other=1')
        with self.assertRaisesRegex(RuntimeError, 'another source'):
            self.run_download()
        self.assertEqual(self.part.read_bytes(), original)
        self.assertEqual(self.server.calls, [])

    def test_unmarked_partial_without_known_checksum_is_not_overwritten(self):
        original = PAYLOAD[:12345]
        self.part.write_bytes(original)
        with self.assertRaisesRegex(RuntimeError, 'existing partial'):
            assets.download(self.url, self.destination)
        self.assertEqual(self.part.read_bytes(), original)
        self.assertEqual(self.server.calls, [])

    def test_resume_disk_budget_counts_only_the_remaining_bytes(self):
        offset = len(PAYLOAD) // 2
        self.part.write_bytes(PAYLOAD[:offset])
        free = 2_000_000_000 + len(PAYLOAD) - offset + 64
        with patch.object(assets.shutil, 'disk_usage', return_value=SimpleNamespace(free=free)):
            self.run_download()
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)

    def test_ignored_range_checks_restart_space_before_truncating(self):
        self.server.mode = 'ignore_range'
        offset = len(PAYLOAD) // 2
        with self.part.open('wb') as stream:
            stream.truncate(offset)
        if getattr(self.part.stat(), 'st_blocks', 0) != 0:
            self.skipTest('Filesystem does not preserve a sparse truncated file')
        free = 2_000_000_000 + len(PAYLOAD) - offset + 64
        with patch.object(assets.shutil, 'disk_usage', return_value=SimpleNamespace(free=free)):
            with self.assertRaises(OSError):
                self.run_download()
        self.assertEqual(self.part.stat().st_size, offset)
        self.assertFalse(self.destination.exists())

    def test_capped_valid_ranges_continue_until_the_complete_source(self):
        self.server.mode = 'capped'
        self.part.write_bytes(PAYLOAD[:12345])
        self.run_download()
        offsets = [int(call['Range'].removeprefix('bytes=').removesuffix('-')) for call in self.server.calls]
        self.assertEqual(offsets, [12345, 1012345, 2012345, 3012345])
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)

    def test_html_and_unauthorized_responses_keep_existing_partial(self):
        original = PAYLOAD[:12345]
        for mode in ['html', 'unauthorized']:
            with self.subTest(mode=mode):
                self.server.mode = mode
                self.part.write_bytes(original)
                with self.assertRaises((RuntimeError, requests.HTTPError)):
                    self.run_download()
                self.assertEqual(self.part.read_bytes(), original)
                self.assertFalse(self.destination.exists())


if __name__ == '__main__':
    unittest.main()
