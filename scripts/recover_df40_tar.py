from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import time
import uuid
import zlib

import pandas as pd
import requests


REPOSITORY = 'sinister007/df40_dataset'
REVISION = 'a203e9b880d69339178f1a6aacd937b07a8ef199'
ARCHIVE = 'dataset_180gb.tar.gz'
ARCHIVE_BYTES = 182_911_998_168
ARCHIVE_SHA256 = 'f821d75714c887004f76ea5df2852fdb1336be3879f7b1580a21d3cce93e4c67'
SOURCE_URL = f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{ARCHIVE}'
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def file_digest(path):
    sha, crc, count = hashlib.sha256(), 0, 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
            crc = zlib.crc32(block, crc)
            count += len(block)
    return sha.hexdigest(), f'{crc:08x}', count


def method_key(value):
    return re.sub(r'[^a-z0-9]', '', str(value).lower())


def native_tail(path, method):
    parts = [part for part in PurePosixPath(path).parts if part not in {'.', 'frames'}]
    matching = [index for index, part in enumerate(parts) if method_key(part) == method]
    if not matching:
        return None
    parts = parts[matching[-1] + 1:]
    if parts and parts[0].lower() == 'ff':
        parts = parts[1:]
    return '/'.join(parts).lower()


def destination_for(root, row):
    relative = PurePosixPath(row['archive_member'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe primary archive member')
    return root / 'images' / row['archive_partition'] / relative


class ProbeLimitReached(RuntimeError):
    pass


class CountedSource:
    def __init__(self, raw, limit, on_progress):
        self.raw, self.limit, self.on_progress = raw, limit, on_progress
        self.bytes_read = 0
        self.sha256 = hashlib.sha256()

    def read(self, size=-1):
        remaining = self.limit - self.bytes_read
        if remaining <= 0:
            raise ProbeLimitReached()
        count = min(remaining, size if size >= 0 else 1024 * 1024)
        block = self.raw.read(count)
        if not block:
            return block
        self.bytes_read += len(block)
        self.sha256.update(block)
        self.on_progress(self.bytes_read)
        return block


def recover(root, probe_bytes=8_000_000_000):
    root = Path(root).resolve()
    population = pd.read_parquet(root / 'inventory.parquet', columns=[
        'archive_member', 'archive_partition', 'native_method', 'zip_crc32', 'uncompressed_bytes'])
    selected = pd.read_parquet(root / 'selection.parquet')
    selected = selected[selected['native_method'].isin(['e4e', 'pixart']) & (selected['archive_partition'] == 'test')]
    records = selected.to_dict('records')
    if not records or selected['sample_id'].duplicated().any():
        raise ValueError('Expected unique selected e4e/pixart test records')
    minimum_space = sum(int(row['uncompressed_bytes']) for row in records
                        if not destination_for(root, row).exists()) + 1024 ** 3
    if shutil.disk_usage(root).free < minimum_space:
        raise OSError('Not enough disk space for only the selected missing images')
    known = defaultdict(list)
    methods = set()
    for row in population.to_dict('records'):
        method = method_key(row['native_method'])
        row['tail'] = native_tail(row['archive_member'], method)
        key = method, int(row['zip_crc32'], 16), int(row['uncompressed_bytes'])
        known[key].append(row)
        methods.add(method)
    wanted = defaultdict(list)
    for row in records:
        key = method_key(row['native_method']), int(row['zip_crc32'], 16), int(row['uncompressed_bytes'])
        wanted[key].append(row)
    checkpoint_path = root / 'metadata' / 'stream_downloads.json'
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
    source = {'kind': 'hf_stream_crc_verified', 'url': SOURCE_URL,
              'repository': REPOSITORY, 'revision': REVISION,
              'lfs_sha256': ARCHIVE_SHA256, 'bytes': ARCHIVE_BYTES,
              'full_sha256_verified': False}
    recovered = set()
    for row in records:
        destination = destination_for(root, row)
        entry = checkpoint.setdefault(row['archive_id'], {'download_source': source.copy(), 'files': {}})
        if destination.is_file() and destination.stat().st_size == int(row['uncompressed_bytes']):
            sha, crc, size = file_digest(destination)
            saved = entry['files'].get(row['sample_id'])
            if crc == row['zip_crc32'] and saved and saved.get('sha256') == sha:
                recovered.add(row['sample_id'])
    progress_path = root / 'metadata' / 'stream_progress.json'
    progress = {'status': 'starting', 'compressed_bytes': 0, 'source_bytes': ARCHIVE_BYTES,
                'selected_images': len(records), 'recovered_images': len(recovered),
                'members_seen': 0, 'first_native_verified': None}
    started = time.monotonic()
    last_print, last_checkpoint = 0, 0

    def save():
        write_json(checkpoint_path, checkpoint)
        write_json(progress_path, progress)

    def on_progress(count):
        nonlocal last_print, last_checkpoint
        progress.update(compressed_bytes=count, elapsed_seconds=round(time.monotonic() - started, 1))
        if count - last_print >= 256_000_000:
            last_print = count
            write_json(progress_path, progress)
            print(json.dumps({'event': 'stream_progress', **progress}), flush=True)
        if count - last_checkpoint >= 16_000_000_000:
            last_checkpoint = count
            save()

    if len(recovered) == len(records):
        progress['status'] = 'already_complete'
        save()
        print(json.dumps(progress), flush=True)
        return 0
    save()
    try:
        with requests.get(SOURCE_URL + '?download=1&probe=' + uuid.uuid4().hex,
                          headers={'Accept-Encoding': 'identity'}, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            if response.status_code != 200 or 'text/html' in response.headers.get('Content-Type', '').lower():
                raise ValueError('Source is not a public full archive stream')
            if response.headers.get('Content-Encoding', 'identity') not in {'identity', ''}:
                raise ValueError('Unexpected source content encoding')
            if int(response.headers.get('Content-Length', ARCHIVE_BYTES)) != ARCHIVE_BYTES:
                raise ValueError('Published source size mismatch')
            reader = CountedSource(response.raw, min(probe_bytes, ARCHIVE_BYTES), on_progress)
            with tarfile.open(fileobj=reader, mode='r|gz', bufsize=1024 * 1024) as archive:
                while True:
                    member = archive.next()
                    if member is None:
                        break
                    progress['members_seen'] += 1
                    progress['last_tar_member'] = member.name
                    archive.members.clear()
                    if not member.isfile() or PurePosixPath(member.name).suffix.lower() not in IMAGE_SUFFIXES:
                        continue
                    found_methods = {method_key(part) for part in PurePosixPath(member.name).parts[:-1]} & methods
                    if len(found_methods) != 1 or member.size > 50 * 1024 * 1024:
                        continue
                    method = next(iter(found_methods))
                    if progress['first_native_verified'] is not None and method not in {'pixart', 'e4e'}:
                        continue
                    content = archive.extractfile(member).read()
                    if len(content) != member.size:
                        raise ValueError('Incomplete source tar member')
                    crc = zlib.crc32(content)
                    key = method, crc, len(content)
                    candidates = known.get(key, [])
                    tail = native_tail(member.name, method)
                    matching = [row for row in candidates if row['tail'] == tail]
                    if not matching and len(candidates) == 1:
                        matching = candidates
                    if not matching:
                        continue
                    if progress['first_native_verified'] is None:
                        proof = {'method': method, 'source_tar_member': member.name,
                                 'primary_archive_member': matching[0]['archive_member'],
                                 'zip_crc32': f'{crc:08x}', 'bytes': len(content),
                                 'sha256': hashlib.sha256(content).hexdigest(),
                                 'compressed_bytes': reader.bytes_read,
                                 'matched_by': 'path_crc_size' if matching[0]['tail'] == tail else 'unique_crc_size'}
                        progress['first_native_verified'] = proof
                        reader.limit = ARCHIVE_BYTES
                        progress['status'] = 'recovering_selected'
                        save()
                        print(json.dumps({'event': 'first_native_verified', **proof}), flush=True)
                    eligible = wanted.get(key, [])
                    path_matches = [row for row in eligible if native_tail(row['archive_member'], method) == tail]
                    if path_matches:
                        eligible = path_matches
                    elif len(candidates) != 1:
                        continue
                    for row in eligible:
                        if row['sample_id'] in recovered:
                            continue
                        destination = destination_for(root, row)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        sha = hashlib.sha256(content).hexdigest()
                        if destination.exists():
                            actual_sha, actual_crc, actual_size = file_digest(destination)
                            if (actual_sha, actual_crc, actual_size) != (sha, row['zip_crc32'], len(content)):
                                raise ValueError('Existing selected image differs from the verified source')
                        else:
                            if shutil.disk_usage(root).free < len(content) + 512 * 1024 ** 2:
                                raise OSError('Insufficient disk space for a selected native image')
                            temporary = destination.with_name(destination.name + '.stream.part')
                            temporary.write_bytes(content)
                            temporary.replace(destination)
                        checkpoint[row['archive_id']]['files'][row['sample_id']] = {
                            'sample_id': row['sample_id'], 'relative_path': destination.relative_to(root).as_posix(),
                            'sha256': sha, 'bytes': len(content), 'zip_crc32': row['zip_crc32'],
                            'source_tar_member': member.name}
                        recovered.add(row['sample_id'])
                        progress['recovered_images'] = len(recovered)
                        if len(recovered) % 100 == 0:
                            save()
                            print(json.dumps({'event': 'recovered_images', **progress}), flush=True)
                    if len(recovered) == len(records):
                        break
            if reader.bytes_read == ARCHIVE_BYTES:
                if reader.sha256.hexdigest() != ARCHIVE_SHA256:
                    raise ValueError('Full source archive SHA256 mismatch')
                for entry in checkpoint.values():
                    entry['download_source']['full_sha256_verified'] = True
            progress['status'] = 'complete' if len(recovered) == len(records) else 'source_exhausted_incomplete'
            save()
            print(json.dumps({'event': 'stream_finished', **progress}), flush=True)
            return 0 if progress['status'] == 'complete' else 3
    except ProbeLimitReached:
        progress['status'] = 'probe_limit_reached'
        save()
        print(json.dumps({'event': 'probe_limit_reached', **progress}), flush=True)
        return 2
    except Exception as error:
        progress.update(status='failed', error_type=type(error).__name__)
        response = getattr(error, 'response', None)
        if response is not None:
            progress['http_status'] = response.status_code
        save()
        print(json.dumps({'event': 'stream_failed', **progress}), flush=True)
        return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--probe-bytes', type=int, default=8_000_000_000)
    args = parser.parse_args()
    if args.probe_bytes <= 0 or args.probe_bytes > ARCHIVE_BYTES:
        parser.error('probe size must be positive and no larger than the source archive')
    return recover(args.root, args.probe_bytes)


if __name__ == '__main__':
    raise SystemExit(main())
