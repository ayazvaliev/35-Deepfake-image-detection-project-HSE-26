from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.datasets import (RAW, ROOT, IMAGE_SUFFIXES, index_openforensics,
                                        mffi_author_split, mffi_image_split, read_mffi_labels, sample_manifest)


MODELS = {
    'face_detection_yunet_2023mar.onnx': (
        'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
        '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'),
    'face_landmarker.task': (
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'),
    'resnet18-f37072fd.pth': ('https://download.pytorch.org/models/resnet18-f37072fd.pth',
        'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'),
}
ZENODO = 'https://zenodo.org/records/5528418/files/'
MODELSCOPE = 'https://www.modelscope.cn/api/v1/datasets/DDLteam/MFFI/repo?Revision=fd9fce422da6ff231906dc06e984674f2332919a&FilePath='


def file_hash(path, algorithm='sha256'):
    digest = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _require_download_space(directory, remaining, reclaim=0):
    if remaining is not None and shutil.disk_usage(directory).free + reclaim < remaining + 2_000_000_000:
        raise OSError(f'Not enough disk space under {directory}')


def _download_length(response):
    value = response.headers.get('Content-Length')
    if value is None:
        return None
    try:
        length = int(value)
    except ValueError as error:
        raise RuntimeError('Invalid download Content-Length') from error
    if length < 0:
        raise RuntimeError('Negative download Content-Length')
    return length


def _write_download_json(path, metadata):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(metadata, indent=2))
    temporary.replace(path)


def _finish_download(part, destination, marker, part_marker, url, expected, algorithm, expected_size):
    size = part.stat().st_size
    if expected_size is not None and size != expected_size:
        raise RuntimeError(f'Download size mismatch: {destination}')
    expected_hash = file_hash(part, algorithm) if expected else None
    if expected and expected_hash.lower() != expected.lower():
        raise RuntimeError(f'Download checksum mismatch: {destination}')
    sha = expected_hash if expected and algorithm.lower() == 'sha256' else file_hash(part)
    _write_download_json(marker, {'source_url': url, 'sha256': sha, 'bytes': size})
    part.replace(destination)
    part_marker.unlink(missing_ok=True)
    return destination


def _source_validator(state):
    etag = state.get('etag')
    if etag and not etag.startswith('W/'):
        return 'ETag', etag
    modified = state.get('last_modified')
    return ('Last-Modified', modified) if modified else (None, None)


def download(url, destination, expected=None, algorithm='sha256', expected_size=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    marker = destination.with_name(destination.name + '.download.json')
    if destination.exists():
        valid = expected_size is None or destination.stat().st_size == expected_size
        if expected:
            valid = valid and file_hash(destination, algorithm).lower() == expected.lower()
        elif marker.exists():
            saved = json.loads(marker.read_text())
            valid = valid and saved.get('source_url') == url and saved.get('sha256') == file_hash(destination)
        else:
            valid = False
        if valid:
            return destination
        raise RuntimeError(f'Existing unverified download: {destination}')
    part = destination.with_name(destination.name + '.part')
    part_marker = destination.with_name(destination.name + '.part.download.json')
    if part.is_symlink() or part_marker.is_symlink():
        raise RuntimeError('Download staging paths must not be symlinks')
    offset = part.stat().st_size if part.exists() else 0
    state = {'source_url': url, 'expected_checksum': expected,
             'algorithm': algorithm, 'expected_size': expected_size}
    if offset and part_marker.exists():
        saved = json.loads(part_marker.read_text())
        if any(saved.get(key) != value for key, value in state.items()):
            raise RuntimeError(f'Partial download belongs to another source: {part}')
        state.update(saved)
    elif offset and not expected:
        raise RuntimeError(f'Cannot verify the source of an existing partial download: {part}')
    hint = expected_size if expected_size is not None else state.get('response_size')
    if hint is not None and offset > hint:
        raise RuntimeError(f'Partial download is larger than the expected source: {part}')
    if offset and expected and hint == offset:
        return _finish_download(part, destination, marker, part_marker, url, expected, algorithm, hint)
    if offset and marker.exists():
        saved = json.loads(marker.read_text())
        if saved.get('source_url') == url and saved.get('bytes') == offset and saved.get('sha256') == file_hash(part):
            return _finish_download(part, destination, marker, part_marker, url, expected, algorithm, hint)
    if offset and not expected and not _source_validator(state)[1]:
        raise RuntimeError('Resuming without a checksum requires an ETag or Last-Modified validator')
    _require_download_space(destination.parent, None if hint is None else hint - offset)
    while True:
        headers = {'Accept-Encoding': 'identity'}
        validator_name, validator = _source_validator(state)
        if offset:
            headers['Range'] = f'bytes={offset}-'
            if validator:
                headers['If-Range'] = validator
        with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            if 'text/html' in response.headers.get('Content-Type', '').lower():
                raise RuntimeError(f'Download returned HTML: {url}')
            if response.headers.get('Content-Encoding', 'identity').lower() not in {'', 'identity'}:
                raise RuntimeError('Download content encoding prevents byte-exact resume')
            length = _download_length(response)
            if response.status_code == 206:
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''), re.IGNORECASE)
                if match is None:
                    raise RuntimeError('Missing or invalid download Content-Range')
                start, end, total = map(int, match.groups())
                if start != offset or end < start or end >= total:
                    raise RuntimeError('Download Content-Range does not match the partial file')
                segment_length = end - start + 1
                if length is not None and length != segment_length:
                    raise RuntimeError('Download Content-Length disagrees with Content-Range')
                if hint is not None and total != hint:
                    raise RuntimeError('Download Content-Range total differs from the expected source size')
                if offset and validator and response.headers.get(validator_name) != validator:
                    raise RuntimeError('Download source validator changed during resume')
                _require_download_space(destination.parent, total - offset)
                mode = 'ab' if offset else 'wb'
            elif response.status_code == 200:
                if expected_size is not None and length is not None and length != expected_size:
                    raise RuntimeError('Download Content-Length differs from the expected source size')
                total = expected_size if expected_size is not None else length
                reclaim = getattr(part.stat(), 'st_blocks', 0) * 512 if offset else 0
                _require_download_space(destination.parent, total, reclaim)
                offset, mode, segment_length = 0, 'wb', length
            else:
                raise RuntimeError(f'Unexpected download HTTP status: {response.status_code}')
            state.update(response_size=total, etag=response.headers.get('ETag'),
                         last_modified=response.headers.get('Last-Modified'))
            _write_download_json(part_marker, state)
            received = 0
            with part.open(mode) as stream, tqdm(total=total, initial=offset, unit='B', unit_scale=True,
                                                desc=destination.name) as progress:
                for chunk in response.iter_content(1024 * 1024):
                    if not chunk:
                        continue
                    if segment_length is not None and received + len(chunk) > segment_length:
                        raise RuntimeError('Download body exceeds its declared range')
                    if total is not None and offset + received + len(chunk) > total:
                        raise RuntimeError('Download body exceeds the expected source size')
                    if total is None:
                        _require_download_space(destination.parent, len(chunk))
                    stream.write(chunk)
                    received += len(chunk)
                    progress.update(len(chunk))
            if segment_length is not None and received != segment_length:
                raise RuntimeError('Download body is shorter than its declared range')
            offset = part.stat().st_size
            if total is None or offset == total:
                break
            if response.status_code != 206 or offset > total:
                raise RuntimeError(f'Download size mismatch: {destination}')
            hint = total
    return _finish_download(part, destination, marker, part_marker, url, expected, algorithm, total)


def safe_target(root, name):
    root = Path(root).resolve()
    target = (root / name).resolve()
    if target != root and root not in target.parents:
        raise ValueError('Unsafe archive member path')
    return target


def extract_zip(archive, destination, wanted_names=None):
    files = []
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            if member.is_dir():
                continue
            if wanted_names is not None and Path(member.filename).name not in wanted_names:
                continue
            target = safe_target(destination, member.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with source.open(member) as incoming, target.open('wb') as outgoing:
                for block in iter(lambda: incoming.read(1024 * 1024), b''):
                    digest.update(block)
                    outgoing.write(block)
            files.append({'path': target.relative_to(Path(destination).resolve()).as_posix(),
                          'sha256': digest.hexdigest(), 'bytes': target.stat().st_size})
    return files


def _kaggle_files_complete(root, files):
    if not isinstance(files, list) or not files:
        return False
    checked = []
    valid_metadata = True
    seen = set()
    for row in files:
        if not isinstance(row, dict) or not isinstance(row.get('path'), str):
            valid_metadata = False
            continue
        target = safe_target(root, row['path'])
        if target == root.resolve() or Path(row['path']).is_absolute():
            raise ValueError('Unsafe dataset file path')
        if target in seen:
            valid_metadata = False
        seen.add(target)
        size, checksum = row.get('bytes'), row.get('sha256')
        valid_size = isinstance(size, int) and not isinstance(size, bool) and size >= 0
        valid_hash = isinstance(checksum, str) and re.fullmatch(r'[0-9a-fA-F]{64}', checksum) is not None
        valid_metadata = valid_metadata and valid_size and valid_hash
        checked.append((target, size, checksum))
    if not valid_metadata:
        return False
    return all(
        path.is_file() and path.stat().st_size == size and file_hash(path).lower() == checksum.lower()
        for path, size, checksum in checked
    )


def _kaggle_marker_complete(root, marker, slug):
    if marker.is_symlink() or marker.with_name(marker.name + '.tmp').is_symlink():
        raise ValueError('Unsafe dataset completion marker')
    if not marker.is_file():
        return False
    try:
        saved = json.loads(marker.read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(saved, dict):
        return False
    if saved.get('source') != slug:
        raise ValueError('Dataset completion marker belongs to a different dataset source')
    return _kaggle_files_complete(root, saved.get('files'))


def kaggle_dataset(slug, name):
    root = safe_target(RAW, name)
    done = root / '.download_complete.json'
    if _kaggle_marker_complete(root, done, slug):
        print(name, 'already downloaded')
        return
    archive = download('https://www.kaggle.com/api/v1/datasets/download/' + slug, RAW / '.archives' / (name + '.zip'))
    files = extract_zip(archive, root)
    if not _kaggle_files_complete(root, files):
        raise RuntimeError('Extracted Kaggle files failed completion verification')
    if done.is_symlink() or done.with_name(done.name + '.tmp').is_symlink():
        raise ValueError('Unsafe dataset completion marker')
    _write_download_json(done, {'source': slug, 'archive_sha256': file_hash(archive), 'files': files})
    archive.unlink()


def openforensics(limit, seed):
    root = RAW / 'openforensics'
    annotation_root = root / 'annotations'
    record = requests.get('https://zenodo.org/api/records/5528418', timeout=60)
    record.raise_for_status()
    files = {entry['key']: entry for entry in record.json()['files']}
    for split in ('Train', 'Val', 'Test-Dev'):
        name = split + '_poly.json'
        entry = files[name]
        algorithm, expected = entry['checksum'].split(':')
        download(ZENODO + name + '?download=1', annotation_root / name, expected, algorithm, entry['size'])
    inventory = index_openforensics(root, existing_only=False)
    selected = sample_manifest(inventory, limit, ['author_split', 'frame_composition', 'n_faces'], seed)
    needed = set(Path(path).name for path in selected['path'])
    selection_sha = hashlib.sha256('\n'.join(sorted(selected['sample_id'])).encode()).hexdigest()
    for name, entry in files.items():
        if not name.endswith('.zip') or name.startswith('Test-Challenge'):
            continue
        algorithm, expected = entry['checksum'].split(':')
        marker = root / (name + '.extracted.json')
        if marker.exists():
            saved = json.loads(marker.read_text())
            matches = saved.get('seed') == seed and saved.get('limit') == limit and saved.get('selection_sha256') == selection_sha
            extracted = saved.get('files', [])
            complete = bool(extracted) and all(safe_target(root / 'images', row['path']).is_file() for row in extracted)
            if matches and complete:
                continue
        archive = download(ZENODO + name + '?download=1', RAW / '.archives' / name, expected, algorithm, entry['size'])
        extracted = extract_zip(archive, root / 'images', needed)
        marker.write_text(json.dumps({'sha256': file_hash(archive), 'source_checksum': entry['checksum'],
                                      'seed': seed, 'limit': limit, 'selection_sha256': selection_sha,
                                      'files': extracted}, indent=2))
        archive.unlink()
    available = index_openforensics(root)
    if not set(selected['sample_id']) <= set(available['sample_id']):
        raise RuntimeError('Not all selected OpenForensics images were extracted')
    (root / 'sampling.json').write_text(json.dumps(selected.attrs['sampling'], indent=2))


def selection_ready(marker, limit, seed, root):
    if not marker.exists():
        return False
    metadata = json.loads(marker.read_text())
    if metadata.get('requested_limit') != limit or metadata.get('seed') != seed:
        raise RuntimeError(f'Existing selection uses another seed or limit: {marker}; choose a fresh data root')
    files = metadata.get('files', [])
    return bool(files) and all(safe_target(root, item['path']).is_file() for item in files)


def extract_mffi_selection(archive, destination, selected, default_split=None):
    destination = Path(destination).resolve()
    wanted = {(row['author_split'], Path(row['filename']).name)
              for row in selected.to_dict('records')}
    found = {}
    with tarfile.open(archive, 'r:gz') as source:
        for member in source:
            if not member.isfile() or Path(member.name).suffix.lower() not in IMAGE_SUFFIXES:
                continue
            split = mffi_image_split(member.name) or default_split
            key = split, Path(member.name).name
            if key not in wanted:
                continue
            if key in found:
                raise ValueError(f'Duplicate archive image key: {key}')
            target = safe_target(destination, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + '.extract.part')
            digest = hashlib.sha256()
            with source.extractfile(member) as incoming, part.open('wb') as outgoing:
                for block in iter(lambda: incoming.read(1024 * 1024), b''):
                    digest.update(block)
                    outgoing.write(block)
            sha = digest.hexdigest()
            if target.exists():
                if file_hash(target) != sha:
                    raise RuntimeError(f'Existing image differs from verified archive: {target}')
                part.unlink()
            else:
                part.replace(target)
            found[key] = {'path': target.relative_to(destination).as_posix(), 'sha256': sha, 'bytes': target.stat().st_size}
    if set(found) != wanted:
        missing = wanted - set(found)
        raise RuntimeError(f'MFFI selected images missing: {len(missing)}')
    return list(found.values())


def mffi(limit, seed, include_test=True, test_only=False):
    root = RAW / 'mffi'
    phase1 = root / 'phase1'
    done = phase1 / '.selection_complete.json'
    if not test_only and not selection_ready(done, limit, seed, root):
        source_url = MODELSCOPE + 'phase1.tar.gz'
        archive = download(source_url, RAW / '.archives' / 'mffi_phase1.tar.gz',
                           '9a5f9dd847d163a84e456872f6b608b53d957330148003d06f7a3386687f104b',
                           expected_size=24_509_568_862)
        labels = []
        with tarfile.open(archive, 'r:gz') as source:
            for member in source:
                if member.isfile() and member.name.endswith('set_label.txt'):
                    target = safe_target(root, member.name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as incoming, target.open('wb') as outgoing:
                        shutil.copyfileobj(incoming, outgoing)
                    frame = read_mffi_labels(target)
                    frame['author_split'] = mffi_author_split(target)
                    labels.append(frame)
        if not labels:
            raise RuntimeError('MFFI phase1 label files missing')
        inventory = pd.concat(labels, ignore_index=True)
        selected = sample_manifest(inventory, limit, ['author_split', 'target'], seed)
        files = extract_mffi_selection(archive, root, selected)
        metadata = {**selected.attrs['sampling'], 'requested_limit': limit,
                    'source_url': source_url, 'archive_sha256': file_hash(archive), 'files': files}
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text(json.dumps(metadata, indent=2))
        archive.unlink()
    if not include_test:
        return
    phase2 = root / 'phase2'
    labels_path = download(MODELSCOPE + 'phase2/testset_label.txt', phase2 / 'testset_label.txt',
                           '24f9be8e2942d066454a9d135465157dbf79e036b40bb4b3d195e40352285288', expected_size=7_095_932)
    marker = phase2 / '.selection_complete.json'
    if selection_ready(marker, limit, seed, phase2):
        return
    inventory = read_mffi_labels(labels_path)
    inventory['author_split'] = 'test'
    selected = sample_manifest(inventory, limit, ['target'], seed)
    source_url = MODELSCOPE + 'phase2/test.tar.gz'
    archive = download(source_url, RAW / '.archives' / 'mffi_test.tar.gz',
                       '27f709ffa91755cbeb5258ec439d1722efdfa49b0e5c460c85429f139c446eb6', expected_size=7_133_438_131)
    files = extract_mffi_selection(archive, phase2, selected, default_split='test')
    marker.write_text(json.dumps({**selected.attrs['sampling'], 'requested_limit': limit,
                                 'source_url': source_url, 'archive_sha256': file_hash(archive), 'files': files}, indent=2))
    archive.unlink()


def models():
    provenance = {}
    for name, (url, checksum) in MODELS.items():
        path = download(url, ROOT / 'models' / name, checksum)
        provenance[name] = {'source': url, 'sha256': file_hash(path), 'bytes': path.stat().st_size}
    (ROOT / 'models' / 'provenance.json').write_text(json.dumps(provenance, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['models', 'datasets'])
    parser.add_argument('--datasets', nargs='+', default=['140k', 'openforensics', 'mffi'],
                        choices=['140k', 'openforensics', 'openforensics_crops', 'mffi'])
    parser.add_argument('--limit', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-test', action='store_true')
    parser.add_argument('--mffi-test-only', action='store_true')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    if args.mffi_test_only and args.skip_test:
        parser.error('--mffi-test-only cannot be combined with --skip-test')
    if args.kind == 'models':
        models()
        return
    for dataset in args.datasets:
        if dataset == '140k':
            kaggle_dataset('xhlulu/140k-real-and-fake-faces', '140k')
        elif dataset == 'openforensics_crops':
            kaggle_dataset('manjilkarki/deepfake-and-real-images', 'openforensics_crops')
        elif dataset == 'openforensics':
            openforensics(args.limit, args.seed)
        elif dataset == 'mffi':
            mffi(args.limit, args.seed, not args.skip_test, test_only=args.mffi_test_only)


if __name__ == '__main__':
    main()
