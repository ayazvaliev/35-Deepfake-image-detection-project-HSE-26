from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path, PurePosixPath

import pandas as pd

from src.preprocessing.splits import proportional_sample


ROOT = Path(__file__).resolve().parents[2]
RAW = Path(os.environ.get('DEEPFAKE_DATA_ROOT', ROOT / 'data' / 'raw')).expanduser()
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def stable_id(dataset, value):
    return dataset + ':' + hashlib.sha256(str(value).encode()).hexdigest()[:24]


def sample_manifest(frame, limit, strata=None, seed=42):
    return proportional_sample(frame, min(limit, len(frame)), strata=strata, seed=seed)


def index_df40(root=None):
    from src.preprocessing.df40 import FORMATS, METHODS

    root = Path(root or RAW / 'df40')
    sample = root / 'sample.parquet'
    if not sample.is_file():
        raise FileNotFoundError(f'Acquired DF40 sample missing: {sample}; run scripts/download_df40.py first')
    frame = pd.read_parquet(sample)
    required = {'dataset', 'sample_id', 'path', 'target', 'native_method'}
    missing = required - set(frame)
    if missing:
        raise ValueError(f'DF40 acquired image columns missing: {sorted(missing)}')
    if frame.empty:
        raise ValueError('DF40 acquired image sample is empty')
    if not frame['dataset'].astype(str).str.casefold().eq('df40').all():
        raise ValueError('DF40 acquired sample contains another dataset')
    if not frame['target'].isin([0, 1]).all():
        raise ValueError('DF40 acquired sample requires binary targets 0/1')
    identifiers = frame['sample_id'].map(lambda value: isinstance(value, str) and bool(value.strip()))
    if not identifiers.all() or frame['sample_id'].duplicated().any():
        raise ValueError('DF40 image sample_id values must be nonempty and unique')
    paths = frame['path'].map(
        lambda value: isinstance(value, str) and bool(value.strip())
        and Path(value).suffix.lower() in FORMATS
    )
    if not paths.all():
        raise ValueError('DF40 acquired sample must contain native image paths, not videos')
    if 'face_index' in frame and frame['face_index'].notna().any():
        raise ValueError('DF40 indexing requires native image rows, not processed face rows')
    families = {method.casefold(): specification[0] for method, specification in METHODS.items()}
    family = frame['native_method'].astype(str).str.casefold().map(families)
    if family.isna().any():
        raise ValueError('DF40 sample contains a method outside native EFS/FE')
    if 'native_family' in frame and not frame['native_family'].astype(str).str.upper().eq(family).all():
        raise ValueError('DF40 native_family does not match its native method')
    if 'manipulation' in frame:
        expected = family.where(frame['target'].eq(1), 'REAL')
        if not frame['manipulation'].astype(str).str.upper().eq(expected).all():
            raise ValueError('DF40 manipulation must preserve EFS/FE for fake and real for genuine images')
    return frame



def index_140k(root=None):
    root = Path(root or RAW / '140k')
    records = []
    for path in sorted(root.rglob('*')):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(root)
        parts = [part.lower() for part in relative.parts]
        classes = [part for part in parts[:-1] if part in {'real', 'fake'}]
        splits = [part for part in parts[:-1] if part in {'train', 'valid', 'val', 'test'}]
        if len(classes) != 1 or len(splits) != 1:
            continue
        target = int(classes[0] == 'fake')
        records.append(dict(
            dataset='140k', sample_id=stable_id('140k', relative), path=str(path.resolve()),
            target=target, author_split=splits[0], generator='StyleGAN' if target else 'real',
            manipulation='EFS' if target else 'real', domain='StyleGAN' if target else 'FFHQ',
            source_group='140k:' + classes[0] + ':' + path.stem, identity_group='', image_label_all_faces=False,
        ))
    if not records:
        raise FileNotFoundError(f'No 140k class/split image folders under {root}')
    frame = pd.DataFrame(records)
    frame['author_split_inventory'] = [['train', 'validate', 'test'] for _ in records]
    return frame


def mffi_author_split(path):
    name = Path(path).name.lower()
    if name.startswith('train'):
        return 'train'
    if name.startswith(('val', 'valid')):
        return 'valid'
    if name.startswith('test'):
        return 'test'
    raise ValueError(f'Unknown MFFI label split: {path}')


def mffi_image_split(path):
    aliases = {'train': 'train', 'training': 'train', 'trainset': 'train',
               'val': 'valid', 'valid': 'valid', 'validation': 'valid',
               'valset': 'valid', 'validset': 'valid', 'validationset': 'valid',
               'test': 'test', 'testset': 'test', 'testd': 'test', 'test-d': 'test'}
    values = {aliases[part.lower()] for part in PurePosixPath(str(path).replace('\\', '/')).parts[:-1]
              if part.lower() in aliases}
    if len(values) > 1:
        raise ValueError(f'Ambiguous MFFI image split: {path}')
    return next(iter(values), None)


def mffi_filename(value):
    filename = PurePosixPath(str(value).strip().replace('\\', '/'))
    if filename.is_absolute() or '..' in filename.parts or not filename.name:
        raise ValueError(f'Unsafe MFFI filename: {value!r}')
    return filename.as_posix()


def read_mffi_labels(path):
    records = []
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            values = next(csv.reader([line])) if ',' in line else line.split()
            if len(values) < 2:
                raise ValueError(f'Invalid MFFI label record: {line[:100]}')
            if values[0].strip().lower() in {'filename', 'path', 'image', 'img_name', 'image_name'}:
                if values[1].strip().lower() in {'target', 'label', 'class'}:
                    continue
            try:
                target = int(values[1])
            except ValueError as error:
                raise ValueError('MFFI target must be explicitly 0 or 1') from error
            if target not in (0, 1):
                raise ValueError('MFFI binary labels expected; multiclass labels need a verified mapping')
            records.append({'filename': mffi_filename(values[0]), 'target': target,
                            'generator': 'unknown' if target else 'real',
                            'manipulation': 'unknown' if target else 'real',
                            'extra_labels': values[2:]})
    return pd.DataFrame(records, columns=['filename', 'target', 'generator', 'manipulation', 'extra_labels'])


def _mffi_image_path(filename, label_path, split, image_index):
    candidates = image_index.get(PurePosixPath(filename).name, [])
    scoped = [path for path in candidates if label_path.parent in path.parents]
    compatible = [path for path in scoped if mffi_image_split(path.relative_to(label_path.parent)) in (None, split)]
    explicit = label_path.parent / filename
    if explicit in compatible:
        return explicit
    if len(compatible) > 1:
        raise ValueError(f'Ambiguous MFFI filename in {split}: {filename}')
    return compatible[0] if compatible else None


def index_mffi(root=None, existing_only=True):
    root = Path(root or RAW / 'mffi')
    label_paths = sorted(root.rglob('*set_label.txt'))
    if not label_paths:
        raise FileNotFoundError(f'MFFI label files missing under {root}')
    image_index = defaultdict(list)
    for path in root.rglob('*'):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            image_index[path.name].append(path)
    present = {mffi_author_split(path) for path in label_paths}
    inventory = [canonical for source, canonical in [('train', 'train'), ('valid', 'validate'), ('test', 'test')] if source in present]
    records = []
    for label_path in label_paths:
        split = mffi_author_split(label_path)
        labels = read_mffi_labels(label_path)
        for row in labels.to_dict('records'):
            filename = row.pop('filename')
            path = _mffi_image_path(filename, label_path, split, image_index)
            if path is None:
                if existing_only:
                    continue
                path = label_path.parent / {'train': 'trainset', 'valid': 'valset', 'test': 'test'}[split] / filename
            relative = path.relative_to(root).as_posix()
            records.append(dict(
                **row, dataset='mffi', sample_id=stable_id('mffi', relative), path=str(path.resolve()),
                author_split=split, author_split_inventory=inventory, domain='unknown',
                source_group='mffi:' + PurePosixPath(filename).stem,
                identity_group='', image_label_all_faces=False,
                label_source=label_path.relative_to(root).as_posix(),
            ))
    if not records:
        raise FileNotFoundError('MFFI labels found but no matching images in their author split')
    frame = pd.DataFrame(records)
    duplicated = frame['sample_id'].duplicated(keep=False)
    if duplicated.any():
        conflict = frame.loc[duplicated].groupby('sample_id')['target'].nunique()
        if (conflict > 1).any():
            raise ValueError('Contradictory MFFI targets for the same image')
        frame = frame.drop_duplicates('sample_id').reset_index(drop=True)
    return frame


def index_openforensics(root=None, existing_only=True):
    root = Path(root or RAW / 'openforensics')
    records = []
    inventory = [canonical for split, canonical in [('Train', 'train'), ('Val', 'validate'), ('Test-Dev', 'test')]
                 if list(root.rglob(split + '_poly.json'))]
    image_index = defaultdict(list)
    for path in root.rglob('*'):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            image_index[path.name].append(path)
    for split in ('Train', 'Val', 'Test-Dev'):
        candidates = sorted(root.rglob(split + '_poly.json'))
        if not candidates:
            continue
        if len(candidates) != 1:
            raise ValueError(f'Ambiguous OpenForensics annotation files for {split}')
        data = json.loads(candidates[0].read_text())
        categories = {int(c['id']): str(c['name']).strip().lower() for c in data['categories']}
        label_map = {}
        for category_id, name in categories.items():
            if name in {'real', 'genuine', 'authentic'}:
                label_map[category_id] = 0
            elif name in {'fake', 'forged', 'forgery'}:
                label_map[category_id] = 1
            else:
                raise ValueError(f'Unknown OpenForensics category: {category_id}={name}')
        annotations = defaultdict(list)
        for annotation in data['annotations']:
            annotations[annotation['image_id']].append({
                'bbox': annotation['bbox'], 'target': label_map[int(annotation['category_id'])],
                'segmentation': annotation.get('segmentation'), 'annotation_id': annotation['id'],
            })
        for image in data['images']:
            name = Path(image['file_name']).name
            matches = [path for path in image_index.get(name, [])
                       if split.lower() in {part.lower() for part in path.relative_to(root).parts[:-1]}]
            if len(matches) > 1:
                raise ValueError(f'Ambiguous OpenForensics image path: {split}/{name}')
            path = matches[0] if matches else None
            if path is None:
                if existing_only:
                    continue
                path = root / 'images' / split / name
            faces = annotations.get(image['id'], [])
            if not faces:
                continue
            labels = [a['target'] for a in faces]
            records.append(dict(
                dataset='openforensics', sample_id=stable_id('openforensics', f'{split}/{image["id"]}'),
                path=str(path.resolve()), target=labels[0] if len(set(labels)) == 1 else None,
                author_split=split.lower(), author_split_inventory=inventory, generator='unknown', manipulation='unknown', domain='in_the_wild',
                source_group=f'openforensics:{split}:{image["id"]}', identity_group='',
                annotations=faces, n_faces=len(faces), frame_composition='mixed' if len(set(labels)) > 1 else 'all_fake' if labels[0] else 'all_real',
                image_label_all_faces=False, width=image['width'], height=image['height'],
            ))
    if not records:
        raise FileNotFoundError(f'Original OpenForensics annotations/images missing under {root}')
    return pd.DataFrame(records)


def write_jsonl(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for row in frame.to_dict('records'):
            row = {k: v for k, v in row.items() if not isinstance(v, float) or pd.notna(v)}
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
