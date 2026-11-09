from __future__ import annotations

import ast
from collections import Counter
import json
from pathlib import Path
import re
import unicodedata

import pandas as pd


REGISTRY_PATH = Path(__file__).resolve().parents[2] / 'configs' / 'label_aliases.json'


def _missing(value):
    if value is None or value is pd.NA:
        return True
    if pd.api.types.is_scalar(value):
        return bool(pd.isna(value))
    return False


def _text(value):
    if _missing(value):
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, set):
        value = sorted(value, key=str)
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def _alias_key(value):
    text = unicodedata.normalize('NFKC', _text(value)).strip().casefold()
    text = re.sub(r'(?<=\d)[._\-\s]+(?=\d)', '\x00', text)
    key = ''.join(character for character in text if character.isalnum() or character == '\x00')
    return key.replace('\x00', '.')


def _token(value):
    text = unicodedata.normalize('NFKC', _text(value)).strip().casefold()
    text = re.sub(r'[^\w]+', '_', text, flags=re.UNICODE)
    return re.sub(r'_+', '_', text).strip('_') or 'unknown'


def _compile(groups, include_canonical=True):
    index = {}
    for canonical, entry in groups.items():
        aliases = list(entry['aliases'])
        if include_canonical:
            aliases.append(canonical)
        for alias in aliases:
            key = _alias_key(alias)
            if key in index and index[key] != canonical:
                raise ValueError(f'Conflicting label aliases: {alias!r}')
            index[key] = canonical
    return index


def _normalize(value, index):
    key = _alias_key(value)
    if key in index:
        return index[key], True
    return _token(value), False


def _tags(value):
    if _missing(value):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text[0] in '[(':
            try:
                parsed = json.loads(text) if text[0] == '[' else ast.literal_eval(text)
            except (ValueError, SyntaxError):
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    parsed = text
            if isinstance(parsed, (list, tuple, set)):
                return _tags(parsed)
        return [tag.strip() for tag in re.split(r'[;|]', text) if tag.strip()]
    if isinstance(value, dict):
        raise ValueError('Transformation tags must be scalar labels or sequences')
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        tags = []
        values = sorted(value, key=str) if isinstance(value, set) else value
        for item in values:
            tags.extend(_tags(item))
        return tags
    return [value]


def _audit(rows, normalized, changed, missing, mappings, unmapped, unmapped_rows):
    return {
        'rows': rows,
        'changed_rows': changed,
        'missing_rows': missing,
        'unmapped_rows': unmapped_rows,
        'mappings': [dict(raw=raw, normalized=canonical, count=int(count), context=context or None)
                     for (raw, canonical, context), count in sorted(mappings.items())],
        'unmapped': [dict(raw=raw, normalized=canonical, count=int(count))
                     for (raw, canonical), count in sorted(unmapped.items())],
        'canonical_counts': {key: int(value) for key, value in sorted(Counter(normalized).items())},
    }


def _raw_column(frame, column):
    raw = column + '_raw'
    if raw in frame:
        return frame[raw].tolist()
    if column in frame:
        return frame[column].tolist()
    return [None] * len(frame)


def normalize_labels(frame):
    registry = json.loads(REGISTRY_PATH.read_text())
    generators = _compile(registry['generators'])
    families = _compile(registry['families'])
    contexts = {key: _compile(value, include_canonical=False)
                for key, value in registry.get('dataset_generator_aliases', {}).items()}
    result = frame.copy(deep=True)
    datasets = frame['dataset'].map(_alias_key).tolist() if 'dataset' in frame else [''] * len(frame)
    columns = {}
    context_mappings = Counter()
    for column, index in [('generator', generators), ('manipulation', families)]:
        raw_values = _raw_column(frame, column)
        result[column + '_raw'] = raw_values
        normalized, changed, missing, unmapped_rows = [], 0, 0, 0
        mappings, unmapped = Counter(), Counter()
        for raw, dataset in zip(raw_values, datasets):
            canonical, known = _normalize(raw, index)
            context = ''
            if column == 'generator' and _alias_key(raw) in contexts.get(dataset, {}):
                canonical = contexts[dataset][_alias_key(raw)]
                known, context = True, dataset
                context_mappings[(dataset, _text(raw), canonical)] += 1
            normalized.append(canonical)
            changed += _text(raw) != canonical
            missing += canonical == 'unknown'
            mappings[(_text(raw), canonical, context)] += 1
            if not known:
                unmapped[(_text(raw), canonical)] += 1
                unmapped_rows += 1
        result[column] = normalized
        columns[column] = _audit(len(frame), normalized, int(changed), int(missing),
                                 mappings, unmapped, int(unmapped_rows))
    raw_transformations = _raw_column(frame, 'transformation')
    raw_tags = _raw_column(frame, 'transformation_tags')
    result['transformation_raw'] = raw_transformations
    result['transformation_tags_raw'] = raw_tags
    transformations, all_tags, changed, missing, unmapped_rows = [], [], 0, 0, 0
    mappings, unmapped, tag_counts = Counter(), Counter(), Counter()
    for raw, raw_extra in zip(raw_transformations, raw_tags):
        tags, row_unmapped, row_mappings = set(), set(), set()
        for value in _tags(raw) + _tags(raw_extra):
            canonical, known = _normalize(value, families)
            if canonical == 'unknown':
                continue
            tags.add(canonical)
            row_mappings.add((_text(value), canonical, ''))
            if not known:
                row_unmapped.add((_text(value), canonical))
        tags = sorted(tags)
        scalar = ';'.join(tags) if tags else 'unknown'
        transformations.append(scalar)
        all_tags.append(tags)
        changed += _text(raw) != scalar
        missing += not tags
        unmapped_rows += bool(row_unmapped)
        mappings.update(row_mappings)
        unmapped.update(row_unmapped)
        tag_counts.update(tags)
    result['transformation'] = transformations
    result['transformation_tags'] = all_tags
    columns['transformation'] = _audit(len(frame), transformations, int(changed), int(missing),
                                      mappings, unmapped, int(unmapped_rows))
    columns['transformation']['tag_face_counts'] = {key: int(value) for key, value in sorted(tag_counts.items())}
    columns['transformation']['missing_tags'] = []
    audit = {
        'registry_version': registry['registry_version'],
        'source_refs': registry.get('source_refs', {}),
        'columns': columns,
        'unmapped_values': int(sum(sum(row['count'] for row in value['unmapped']) for value in columns.values())),
        'context_aliases_applied': [dict(dataset=dataset, raw=raw, normalized=canonical, count=int(count))
                                    for (dataset, raw, canonical), count in sorted(context_mappings.items())],
    }
    result.attrs = {**frame.attrs, 'label_normalization': audit}
    return result
