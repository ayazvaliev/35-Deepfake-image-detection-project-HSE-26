from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.datasets import index_140k, index_df40, index_mffi, index_openforensics, sample_manifest, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['df40', '140k', 'mffi', 'openforensics'], required=True)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--limit', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--require-count', action='store_true')
    parser.add_argument('--mffi-test-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    if args.mffi_test_only and args.dataset != 'mffi':
        parser.error('--mffi-test-only applies only to MFFI')
    root = args.root
    if args.mffi_test_only:
        from src.preprocessing.datasets import RAW

        root = (root or RAW / 'mffi') / 'phase2'
    frame = {'df40': index_df40, '140k': index_140k, 'mffi': index_mffi, 'openforensics': index_openforensics}[args.dataset](root)
    if args.dataset == 'df40':
        strata = [name for name in ('native_method', 'target', 'generator', 'manipulation',
                                     'domain', 'author_split', 'source_format') if name in frame]
    else:
        strata = ['author_split', 'frame_composition', 'n_faces'] if args.dataset == 'openforensics' else ['author_split', 'target']
    selected = sample_manifest(frame, args.limit, strata, args.seed)
    if args.require_count and len(selected) != args.limit:
        raise RuntimeError(f'{args.dataset}: expected {args.limit:,} images, found {len(selected):,}')
    if args.output.suffix.lower() == '.parquet':
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + '.tmp')
        selected.to_parquet(temporary, index=False)
        temporary.replace(args.output)
    else:
        write_jsonl(selected, args.output)
    args.output.with_suffix('.sampling.json').write_text(json.dumps(selected.attrs['sampling'], indent=2))
    print(f'{args.dataset}: {len(frame):,} images available, {len(selected):,} selected')


if __name__ == '__main__':
    main()
