from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.datasets import index_140k, index_mffi, index_openforensics, sample_manifest, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['140k', 'mffi', 'openforensics'], required=True)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--limit', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    frame = {'140k': index_140k, 'mffi': index_mffi, 'openforensics': index_openforensics}[args.dataset](args.root)
    strata = ['author_split', 'frame_composition', 'n_faces'] if args.dataset == 'openforensics' else ['author_split', 'target']
    selected = sample_manifest(frame, args.limit, strata, args.seed)
    write_jsonl(selected, args.output)
    args.output.with_suffix('.sampling.json').write_text(json.dumps(selected.attrs['sampling'], indent=2))
    print(f'{args.dataset}: {len(frame):,} images available, {len(selected):,} selected')


if __name__ == '__main__':
    main()
