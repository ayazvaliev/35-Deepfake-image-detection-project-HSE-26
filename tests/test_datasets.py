import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.download_assets import extract_mffi_selection, selection_ready
from src.preprocessing.datasets import index_140k, index_mffi, index_openforensics, read_mffi_labels


class DatasetAdapterTests(unittest.TestCase):
    def test_actual_mffi_phase1_header_and_binary_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'trainset_label.txt'
            path.write_text('img_name,target\na.jpg,1\nb.jpg,0\n')
            labels = read_mffi_labels(path)
            self.assertEqual(labels['target'].tolist(), [1, 0])
            self.assertEqual(labels['generator'].tolist(), ['unknown', 'real'])
            path.write_text('a.jpg,2\n')
            with self.assertRaisesRegex(ValueError, 'verified mapping'):
                read_mffi_labels(path)

    def test_mffi_same_filename_keeps_its_author_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for phase, directory, label, target in [('phase1', 'trainset', 'trainset_label.txt', 1),
                                                     ('phase2', 'test', 'testset_label.txt', 0)]:
                folder = root / phase / directory
                folder.mkdir(parents=True)
                (folder / 'same.jpg').write_bytes(phase.encode())
                (folder.parent / label).write_text(f'same.jpg,{target}\n')
            frame = index_mffi(root)
            self.assertEqual(len(frame), 2)
            train = frame[frame['author_split'] == 'train'].iloc[0]
            test = frame[frame['author_split'] == 'test'].iloc[0]
            self.assertIn('/phase1/trainset/', train['path'])
            self.assertIn('/phase2/test/', test['path'])
            self.assertEqual((train['target'], test['target']), (1, 0))
            self.assertEqual(train['author_split_inventory'], ['train', 'test'])

    def test_mffi_never_substitutes_a_test_file_for_missing_train(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'phase1').mkdir()
            (root / 'phase1/trainset_label.txt').write_text('same.jpg,1\n')
            (root / 'phase2/test').mkdir(parents=True)
            (root / 'phase2/test/same.jpg').write_bytes(b'test')
            with self.assertRaisesRegex(FileNotFoundError, 'author split'):
                index_mffi(root)

    def test_openforensics_mixed_faces_and_full_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'annotations').mkdir()
            (root / 'images/Images/Train').mkdir(parents=True)
            (root / 'images/Images/Train/same.jpg').write_bytes(b'image')
            for split in ['Train', 'Val', 'Test-Dev']:
                payload = {'categories': [{'id': 0, 'name': 'Real'}, {'id': 1, 'name': 'Fake'}],
                           'images': [{'id': 1, 'file_name': f'Images/{split}/same.jpg', 'width': 100, 'height': 100}],
                           'annotations': [{'id': 1, 'image_id': 1, 'category_id': 0, 'bbox': [0, 0, 20, 20]},
                                           {'id': 2, 'image_id': 1, 'category_id': 1, 'bbox': [50, 50, 20, 20]}]}
                (root / 'annotations' / f'{split}_poly.json').write_text(json.dumps(payload))
            frame = index_openforensics(root)
            self.assertEqual(len(frame), 1)
            row = frame.iloc[0]
            self.assertTrue(pd.isna(row['target']))
            self.assertEqual([item['target'] for item in row['annotations']], [0, 1])
            self.assertEqual(row['frame_composition'], 'mixed')
            self.assertEqual(row['author_split_inventory'], ['train', 'validate', 'test'])

    def test_140k_folder_labels_override_inverted_csv_convention(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ['real', 'fake']:
                folder = root / 'real_vs_fake/real-vs-fake/train' / name
                folder.mkdir(parents=True)
                (folder / 'image.jpg').write_bytes(b'image')
            frame = index_140k(root)
            self.assertEqual(frame.set_index('generator')['target'].to_dict(), {'StyleGAN': 1, 'real': 0})
            self.assertEqual(frame.iloc[0]['author_split_inventory'], ['train', 'validate', 'test'])

    def test_mffi_tar_selection_uses_split_and_filename(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'phase1.tar.gz'
            with tarfile.open(archive, 'w:gz') as stream:
                for split in ['trainset', 'valset']:
                    content = split.encode()
                    member = tarfile.TarInfo(f'phase1/{split}/same.jpg')
                    member.size = len(content)
                    stream.addfile(member, io.BytesIO(content))
            selected = pd.DataFrame({'author_split': ['valid'], 'filename': ['same.jpg']})
            files = extract_mffi_selection(archive, root / 'raw', selected)
            self.assertFalse((root / 'raw/phase1/trainset/same.jpg').exists())
            self.assertEqual((root / 'raw/phase1/valset/same.jpg').read_bytes(), b'valset')
            self.assertEqual(len(files[0]['sha256']), 64)

    def test_selection_marker_rejects_another_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / 'selection.json'
            marker.write_text(json.dumps({'seed': 42, 'requested_limit': 20, 'files': []}))
            with self.assertRaisesRegex(RuntimeError, 'another seed or limit'):
                selection_ready(marker, 20, 7, root)


if __name__ == '__main__':
    unittest.main()
