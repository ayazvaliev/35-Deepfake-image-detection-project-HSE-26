import unittest

import pandas as pd

from src.preprocessing.splits import build_composite_splits


def population(partitions, inventory=None):
    rows = []
    for split, count in partitions:
        for index in range(count):
            number = len(rows)
            row = {'dataset': 'source', 'sample_id': f'row-{number}', 'source_group': f'group-{number}',
                   'target': index % 2, 'author_split': split, 'generator': f'method-{index % 4}',
                   'face_mask_fraction_of_crop': 0.5,
                   'source_format': 'JPEG' if index % 3 else 'PNG',
                   'yaw_bin': ['frontal', 'three_quarter', 'profile'][index % 3],
                   'light_bin': 'dark' if index % 2 else 'bright'}
            if inventory is not None:
                row['author_split_inventory'] = inventory
            rows.append(row)
    return pd.DataFrame(rows)


class AuthorSplitPolicyTests(unittest.TestCase):
    def assert_disjoint_complete(self, source, result):
        sets = [set(result[name]['sample_id']) for name in ['train', 'validate', 'test']]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(set.union(*sets), set(source['sample_id']))
        self.assertEqual(len(result['excluded']), 0)

    def test_three_native_parts_are_kept_exactly(self):
        source = population([('train', 100), ('valid', 30), ('test', 20)])
        result = build_composite_splits(source, split_policy="author_preserved")
        self.assertEqual([len(result[name]) for name in ['train', 'validate', 'test']], [100, 30, 20])
        self.assertEqual(set(result['validate']['author_split']), {'valid'})
        self.assertEqual(set(result['test']['author_split']), {'test'})
        self.assert_disjoint_complete(source, result)

    def test_train_test_derives_validation_only_from_train(self):
        source = population([('train', 100), ('test', 30)])
        result = build_composite_splits(source, author_validation_fraction=0.2, split_policy="author_preserved")
        self.assertEqual([len(result[name]) for name in ['train', 'validate', 'test']], [80, 20, 30])
        self.assertEqual(set(result['validate']['author_split']), {'train'})
        self.assertEqual(set(result['test']['author_split']), {'test'})
        self.assert_disjoint_complete(source, result)

    def test_train_valid_reserves_native_valid_for_final_test(self):
        source = population([('train', 100), ('valid', 30)])
        result = build_composite_splits(source, author_validation_fraction=0.2, split_policy="author_preserved")
        self.assertEqual([len(result[name]) for name in ['train', 'validate', 'test']], [80, 20, 30])
        self.assertEqual(set(result['validate']['author_split']), {'train'})
        self.assertEqual(set(result['test']['author_split']), {'valid'})
        self.assert_disjoint_complete(source, result)

    def test_unsplit_data_uses_seventy_fifteen_fifteen(self):
        source = population([('', 1000)])
        result = build_composite_splits(source, split_policy="author_preserved")
        counts = [len(result[name]) for name in ['train', 'validate', 'test']]
        for actual, expected in zip(counts, [700, 150, 150]):
            self.assertLessEqual(abs(actual - expected), 2)
        self.assert_disjoint_complete(source, result)

    def test_full_inventory_survives_sampling_that_loses_native_validation(self):
        source = population([('train', 100), ('test', 30)], ['train', 'validate', 'test'])
        result = build_composite_splits(source, split_policy="author_preserved")
        self.assertEqual([len(result[name]) for name in ['train', 'validate', 'test']], [100, 0, 30])
        policy = result['report']['author_split_policies']['source']
        self.assertEqual(policy['basis'], 'inventory')
        self.assertEqual(set(policy['source_partitions']), {'train', 'validate', 'test'})
        self.assert_disjoint_complete(source, result)


if __name__ == '__main__':
    unittest.main()
