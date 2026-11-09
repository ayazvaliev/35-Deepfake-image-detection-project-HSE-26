import unittest

import numpy as np
import pandas as pd

from src.preprocessing.splits import build_composite_splits


BIN = 'face_mask_fraction_of_crop_bin'
VALUE = 'face_mask_fraction_of_crop'


def manifest(values):
    return pd.DataFrame([{'dataset': 'example', 'sample_id': f'image-{index}', 'target': index % 2,
                          'source_group': f'group-{index}', VALUE: value}
                         for index, value in enumerate(values)])


def combined(result):
    return pd.concat([result[name] for name in ['train', 'validate', 'test', 'excluded']],
                     ignore_index=True).sort_values('sample_id').set_index('sample_id')


class FaceMaskCropBalanceTests(unittest.TestCase):
    def test_quarter_bins_include_zero_and_the_closed_upper_boundary(self):
        source = manifest([0, 0.2499, 0.25, 0.4999, 0.5, 0.7499, 0.75, 1, None, np.nan])
        result = build_composite_splits(source)
        rows = combined(result).loc[source['sample_id']]
        self.assertEqual(rows[BIN].tolist(), ['[0,0.25)', '[0,0.25)', '[0.25,0.5)', '[0.25,0.5)',
                                             '[0.5,0.75)', '[0.5,0.75)', '[0.75,1]', '[0.75,1]',
                                             'unknown', 'unknown'])
        self.assertEqual(result['report']['face_mask_crop_occupancy']['known_rows'], 8)
        self.assertEqual(result['report']['face_mask_crop_occupancy']['unknown_rows'], 2)

    def test_legacy_bounding_boxes_cannot_substitute_for_missing_face_masks(self):
        source = manifest([None] * 8).drop(columns=VALUE)
        source['bbox'] = [[0, 0, 100, 100]] * len(source)
        source['crop_bounds'] = [[0, 0, 100, 100]] * len(source)
        result = build_composite_splits(source)
        rows = combined(result)
        self.assertTrue(rows[VALUE].isna().all())
        self.assertEqual(set(rows[BIN]), {'unknown'})
        self.assertNotIn('bbox_fraction_of_crop', rows)
        self.assertNotIn('bbox_fraction_of_crop_bin', rows)

    def test_invalid_occupancy_values_are_unknown(self):
        source = manifest([-0.01, 1.01, np.inf, -np.inf, 'invalid', pd.NA])
        result = build_composite_splits(source)
        rows = combined(result)
        self.assertTrue(rows[VALUE].isna().all())
        self.assertEqual(set(rows[BIN]), {'unknown'})

    def test_bins_are_recomputed_from_current_mask_measurements(self):
        source = manifest([0.1, 0.9])
        source[BIN] = ['stale', 'stale']
        rows = combined(build_composite_splits(source))
        self.assertEqual(rows.loc['image-0', BIN], '[0,0.25)')
        self.assertEqual(rows.loc['image-1', BIN], '[0.75,1]')

    def test_mask_occupancy_is_balanced_within_dataset_and_target_in_all_three_parts(self):
        records = []
        for dataset in ['first', 'second']:
            for target in [0, 1]:
                for value in [0.1, 0.35, 0.6, 0.9, None]:
                    for repeat in range(40):
                        index = len(records)
                        records.append({'dataset': dataset, 'sample_id': f'image-{index}',
                                        'target': target, 'source_group': f'group-{index}', VALUE: value,
                                        'generator': 'StyleGAN2' if target else 'real',
                                        'manipulation': 'EFS' if target else 'real'})
        source = pd.DataFrame(records)
        result = build_composite_splits(source, seed=19)
        for partition, expected in [('train', 336), ('validate', 72), ('test', 72)]:
            self.assertLessEqual(abs(len(result[partition]) - expected), 2)
        rows = [row for row in result['report']['balance'] if row['feature'] == BIN]
        self.assertEqual(len(rows), 12)
        self.assertEqual({tuple(sorted(row['condition'])) for row in rows}, {('dataset', 'target')})
        self.assertEqual({row['category'] for row in rows},
                         {'[0.25,0.5)', '[0.5,0.75)', '[0.75,1]'})
        self.assertLessEqual(max(row['max_pairwise_gap'] for row in rows), 0.04)
        for row in rows:
            self.assertTrue(all(row[name + '_n'] > 0 for name in ['train', 'validate', 'test']))
        self.assertFalse(any(result['report']['overlap_groups'].values()))
        self.assertEqual(len(result['excluded']), 320)
        small = result['excluded'][VALUE].eq(0.1)
        missing = result['excluded'][VALUE].isna()
        self.assertEqual(int(small.sum()), 160)
        self.assertEqual(int(missing.sum()), 160)
        self.assertTrue(result['excluded'].loc[small, 'flags'].map(lambda flags: 'face_mask_too_small' in flags).all())
        self.assertTrue(result['excluded'].loc[missing, 'flags'].map(lambda flags: 'face_mask_missing' in flags).all())
        variant = source.copy()
        training_ids = set(result['train']['sample_id'])
        variant['mask_source'] = np.where(variant['sample_id'].isin(training_ids), 'annotation', 'mediapipe_face_oval')
        without_source_balance = build_composite_splits(variant, seed=19)
        for partition in ['train', 'validate', 'test']:
            self.assertEqual(set(result[partition]['sample_id']), set(without_source_balance[partition]['sample_id']))
        self.assertFalse(any(row['feature'] == 'mask_source' for row in without_source_balance['report']['balance']))

    def test_small_known_masks_are_excluded_at_the_inclusive_threshold_from_all_parts(self):
        records = []
        for split in ['train', 'valid', 'test']:
            for fraction in [0, 0.25, 0.25001, None]:
                index = len(records)
                records.append({'dataset': 'example', 'sample_id': f'image-{index}', 'target': index % 2,
                                'source_group': f'group-{index}', 'author_split': split,
                                'eligible': True, 'flags': ['existing_audit_flag'], 'generator': 'unknown', VALUE: fraction})
        records.append({'dataset': 'example', 'sample_id': 'already-excluded', 'target': 0,
                        'source_group': 'already-excluded', 'author_split': 'train',
                        'eligible': False, 'flags': ['earlier_exclusion'], VALUE: 0.25})
        result = build_composite_splits(pd.DataFrame(records))
        for partition in ['train', 'validate', 'test']:
            kept = result[partition]
            self.assertEqual(len(kept), 1)
            self.assertEqual(kept[VALUE].tolist(), [0.25001])
            self.assertEqual(kept['generator'].tolist(), ['unknown'])
            self.assertTrue((kept[VALUE].isna() | kept[VALUE].gt(0.25)).all())
            self.assertTrue(kept['flags'].map(lambda flags: flags == ['existing_audit_flag']).all())
        excluded = result['excluded'].set_index('sample_id')
        self.assertEqual(len(excluded), 10)
        small = excluded[VALUE].notna()
        self.assertTrue(excluded.loc[small, 'flags'].map(lambda flags: 'face_mask_too_small' in flags).all())
        self.assertTrue(excluded.loc[~small, 'flags'].map(lambda flags: 'face_mask_missing' in flags).all())
        self.assertTrue(excluded.loc[small, VALUE].le(0.25).all())
        self.assertTrue(excluded['exclusion_flags'].map(
            lambda flags: bool({'face_mask_too_small', 'face_mask_missing'} & set(flags))).all())
        self.assertFalse(excluded['eligible'].any())
        self.assertEqual(set(excluded.at['already-excluded', 'flags']),
                         {'earlier_exclusion', 'face_mask_too_small'})
        self.assertTrue(excluded.drop(index='already-excluded')['flags'].map(
            lambda flags: 'existing_audit_flag' in flags).all())
        audit = result['report']['face_mask_crop_occupancy']
        self.assertEqual(audit['excluded_small_mask_rows'], 7)
        self.assertEqual(audit['exclusion_threshold_inclusive'], 0.25)
        self.assertTrue(audit['unknown_excluded_by_occupancy'])
        self.assertEqual(audit['excluded_missing_mask_rows'], 3)
        self.assertEqual(result['report']['initially_ineligible'], 10)

    def test_mask_sources_remain_distinct_and_bbox_coverage_stays_a_diagnostic(self):
        records = []
        for split in ['train', 'valid', 'test']:
            for mask_source, fraction in [('annotation', 0.4), ('mediapipe_face_oval', 0.4), ('missing', None)]:
                for repeat in range(4):
                    index = len(records)
                    records.append({'dataset': 'example', 'sample_id': f'image-{index}', 'target': repeat % 2,
                                    'source_group': f'group-{index}', 'author_split': split,
                                    VALUE: fraction, 'mask_source': mask_source,
                                    'face_coverage_fraction': 0.5})
        result = build_composite_splits(pd.DataFrame(records))
        rows = combined(result)
        self.assertTrue(rows['face_coverage_fraction'].eq(0.5).all())
        self.assertNotIn('face_coverage_fraction_bin', rows)
        features = {row['feature'] for row in result['report']['balance']}
        self.assertNotIn('face_coverage_fraction', features)
        self.assertNotIn('face_coverage_fraction_bin', features)
        self.assertNotIn('mask_source', features)
        sources = {row['mask_source'] for row in result['report']['mask_source_coverage']}
        self.assertEqual(sources, {'annotation', 'mediapipe_face_oval', 'missing'})
        self.assertEqual(sum(row['rows'] for row in result['report']['mask_source_coverage']), len(records))
        self.assertFalse(result['report']['face_mask_crop_occupancy']['mask_source_balanced'])
        self.assertIn(BIN, features)


if __name__ == '__main__':
    unittest.main()
