import json
import unittest

import numpy as np
import pandas as pd

from src.preprocessing.labels import normalize_labels


class LabelNormalizationTests(unittest.TestCase):
    def test_stylegan_versions_remain_distinct_and_aliases_converge(self):
        frame = pd.DataFrame({'generator': ['StyleGAN', 'Style GAN', 'StyleGAN1',
                                           'StyleGAN2', 'Style.GAN-2', 'StyleGAN3', 'StyleGAN2-ADA']})
        result = normalize_labels(frame)
        self.assertEqual(result['generator'].tolist(),
                         ['stylegan', 'stylegan', 'stylegan', 'stylegan2', 'stylegan2', 'stylegan3', 'stylegan2_ada'])
        self.assertEqual(result['generator_raw'].tolist(), frame['generator'].tolist())

    def test_stargan_and_diffusion_methods_are_not_collapsed(self):
        frame = pd.DataFrame({'generator': ['StarGAN', 'StarGAN v2', 'DDIM', 'DDPM', 'RDDM']})
        self.assertEqual(normalize_labels(frame)['generator'].tolist(),
                         ['stargan', 'stargan_v2', 'ddim', 'ddpm', 'rddm'])

    def test_stable_diffusion_decimal_version_is_preserved(self):
        values = ['SD2.1', 'SD-2.1', 'SD_2_1', 'StableDiffusion2.1', 'Stable Diffusion v2.1',
                  'SD21', 'StableDiffusion2.0', 'Stable Diffusion 2.1 Base']
        result = normalize_labels(pd.DataFrame({'generator': values}))
        self.assertEqual(result['generator'].tolist()[:5], ['stable_diffusion_2_1'] * 5)
        self.assertEqual(result['generator'].iloc[5], 'sd21')
        self.assertNotEqual(result['generator'].iloc[6], 'stable_diffusion_2_1')
        self.assertNotEqual(result['generator'].iloc[7], 'stable_diffusion_2_1')
        self.assertEqual(result.attrs['label_normalization']['columns']['generator']['unmapped_rows'], 3)

    def test_faceswap_algorithm_is_separate_from_manipulation_family(self):
        result = normalize_labels(pd.DataFrame({'generator': ['FaceSwap'], 'manipulation': ['FaceSwap']}))
        self.assertEqual(result.loc[0, 'generator'], 'faceswap')
        self.assertEqual(result.loc[0, 'manipulation'], 'face_swap')

    def test_project_manipulation_aliases(self):
        values = ['EFS', 'EntireFaceSynthesis', 'entire_face_synthesis', 'FE', 'FaceEditing',
                  'FS', 'FR', 'reenactment', 'FSR', 'Super Resolution', 'FMPS', 'MFPS',
                  'manualPhotoshop', 'real', 'original', 'authentic', 'unlabeled', None, '']
        expected = ['face_synthesis'] * 3 + ['face_edit'] * 2 + ['face_swap',
                    'face_reenactment', 'face_reenactment', 'super_resolution', 'super_resolution',
                    'manual_edit', 'manual_edit', 'manual_edit', 'real', 'real', 'real',
                    'unknown', 'unknown', 'unknown']
        result = normalize_labels(pd.DataFrame({'manipulation': values}))
        self.assertEqual(result['manipulation'].tolist(), expected)

    def test_verified_df40_context_deduplicates_the_stylegan_pool(self):
        frame = pd.DataFrame({'dataset': ['DF40', '140k', 'other'],
                              'generator': ['WhichisReal', 'StyleGAN', 'WhichisReal'],
                              'native_method': ['whichfaceisreal', 'StyleGAN', 'WhichisReal']})
        result = normalize_labels(frame)
        self.assertEqual(result['generator'].tolist(), ['stylegan', 'stylegan', 'whichfaceisreal'])
        pd.testing.assert_series_equal(result['native_method'], frame['native_method'])
        applied = result.attrs['label_normalization']['context_aliases_applied']
        self.assertEqual(applied, [{'dataset': 'df40', 'raw': 'WhichisReal', 'normalized': 'stylegan', 'count': 1}])
        self.assertIn('whichfaceisreal_methods', result.attrs['label_normalization']['source_refs'])

    def test_unversioned_brands_are_not_globally_assigned_df40_variants(self):
        frame = pd.DataFrame({'dataset': ['DF40'] * 4 + ['other'] * 4,
                              'generator': ['PixArt', 'DiT', 'SiT', 'MidJourney'] * 2})
        result = normalize_labels(frame)
        self.assertEqual(result['generator'].tolist(), ['pixart_alpha', 'dit_xl_2', 'sit_xl_2', 'midjourney_v6',
                                                       'pixart', 'dit', 'sit', 'midjourney'])

    def test_all_current_df40_native_methods_have_known_canonical_models(self):
        values = ['VQGAN', 'StyleGAN2', 'StyleGAN3', 'StyleGANXL', 'sd2.1', 'ddim', 'RDDM',
                  'pixart', 'DiT', 'SiT', 'MidJourney', 'whichfaceisreal', 'CollabDiff',
                  'e4e', 'stargan', 'starganv2', 'styleclip']
        result = normalize_labels(pd.DataFrame({'dataset': ['DF40'] * len(values), 'generator': values}))
        self.assertFalse((result['generator'] == 'unknown').any())
        self.assertEqual(result.attrs['label_normalization']['columns']['generator']['unmapped'], [])
        self.assertEqual(result['generator'].nunique(), 17)

    def test_optional_multilabel_tags_are_normalized_without_losing_distinct_tags(self):
        raw = [['FS', 'Face Swap', 'smile', 'hair colour'], np.array(['FE', 'age']), None]
        frame = pd.DataFrame({'transformation': raw})
        result = normalize_labels(frame)
        self.assertEqual(result['transformation_tags'].tolist(), [['face_swap', 'hair_colour', 'smile'], ['age', 'face_edit'], []])
        self.assertEqual(result['transformation'].iloc[0], 'face_swap;hair_colour;smile')
        self.assertEqual(result['transformation_raw'].iloc[0], raw[0])
        np.testing.assert_array_equal(result['transformation_raw'].iloc[1], raw[1])
        self.assertEqual(result['transformation'].iloc[2], 'unknown')
        self.assertEqual(result.attrs['label_normalization']['columns']['transformation']['tag_face_counts']['face_swap'], 1)

    def test_json_tags_and_explicit_tag_column_preserve_raw_inputs(self):
        frame = pd.DataFrame({'transformation': ['["FS", "FR"]', 'FS;FR'],
                              'transformation_tags': [['FSR', 'smile'], ['FSR', 'smile']]})
        result = normalize_labels(frame)
        expected = ['face_reenactment', 'face_swap', 'smile', 'super_resolution']
        self.assertEqual(result['transformation_tags'].tolist(), [expected, expected])
        self.assertEqual(result['transformation_tags_raw'].tolist(), frame['transformation_tags'].tolist())
        self.assertEqual(result['transformation_raw'].tolist(), frame['transformation'].tolist())

    def test_new_labels_remain_distinct_and_are_reported_as_unmapped(self):
        result = normalize_labels(pd.DataFrame({'generator': ['New Model v7', 'New Model v8', 'GAN'],
                                               'transformation_tags': [['smile', 'age'], [], []]}))
        self.assertEqual(result['generator'].tolist(), ['new_model_v7', 'new_model_v8', 'gan'])
        audit = result.attrs['label_normalization']
        self.assertEqual(audit['columns']['generator']['unmapped_rows'], 3)
        self.assertEqual(audit['columns']['generator']['unmapped'][0]['count'], 1)
        self.assertGreater(audit['unmapped_values'], 3)
        json.dumps(audit)

    def test_target_identifiers_and_native_taxonomy_are_unchanged(self):
        frame = pd.DataFrame({'target': pd.Series([0, 1, None], dtype='Int64'),
                              'generator': ['real', 'StyleGAN2', None], 'native_method': ['raw A', 'raw B', 'raw C'],
                              'native_family': ['real', 'EFS', 'FE'], 'sample_id': ['a', 'b', 'c'],
                              'source_group': ['g1', 'g2', 'g3'], 'author_split': ['train', 'val', 'test'],
                              'path': ['/a', '/b', '/c']})
        original = frame.copy(deep=True)
        result = normalize_labels(frame)
        for column in ['target', 'native_method', 'native_family', 'sample_id', 'source_group', 'author_split', 'path']:
            pd.testing.assert_series_equal(result[column], frame[column])
        pd.testing.assert_frame_equal(frame, original)

    def test_normalization_is_idempotent_and_preserves_prior_attributes(self):
        frame = pd.DataFrame({'dataset': ['DF40', '140k'], 'generator': ['WhichisReal', 'StyleGAN'],
                              'manipulation': ['EFS', 'EFS'], 'transformation_tags': [['FSR', 'smile'], []]})
        frame.attrs['sampling'] = {'seed': 42}
        first = normalize_labels(frame)
        second = normalize_labels(first)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.attrs, second.attrs)
        self.assertEqual(first.attrs['sampling'], {'seed': 42})

    def test_missing_fine_tags_are_not_invented_from_native_paths(self):
        frame = pd.DataFrame({'generator': ['e4e'], 'manipulation': ['FE'],
                              'path': ['e4e/ff/000/inversions/123.jpg'], 'native_method': ['e4e']})
        result = normalize_labels(frame)
        self.assertEqual(result.loc[0, 'transformation_tags'], [])
        self.assertEqual(result.loc[0, 'transformation'], 'unknown')
        self.assertEqual(result.loc[0, 'manipulation'], 'face_edit')

    def test_empty_frame_has_a_valid_serializable_audit(self):
        result = normalize_labels(pd.DataFrame(columns=['target']))
        self.assertTrue(result.empty)
        self.assertEqual(result.attrs['label_normalization']['columns']['generator']['rows'], 0)
        json.dumps(result.attrs['label_normalization'])


if __name__ == '__main__':
    unittest.main()
