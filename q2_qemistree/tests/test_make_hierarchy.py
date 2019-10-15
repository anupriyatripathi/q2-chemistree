# ----------------------------------------------------------------------------
# Copyright (c) 2016-2018, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import TestCase, main
import os
import qiime2
import pandas as pd
from biom.table import Table
from biom import load_table
from q2_qemistree import make_hierarchy
from q2_qemistree import CSIDirFmt

from q2_qemistree._hierarchy import merge_feature_data


class TestHierarchy(TestCase):
    def setUp(self):
        THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        tablefp = Table({}, [], [])
        self.emptyfeatures = tablefp
        goodtable = os.path.join(THIS_DIR, 'data/features_formated.biom')
        self.features = load_table(goodtable)
        goodtable = os.path.join(THIS_DIR, 'data/features2_formated.biom')
        self.features2 = load_table(goodtable)
        self.goodcsi = qiime2.Artifact.load(os.path.join(THIS_DIR,
                                                         'data/csiFolder.qza'))
        self.goodcsi2 = qiime2.Artifact.load(os.path.join(
                                            THIS_DIR, 'data/csiFolder2.qza'))

    def test_unequal_inputs(self):
        goodcsi = self.goodcsi.view(CSIDirFmt)
        msg = ("The feature tables and CSI results should have a one-to-one"
               " correspondance.")
        with self.assertRaisesRegex(ValueError, msg):
            make_hierarchy([goodcsi], [self.features, self.features2])

    def test_mergeFeatureDataSingle(self):
        goodcsi1 = self.goodcsi.view(CSIDirFmt)
        treeout, merged_fts, merged_fps, merged_fdata = make_hierarchy(
            [goodcsi1], [self.features])
        featrs = sorted(list(merged_fts.ids(axis='observation')))
        fdata_featrs = sorted(list(merged_fdata.index))
        self.assertEqual(len(featrs) == 3, True)
        self.assertEqual(fdata_featrs, featrs)

    def test_mergeFeatureDataMultiple(self):
        goodcsi1 = self.goodcsi.view(CSIDirFmt)
        goodcsi2 = self.goodcsi2.view(CSIDirFmt)
        treeout, merged_fts, merged_fps, merged_fdata = make_hierarchy(
            [goodcsi1, goodcsi2], [self.features, self.features2])
        featrs = sorted(list(merged_fts.ids(axis='observation')))
        fdata_featrs = sorted(list(merged_fdata.index))
        self.assertEqual(len(featrs) == 9, True)
        self.assertEqual(fdata_featrs, featrs)

    def test_FeatureDataMultipleRepeated(self):
        fdata1 = pd.DataFrame(index=list('aabbc'),
                              data=[['1', '1'], ['1', '2'], ['1', '3'],
                              ['1', '4'], ['1', '5']],
                              columns=['table_number', '#featureID'])
        fdata2 = pd.DataFrame(index=list('accdef'),
                              data=[['2', '1'], ['2', '2'], ['2', '3'],
                              ['2', '4'], ['2', '5'], ['2', '6']],
                              columns=['table_number', '#featureID'])
        merged_fdata = merge_feature_data([fdata1, fdata2])
        fdata_featrs = sorted(list(merged_fdata.index))
        self.assertEqual(merged_fdata.loc['a', 'table_number'], '1,1,2')
        self.assertEqual(merged_fdata.loc['c', 'table_number'], '1,2,2')
        self.assertEqual(fdata_featrs, ['a', 'b', 'c', 'd', 'e', 'f'])

    def test_emptyFeatures(self):
        goodcsi = self.goodcsi.view(CSIDirFmt)
        with self.assertRaises(ValueError):
            make_hierarchy([goodcsi], [self.emptyfeatures])

    def test_tipMatchSingle(self):
        goodcsi = self.goodcsi.view(CSIDirFmt)
        treeout, merged_fts, merged_fps, merged_fdata = make_hierarchy(
            [goodcsi], [self.features])
        tip_names = {node.name for node in treeout.tips()}
        self.assertEqual(tip_names, set(merged_fts._observation_ids))

    def test_Pipeline(self):
        goodcsi1 = self.goodcsi.view(CSIDirFmt)
        goodcsi2 = self.goodcsi2.view(CSIDirFmt)
        treeout, merged_fts, merged_fps, merged_fdata = make_hierarchy(
            [goodcsi1, goodcsi2], [self.features, self.features2])
        tip_names = {node.name for node in treeout.tips()}
        self.assertEqual(tip_names, set(merged_fts._observation_ids))


if __name__ == '__main__':
    main()
