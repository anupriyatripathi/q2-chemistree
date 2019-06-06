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
import numpy as np
import numpy.testing as npt
from biom.table import Table
from biom import load_table
from q2_qemistree import make_hierarchy
from q2_qemistree import CSIDirFmt

from q2_qemistree._collate_fingerprint import collate_fingerprint
from q2_qemistree._transformer import _read_dataframe
from q2_qemistree._hierarchy import jaccard_dm, _pdist_union


class TestHierarchy(TestCase):
    def setUp(self):
        THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        tablefp = Table({}, [], [])
        self.emptyfeatures = tablefp
        goodtable = os.path.join(THIS_DIR, 'data/feature_table1.biom')
        self.features = load_table(goodtable)
        goodfdata = os.path.join(THIS_DIR, 'data/feature_data1.txt')
        self.fdata = _read_dataframe(goodfdata)
        goodtable = os.path.join(THIS_DIR, 'data/feature_table2.biom')
        self.features2 = load_table(goodtable)
        goodfdata = os.path.join(THIS_DIR, 'data/feature_data2.txt')
        self.fdata2 = _read_dataframe(goodfdata)
        self.goodcsi = qiime2.Artifact.load(os.path.join(THIS_DIR,
                                                         'data/goodcsi1.qza'))
        goodcsi = self.goodcsi.view(CSIDirFmt)
        self.collated = collate_fingerprint(goodcsi)
        self.goodcsi2 = qiime2.Artifact.load(os.path.join(
                                            THIS_DIR, 'data/goodcsi2.qza'))
        goodcsi = self.goodcsi2.view(CSIDirFmt)
        self.collated2 = collate_fingerprint(goodcsi)

    def test_unequal_inputs1(self):
        goodcsi = self.goodcsi.view(CSIDirFmt)
        msg = ("The feature tables and CSI results should have a one-to-one"
               " correspondance.")
        with self.assertRaisesRegex(ValueError, msg):
            make_hierarchy([goodcsi], [self.features, self.features2],
                           [self.fdata, self.fdata2])

    def test_unequal_inputs2(self):
        goodcsi = self.goodcsi.view(CSIDirFmt)
        goodcsi2 = self.goodcsi2.view(CSIDirFmt)
        msg = ("The feature tables and feature data should have a one-to-one"
               " correspondance.")
        with self.assertRaisesRegex(ValueError, msg):
            make_hierarchy([goodcsi, goodcsi2],
                           [self.features, self.features2], [self.fdata])

    def test_mergeFeatureDataSingle(self):
        goodcsi1 = self.goodcsi.view(CSIDirFmt)
        treeout, merged_fts, merged_fdata = make_hierarchy(
            [goodcsi1], [self.features], [self.fdata])
        featrs = sorted(list(merged_fts.ids(axis='observation')))
        fdata_featrs = sorted(list(merged_fdata.index))
        self.assertEqual(len(featrs) == 5, True)
        self.assertEqual(fdata_featrs, featrs)

    def test_mergeFeatureDataMultiple(self):
        goodcsi1 = self.goodcsi.view(CSIDirFmt)
        goodcsi2 = self.goodcsi2.view(CSIDirFmt)
        treeout, merged_fts, merged_fdata = make_hierarchy(
            [goodcsi1, goodcsi2], [self.features, self.features2],
            [self.fdata, self.fdata2])
        featrs = sorted(list(merged_fts.ids(axis='observation')))
        fdata_featrs = sorted(list(merged_fdata.index))
        self.assertEqual(len(featrs) == 11, True)
        self.assertEqual(fdata_featrs, featrs)

    def test_emptyFeatures(self):
        goodcsi = self.goodcsi.view(CSIDirFmt)
        with self.assertRaises(ValueError):
            make_hierarchy([goodcsi], [self.emptyfeatures], [self.fdata])

    def test_tipMatchSingle(self):
        goodcsi = self.goodcsi.view(CSIDirFmt)
        treeout, feature_table, merged_fdata = make_hierarchy(
            [goodcsi], [self.features], [self.fdata])
        tip_names = {node.name for node in treeout.tips()}
        self.assertEqual(tip_names, set(feature_table._observation_ids))

    def test_Pipeline(self):
        goodcsi1 = self.goodcsi.view(CSIDirFmt)
        goodcsi2 = self.goodcsi2.view(CSIDirFmt)
        treeout, merged_fts, merged_fdata = make_hierarchy(
            [goodcsi1, goodcsi2], [self.features, self.features2],
            [self.fdata, self.fdata2])
        tip_names = {node.name for node in treeout.tips()}
        self.assertEqual(tip_names, set(merged_fts._observation_ids))


class TestDistances(TestCase):
    def test_pdist_union_identical(self):
        u = np.array([1, 1, 1, 0, 0, 0])
        v = np.array([1, 1, 1, 0, 0, 0])
        self.assertEqual(_pdist_union(u, v), 3.0)

    def test_pdist_union_non_overlapping(self):
        u = np.array([0, 1, 0, 1, 0])
        v = np.array([1, 0, 1, 0, 1])
        self.assertEqual(_pdist_union(u, v), 5.0)

    def test_pdist_union_all_same(self):
        u = np.array([1, 1, 1, 1, 1])
        v = np.array([1, 1, 1, 1, 1])
        self.assertEqual(_pdist_union(u, v), 5.0)

    def test_pdist_union_zero(self):
        u = np.array([0, 0, 0, 0, 0])
        v = np.array([0, 0, 0, 0, 0])
        self.assertEqual(_pdist_union(u, v), 0.0)

    def test_jaccard_dm_two_not_equal(self):
        x = np.array([[1, 0, 0, 0], [1, 1, 0, 0]])
        obs = jaccard_dm(x, np.array([33.2, 44.3]), tolerance=0.1)

        # the m/z are different hence + 1
        # j = (1 + 1) / (2 + 1) = 0.6666666
        npt.assert_almost_equal(obs, np.array([2.0/3.0]))

    def test_jaccard_dm_two_equal(self):
        x = np.array([[1, 0, 0, 0], [1, 1, 0, 0]])
        obs = jaccard_dm(x, np.array([33.2, 33.27]), tolerance=0.1)

        # the m/z are the same hence + 0
        # j = (1 + 0) / (2 + 1) = 0.3333333
        npt.assert_almost_equal(obs, np.array([1/3.0]))

    def test_jaccard_dm_two_identical(self):
        x = np.array([[1, 0, 1, 0], [1, 0, 1, 0]])
        obs = jaccard_dm(x, np.array([33.2, 33.2]), tolerance=0.1)

        # the m/z are the same hence + 0
        # j = (0 + 0) / (2 + 1) = 0.0
        npt.assert_almost_equal(obs, np.array([0.0]))

    def test_jaccard_dm_multiple_vectors(self):
        x = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ])
        y = np.array([33.2, 12, 11, 400, 33.23])

        obs = jaccard_dm(x, y, tolerance=0.2)

        #       | 0,   3/6, 1/6, 4/6, 0/6 |
        #       | 3/6,   0, 3/6, 2/4, 3/6 |
        # dm =  | 1/6, 3/6,   0, 4/6, 1/6 |
        #       | 4/6, 2/4, 4/6,   0, 4/6 |
        #       | 0/6, 3/6, 1/6, 4/6,   0 |
        # lower triangle in column-wise order
        exp = np.array([3/6, 1/6, 4/6, 0/6, 3/6, 2/4, 3/6, 4/6, 1/6, 4/6])

        npt.assert_almost_equal(obs, exp)


if __name__ == '__main__':
    main()
