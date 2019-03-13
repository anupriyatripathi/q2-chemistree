# ----------------------------------------------------------------------------
# Copyright (c) 2016-2018, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import TestCase, main
import os
from biom import load_table
from biom.table import Table

from q2_chemistree._collate_fingerprint import collate_fingerprint
from q2_chemistree._match import match_label


class TestMatch(TestCase):
    def setUp(self):
        THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        table = Table({}, [], [])
        self.emptyfps = table
        table = Table({}, ['a', 'b', 'c'], [])
        self.wrongtips = table
        goodtable = os.path.join(THIS_DIR, 'data/features_formated.biom')
        self.features = load_table(goodtable)
        goodcsi = os.path.join(THIS_DIR, 'data/goodcsi')
        self.tablefp = collate_fingerprint(goodcsi)

    def test_emptyTable(self):
        with self.assertRaises(ValueError):
            match_label(self.emptyfps, self.features)

    def test_tipMismatch(self):
        with self.assertRaises(ValueError):
            match_label(self.wrongtips, self.features)

    def test_matchPipeline(self):
        relabel_fps, matched_table = match_label(self.tablefp, self.features)
        featrs = set(matched_table.ids(axis='observation'))
        fps = set(relabel_fps.index)
        self.assertEqual(fps, featrs)


if __name__ == '__main__':
    main()
